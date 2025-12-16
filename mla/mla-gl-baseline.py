# mla_baseline_gluon.py
import math
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# ---------------------------
# Guards
# ---------------------------

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


# ---------------------------
# Reference MLA (uses compression)
# ---------------------------
# Shapes:
#   q:   [Z, H, M, D]
#   c:   [Z, N, DC]         (compressed KV latent)
#   w_k: [H, DC, D]         (up-projection for K)
#   w_v: [H, DC, D]         (up-projection for V)
#
# Computes:
#   k[z,h,n,d] = sum_r c[z,n,r] * w_k[h,r,d]
#   v[z,h,n,d] = sum_r c[z,n,r] * w_v[h,r,d]
#   out = SDPA(q, k, v)
def ref_mla(q, c, w_k, w_v, *, causal: bool, sm_scale: float):
    # (Z, N, DC) x (H, DC, D) -> (Z, H, N, D)
    k = torch.einsum("znr,hrd->zhnd", c, w_k)
    v = torch.einsum("znr,hrd->zhnd", c, w_v)
    # PyTorch SDPA expects [Z, H, M, D] and [Z, H, N, D]
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        is_causal=causal,
        scale=sm_scale,
    )
    return out


# ---------------------------
# Baseline Gluon Attention Kernel (no TMA / no TMEM)
# ---------------------------
# Computes O = softmax(QK^T * sm_scale) V
# q, k, v are full (already "up-projected" outside the kernel).
#
# Launch shape: one program handles (z,h, block_m)
@gluon.jit
def attn_fwd_baseline_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    Z: gl.int32, H: gl.int32, M: gl.int32, N: gl.int32,
    D: gl.constexpr,
    stride_qz: gl.int32, stride_qh: gl.int32, stride_qm: gl.int32,
    stride_kz: gl.int32, stride_kh: gl.int32, stride_kn: gl.int32,
    stride_vz: gl.int32, stride_vh: gl.int32, stride_vn: gl.int32,
    stride_oz: gl.int32, stride_oh: gl.int32, stride_om: gl.int32,
    sm_scale: gl.float32,
    CAUSAL: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    pid = gl.program_id(0)

    # Linearize: pid -> (z, h, pid_m)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_zh = pid // num_pid_m
    z = pid_zh // H
    h = pid_zh - z * H  # pid_zh % H, but keep in tensor form

    start_m = pid_m * BLOCK_M
    offs_m = start_m + gl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # q: [BLOCK_M, D]
    offs_d = gl.arange(0, D)
    q_base = q_ptr + z * stride_qz + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    q = gl.load(q_base, mask=mask_m[:, None], other=0.0).to(gl.float32)

    # Online softmax state
    # m_i: [BLOCK_M], l_i: [BLOCK_M], o: [BLOCK_M, D]
    m_i = gl.full([BLOCK_M], -float("inf"), gl.float32)
    l_i = gl.full([BLOCK_M], 0.0, gl.float32)
    o = gl.full([BLOCK_M, D], 0.0, gl.float32)

    # Use exp2 for stability: exp(x) = exp2(x * log2(e))
    qk_scale = sm_scale * 1.4426950408889634  # log2(e)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + gl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # k,v blocks: [BLOCK_N, D]
        k_base = k_ptr + z * stride_kz + h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :]
        v_base = v_ptr + z * stride_vz + h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :]

        k = gl.load(k_base, mask=mask_n[:, None], other=0.0).to(gl.float32)
        v = gl.load(v_base, mask=mask_n[:, None], other=0.0).to(gl.float32)

        # scores = q @ k^T : [BM, BN]
        # naive reduction over D in chunks
        scores = gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32)
        CHUNK: gl.constexpr = 16
        for d0 in gl.static_range(0, D, CHUNK):
            qk_q = q[:, d0:d0 + CHUNK]                            # [BM, CHUNK]
            qk_k = k[:, d0:d0 + CHUNK]                            # [BN, CHUNK]
            # accumulate: sum_c q[m,c] * k[n,c]
            # do it as BN outer products over CHUNK
            # scores[m,n] += sum_c q[m,c] * k[n,c]
            for c in gl.static_range(0, CHUNK):
                scores += qk_q[:, c:c+1] * qk_k[:, c:c+1].T

        scores = scores * qk_scale

        # Apply masks (padding + causal)
        if CAUSAL:
            # Assume self-attn style causal: key index <= query index
            q_pos = offs_m[:, None]
            k_pos = offs_n[None, :]
            causal_mask = k_pos <= q_pos
            valid = causal_mask & (mask_m[:, None] & mask_n[None, :])
        else:
            valid = mask_m[:, None] & mask_n[None, :]

        scores = gl.where(valid, scores, -float("inf"))

        # Online softmax update:
        # m_new = max(m_old, rowmax(scores))
        rowmax = gl.max(scores, axis=1)                 # [BM]
        m_new = gl.maximum(m_i, rowmax)

        # alpha = exp(m_old - m_new)
        alpha = gl.exp2(m_i - m_new)                    # [BM]
        # p = exp(scores - m_new[:,None])
        p = gl.exp2(scores - m_new[:, None])            # [BM, BN]

        # l_new = l_old * alpha + sum(p)
        l_new = l_i * alpha + gl.sum(p, axis=1)

        # o_new = o_old * alpha[:,None] + p @ v
        o = o * alpha[:, None]
        # accumulate p@v with a BN loop (keeps it “baseline” and simple)
        for n in gl.static_range(0, BLOCK_N):
            o += p[:, n:n+1] * v[n:n+1, :]

        m_i = m_new
        l_i = l_new

    # Normalize
    o = o / l_i[:, None]

    # Store
    o_base = o_ptr + z * stride_oz + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    gl.store(o_base, o.to(gl.float16), mask=mask_m[:, None])


def _launch_attn(q, k, v, *, causal: bool, sm_scale: float, BLOCK_M=16, BLOCK_N=16):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    Z, H, M, D = q.shape
    _, _, N, Dk = k.shape
    assert D == Dk and v.shape == (Z, H, N, D)

    # Ensure contiguous last-dim
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty((Z, H, M, D), device=q.device, dtype=torch.float16)

    # Strides in elements (not bytes)
    stride_qz, stride_qh, stride_qm = q.stride(0), q.stride(1), q.stride(2)
    stride_kz, stride_kh, stride_kn = k.stride(0), k.stride(1), k.stride(2)
    stride_vz, stride_vh, stride_vn = v.stride(0), v.stride(1), v.stride(2)
    stride_oz, stride_oh, stride_om = o.stride(0), o.stride(1), o.stride(2)

    num_pid_m = triton.cdiv(M, BLOCK_M)
    grid = (Z * H * num_pid_m,)

    attn_fwd_baseline_kernel[grid](
        q, k, v, o,
        Z, H, M, N,
        D=D,
        stride_qz=stride_qz, stride_qh=stride_qh, stride_qm=stride_qm,
        stride_kz=stride_kz, stride_kh=stride_kh, stride_kn=stride_kn,
        stride_vz=stride_vz, stride_vh=stride_vh, stride_vn=stride_vn,
        stride_oz=stride_oz, stride_oh=stride_oh, stride_om=stride_om,
        sm_scale=sm_scale,
        CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return o


# ---------------------------
# MLA forward "baseline"
# (projection outside kernel, attention inside kernel)
# ---------------------------
def mla_forward_baseline(q, c, w_k, w_v, *, causal: bool, sm_scale: float, BLOCK_M=16, BLOCK_N=16):
    k = torch.einsum("znr,hrd->zhnd", c, w_k).contiguous()
    v = torch.einsum("znr,hrd->zhnd", c, w_v).contiguous()
    return _launch_attn(q, k, v, causal=causal, sm_scale=sm_scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)


# ---------------------------
# PyTest
# ---------------------------
@pytest.mark.skipif(not is_blackwell(), reason="Gluon attention is only supported on Blackwell GPUs")
@pytest.mark.parametrize("Z", [1, 2])
@pytest.mark.parametrize("H", [2, 8])
@pytest.mark.parametrize("N_CTX", [128, 256, 512])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("DC", [32, 64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mla_baseline_matches_ref(Z, H, N_CTX, D, DC, causal, dtype):
    device = "cuda"
    torch.manual_seed(0)

    # q: [Z,H,M,D], self-attn so M=N_CTX, N=N_CTX
    q = torch.randn((Z, H, N_CTX, D), device=device, dtype=dtype)

    # compressed KV latent
    c = torch.randn((Z, N_CTX, DC), device=device, dtype=dtype)

    # up-projection per head
    w_k = torch.randn((H, DC, D), device=device, dtype=dtype)
    w_v = torch.randn((H, DC, D), device=device, dtype=dtype)

    sm_scale = 1.0 / math.sqrt(D)

    ref = ref_mla(q, c, w_k, w_v, causal=causal, sm_scale=sm_scale).to(torch.float16)
    out = mla_forward_baseline(q, c, w_k, w_v, causal=causal, sm_scale=sm_scale).to(torch.float16)

    torch.testing.assert_close(ref, out, atol=2e-2, rtol=0)


if __name__ == "__main__":
    if not is_blackwell():
        print("Not on Blackwell; skipping.")
    else:
        # Quick manual run
        Z, H, N_CTX, D, DC = 1, 4, 256, 64, 32
        q = torch.randn((Z, H, N_CTX, D), device="cuda", dtype=torch.float16)
        c = torch.randn((Z, N_CTX, DC), device="cuda", dtype=torch.float16)
        w_k = torch.randn((H, DC, D), device="cuda", dtype=torch.float16)
        w_v = torch.randn((H, DC, D), device="cuda", dtype=torch.float16)
        sm_scale = 1.0 / math.sqrt(D)
        y = mla_forward_baseline(q, c, w_k, w_v, causal=True, sm_scale=sm_scale)
        print("ok:", y.shape, y.dtype)