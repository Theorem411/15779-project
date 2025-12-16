# flash_attn_fwd_blackwell_only.py
#
# Pruned Triton Flash-Attention example:
#  1) CUDA only (no HIP)
#  2) Blackwell only (CC major == 10)
#  3) Forward attention only (no bwd / no autograd.Function)
#  4) No Torch integration beyond using torch tensors to launch the kernel
#  5) pytest compares vs torch.nn.functional.scaled_dot_product_attention

import math
import pytest
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


def _is_blackwell_cuda() -> bool:
    return torch.cuda.is_available() and (torch.version.hip is None) and (torch.cuda.get_device_capability()[0] == 10)


@triton.jit
def _attn_fwd_bw(
    Q, K, V, O,
    Z: tl.constexpr, H: tl.constexpr, N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
    stride_z: tl.constexpr, stride_h: tl.constexpr, stride_m: tl.constexpr, stride_d: tl.constexpr,
    sm_scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)  # tiles over queries
    pid_hz = tl.program_id(1) # tiles over (z, h)

    off_z = pid_hz // H
    off_h = pid_hz % H

    # base offset for this (z,h)
    base = off_z * stride_z + off_h * stride_h

    # query rows in this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # load Q: [BM, D]
    q_ptrs = Q + base + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float16)

    # online softmax state
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # use exp2 like the original example
    qk_scale = sm_scale * 1.4426950408889634  # 1/log(2)

    # loop over keys/values
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < N_CTX

        # load K as [D, BN] for tl.dot(q [BM,D], k [D,BN]) => [BM,BN]
        k_ptrs = K + base + offs_n[None, :] * stride_m + offs_d[:, None] * stride_d
        k = tl.load(k_ptrs, mask=kv_mask[None, :] & (offs_d[:, None] < HEAD_DIM), other=0.0).to(tl.float16)

        # scores: [BM, BN]
        qk = tl.dot(q, k).to(tl.float32) * qk_scale

        if CAUSAL:
            # keep j <= i, where i is query index (offs_m) and j is key index (offs_n)
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, -1.0e6)

        # online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])  # [BM,BN]
        l_ij = tl.sum(p, axis=1)              # [BM]
        alpha = tl.math.exp2(m_i - m_ij)      # [BM]

        # rescale previous accumulator
        acc = acc * alpha[:, None]

        # load V as [BN, D] for tl.dot(p [BM,BN], v [BN,D]) => [BM,D]
        v_ptrs = V + base + offs_n[:, None] * stride_m + offs_d[None, :] * stride_d
        v = tl.load(v_ptrs, mask=kv_mask[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0).to(tl.float16)

        # accumulate
        acc = tl.dot(p.to(tl.float16), v, acc)

        # update running stats
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # write output
    out = (acc / l_i[:, None]).to(tl.float16)
    o_ptrs = O + base + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
    tl.store(o_ptrs, out, mask=q_mask)


@torch.no_grad()
def attention_fwd_blackwell(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool, sm_scale: float,
                            block_m: int = 128, block_n: int = 64, num_warps: int = 4, num_stages: int = 2):
    """
    q,k,v: [Z, H, N_CTX, HEAD_DIM], contiguous, fp16, cuda
    returns: [Z, H, N_CTX, HEAD_DIM]
    """
    if not _is_blackwell_cuda():
        raise RuntimeError("This pruned file is Blackwell CUDA-only (device capability major == 10).")
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.shape == k.shape == v.shape
    assert q.dim() == 4

    Z, H, N_CTX, HEAD_DIM = q.shape
    o = torch.empty_like(q)

    grid = (triton.cdiv(N_CTX, block_m), Z * H)

    _attn_fwd_bw[grid](
        q, k, v, o,
        Z=Z, H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM,
        stride_z=q.stride(0), stride_h=q.stride(1), stride_m=q.stride(2), stride_d=q.stride(3),
        sm_scale=sm_scale,
        CAUSAL=causal,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


def _sdpa_ref(q, k, v, causal: bool, sm_scale: float):
    # ensure the same scale behavior as our kernel
    # (PyTorch SDPA lets you pass scale explicitly)
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal, scale=sm_scale)
    except Exception:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal, scale=sm_scale)


# -------------------------
# Pytest
# -------------------------

@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM", [
    (1, 8, 128, 64),
    (2, 4, 256, 64),
    (1, 16, 512, 128),
])
@pytest.mark.parametrize("causal", [False, True])
def test_fwd_vs_sdpa(Z, H, N_CTX, HEAD_DIM, causal):
    if not _is_blackwell_cuda():
        pytest.skip("Blackwell CUDA-only test (device capability major == 10).")

    torch.manual_seed(0)
    q = torch.randn((Z, H, N_CTX, HEAD_DIM), device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn((Z, H, N_CTX, HEAD_DIM), device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn((Z, H, N_CTX, HEAD_DIM), device="cuda", dtype=torch.float16).contiguous()

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = attention_fwd_blackwell(q, k, v, causal=causal, sm_scale=sm_scale)
    ref = _sdpa_ref(q, k, v, causal=causal, sm_scale=sm_scale)

    # FP16 forward: allow small numerical drift
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    # quick manual run
    if _is_blackwell_cuda():
        test_fwd_vs_sdpa(1, 8, 128, 64, False)
        print("OK")
    else:
        print("Not on Blackwell CUDA; skipping.")