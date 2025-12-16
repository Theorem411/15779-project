# tt-gemm.py
# %%
# Triton Blackwell GEMM (B is always provided as transposed: B == [K, N])
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# ============================================================
# Device gating
# ============================================================

def is_blackwell() -> bool:
    return True
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


# ============================================================
# 1) Scheduler Logic: Grouped Persistent (Swizzling)
# ============================================================

@triton.jit
def get_swizzled_tile_pid(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    group_inner_id = tile_id % num_pid_in_group
    pid_m = first_pid_m + (group_inner_id % group_size_m)
    pid_n = group_inner_id // group_size_m
    return pid_m, pid_n


# ============================================================
# 2) Kernel: B hardcoded as transposed [K, N]
#    A: [M, K], B: [K, N], C: [M, N]
# ============================================================

@triton.jit
def matmul_blackwell_kernel(
    a_desc, b_desc, c_desc,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # persistent: stride by NUM_SMS across the full tile space
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = get_swizzled_tile_pid(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)

        offs_m = pid_m * BLOCK_M
        offs_n = pid_n * BLOCK_N

        # K loop
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            # A tile: [BLOCK_M, BLOCK_K]
            a = a_desc.load([offs_m, offs_k])
            # B is hardcoded as transposed in memory: [K, N]
            # B tile: [BLOCK_K, BLOCK_N]
            b = b_desc.load([offs_k, offs_n])
            acc = tl.dot(a, b, acc)

        # store C tile: [BLOCK_M, BLOCK_N]
        c_desc.store([offs_m, offs_n], acc.to(tl.float16))

        # reset accumulator for next tile
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


# ============================================================
# 3) Host-side wrapper with the SAME signature as Gluon GEMMs
#    matmul_blackwell(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
# ============================================================

def matmul_blackwell(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, *, group_size_m: int = 8):
    """
    A: [M, K] fp16
    B: [K, N] fp16  (IMPORTANT: already transposed / column-major logical operand)
    C: [M, N] fp16  (preallocated; this function writes into it)
    """
    if not is_blackwell():
        raise RuntimeError(
            f"Requires Blackwell (CC major == 10). Got {torch.cuda.get_device_name(0)} "
            f"CC={torch.cuda.get_device_capability(0)}"
        )

    M, K_a = A.shape
    K_b, N = B.shape
    assert K_a == K_b, f"A is [M,K]={A.shape} but B is [K,N]={B.shape}"
    assert C.shape == (M, N), f"C must be [M,N]={M,N}, got {C.shape}"
    assert A.dtype == torch.float16 and B.dtype == torch.float16 and C.dtype == torch.float16
    assert A.is_cuda and B.is_cuda and C.is_cuda

    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count

    # TMA descriptors: set block shapes that match the kernel's tiling.
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    # B is *hardcoded transposed*: [K, N] → block shape [BLOCK_K, BLOCK_N]
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])

    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)

    def grid(META):
        # persistent grid: up to NUM_SMS programs, but never more than tiles
        return (min(NUM_SMS, num_tiles),)

    matmul_blackwell_kernel[grid](
        a_desc, b_desc, c_desc,
        M=M, N=N, K=K_a,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=group_size_m,
        NUM_SMS=NUM_SMS,
        num_warps=num_warps,
    )
    # match Gluon runners: no return value (writes into C)
    return None


# ============================================================
# Optional standalone quick check
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    if not is_blackwell():
        raise RuntimeError("tt-gemm.py requires Blackwell (SM100 / CC major 10).")

    M = N = 2048
    K = 2048
    BM = BN = BK = 128
    num_warps = 8

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)  # already [K,N]
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    matmul_blackwell(A, B, C, BM, BN, BK, num_warps)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)
    print("OK")