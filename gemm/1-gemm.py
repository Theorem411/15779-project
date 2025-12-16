# gemm1.py
import pytest
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared

from triton.experimental.gluon.language.nvidia.blackwell import (
    tma,
    mbarrier,
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_32x32b_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


# ============================================================
# Blocked GEMM baseline: TMA + TCGEN05 + Multi-buffer pipeline
# (B always comes in as [K, N])
# ============================================================

@gluon.jit
def get_and_increment(counter):
    return counter % 2, counter // 2 & 1, counter + 1


@gluon.jit
def blocked_matmul_pipelined_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * (2 * BLOCK_M)
    off_n = pid_n * BLOCK_N

    # u := upper tile, v := lower tile
    u_bufs = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [2] + b_desc.block_type.shape, b_desc.layout)

    # Use two accumulators!
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    ub_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    vb_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    mma_ub_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mma_vb_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    load_ub_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    load_v_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(mma_ub_bars.index(i), count=1)
        mbarrier.init(mma_vb_bars.index(i), count=1)
        mbarrier.init(load_ub_bars.index(i), count=1)
        mbarrier.init(load_v_bars.index(i), count=1)

    load_counter = 0
    mma_counter = 0
    k = 0
    ub_acc = False
    vb_acc = False

    # U1, B1
    load_index, load_phase, load_counter = get_and_increment(load_counter)
    load_ub_bar = load_ub_bars.index(load_index)
    mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))
    tma.async_copy_global_to_shared(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
    # V1
    load_v_bar = load_v_bars.index(load_index)
    mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m + BLOCK_M, k], load_v_bar, v_bufs.index(load_index))
    k += BLOCK_K

    # U2, B2
    load_index, load_phase, load_counter = get_and_increment(load_counter)
    load_ub_bar = load_ub_bars.index(load_index)
    mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))
    tma.async_copy_global_to_shared(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
    # V2
    load_v_bar = load_v_bars.index(load_index)
    mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m + BLOCK_M, k], load_v_bar, v_bufs.index(load_index))
    k += BLOCK_K

    for _ in range(gl.cdiv(K, BLOCK_K) - 2):
        # wait Ui and Bi, UBi
        mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
        mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
        tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=ub_acc)
        tcgen05_commit(mma_ub_bars.index(mma_index))
        ub_acc = True
        # wait Vi, VBi
        mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
        tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=vb_acc)
        tcgen05_commit(mma_vb_bars.index(mma_index))
        vb_acc = True

        # wait UBi, U(i+2)
        load_index, load_phase, load_counter = get_and_increment(load_counter)
        mbarrier.wait(mma_ub_bars.index(mma_index), mma_phase)
        load_ub_bar = load_ub_bars.index(load_index)
        mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))

        # wait VBi, B(i+2), V(i+2)
        mbarrier.wait(mma_vb_bars.index(mma_index), mma_phase)
        tma.async_copy_global_to_shared(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
        load_v_bar = load_v_bars.index(load_index)
        mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m + BLOCK_M, k], load_v_bar, v_bufs.index(load_index))
        k += BLOCK_K

    acc_reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, BLOCK_N, [BLOCK_M, BLOCK_N], num_warps)

    mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
    ub_bar = mma_ub_bars.index(mma_index)
    vb_bar = mma_vb_bars.index(mma_index)
    epilogue_phase = mma_phase

    # wait U(N-1) and B(N-1), UB(N-1)
    mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
    tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=True)
    # wait V(N-1), VB(N-1)
    mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
    tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=True)

    # Wait UN and BN, UBN
    mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
    mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
    tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=True)
    tcgen05_commit(ub_bar)
    # Wait VN and VBN
    mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
    tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=True)
    tcgen05_commit(vb_bar)

    # Wait UBN, UB epilogue
    mbarrier.wait(ub_bar, epilogue_phase)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    ub = ub_tmem.load(acc_reg_layout)
    c_smem.store(ub.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)

    # Wait VBN, VB epilogue
    mbarrier.wait(vb_bar, epilogue_phase)
    vb = vb_tmem.load(acc_reg_layout)
    tma.store_wait(pendings=0)
    c_smem.store(vb.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m + BLOCK_M, off_n], c_smem)
    tma.store_wait(pendings=0)


def blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, 2 * BLOCK_M), triton.cdiv(N, BLOCK_N))
    blocked_matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 128, 128)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_blocked_matmul_pipelined(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)