# flash_mla_gluon.py
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_32x32b_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def is_blackwell():
    t = triton.runtime.driver.active.get_current_target()
    return t.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


@gluon.constexpr_function
def get_mma_instr_shape(shape, element_ty):
    m = 128 if shape[0] >= 128 else 64
    n = 256 if shape[1] >= 256 else shape[1]
    k = 256 // element_ty.primitive_bitwidth
    return (m, n, k)


@gluon.jit
def _init_ring(empty_bars, ready_bars, num_buffers: gl.constexpr, num_consumers: gl.constexpr = 1):
    # Match the working pattern used in the official Gluon attention tutorial:
    # - ready starts "not arrived"
    # - empty starts "arrived" (buffer is initially free)
    for i in gl.static_range(num_buffers):
        mbarrier.init(ready_bars.index(i), count=1)
        mbarrier.init(empty_bars.index(i), count=num_consumers)
        mbarrier.arrive(empty_bars.index(i), count=num_consumers)


@gluon.jit
def _ring_next(idx, phase, num_buffers: gl.constexpr):
    if num_buffers == 1:
        return gl.to_tensor(0), phase ^ 1
    nxt = idx + 1
    wrap = nxt == num_buffers
    idx2 = gl.where(wrap, 0, nxt)
    phase2 = gl.where(wrap, phase ^ 1, phase)
    return idx2, phase2


@gluon.jit
def _split_n_once(x):
    # (M, N) -> two tensors each (M, N/2), preserving contiguity along N for stores
    x0, x1 = x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1).split()
    return x0, x1


# -----------------------------------------------------------------------------
# Partitions
# -----------------------------------------------------------------------------

@gluon.jit
def producer_kv_partition(args, n_tiles: gl.tensor, BN: gl.constexpr, num_stages: gl.constexpr):
    # args = (k_desc, v_desc, k_bufs, v_bufs, empty_bars, ready_bars)
    k_desc, v_desc, k_bufs, v_bufs, empty_bars, ready_bars = args

    idx = gl.to_tensor(0)
    phase = gl.to_tensor(0)
    bytes_kv: gl.constexpr = k_desc.block_type.nbytes + v_desc.block_type.nbytes

    for t in range(n_tiles):
        empty = empty_bars.index(idx)
        ready = ready_bars.index(idx)

        mbarrier.wait(empty, phase)
        mbarrier.expect(ready, bytes_kv)

        off_n = t * BN
        tma.async_copy_global_to_shared(k_desc, [off_n, 0], ready, k_bufs.index(idx))
        tma.async_copy_global_to_shared(v_desc, [off_n, 0], ready, v_bufs.index(idx))

        idx, phase = _ring_next(idx, phase, num_stages)


@gluon.jit
def consumer_mma_partition(args, n_tiles: gl.tensor, BM: gl.constexpr, BN: gl.constexpr, BD: gl.constexpr,
                          num_stages: gl.constexpr):
    # args =
    # (q_buf, k_bufs, v_bufs, empty_bars, ready_bars, o_acc, o_ready_bar)
    q_buf, k_bufs, v_bufs, empty_bars, ready_bars, o_acc, o_ready_bar = args

    idx = gl.to_tensor(0)
    phase = gl.to_tensor(0)

    # S accumulator in TMEM: [BM, BN]
    s_instr_m, s_instr_n, _ = get_mma_instr_shape([BM, BN], gl.float32)
    s_tmem_layout: gl.constexpr = TensorMemoryLayout([s_instr_m, s_instr_n], col_stride=1)
    s_acc = allocate_tensor_memory(gl.float32, [1, BM, BN], s_tmem_layout)

    # O accumulator in TMEM: [BM, BD] (passed in as o_acc, but we still need to init it)
    # Make sure the first use has use_acc=False.
    o_inited = False

    for _ in range(n_tiles):
        ready = ready_bars.index(idx)
        empty = empty_bars.index(idx)

        mbarrier.wait(ready, phase)

        # Q[K]^T : Q=[BM,BD], K=[BN,BD] so we permute K to [BD,BN]
        k_tile = k_bufs.index(idx).permute((1, 0))
        tcgen05_mma(q_buf, k_tile, s_acc, use_acc=False)

        # (softmax would be here in a real attention/MLA kernel)
        # P @ V : P=[BM,BN], V=[BN,BD] => O=[BM,BD]
        v_tile = v_bufs.index(idx)
        tcgen05_mma(s_acc, v_tile, o_acc, use_acc=o_inited, mbarriers=[])
        o_inited = True

        # Release this KV ring slot *after* the compute that uses it.
        tcgen05_commit(empty)
        idx, phase = _ring_next(idx, phase, num_stages)

    # Signal epilogue that O is ready in TMEM.
    tcgen05_commit(o_ready_bar)


@gluon.jit
def epilogue_partition(args, BM: gl.constexpr, BD: gl.constexpr):
    # args = (o_desc, o_acc, o_ready_bar)
    o_desc, o_acc, o_ready_bar = args

    mbarrier.wait(o_ready_bar, 0)

    # Load TMEM -> regs using the correct register layout for this tile shape.
    instr_m, instr_n, _ = get_mma_instr_shape([BM, BD], gl.float32)
    reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(instr_m, instr_n, [BM, BD], num_warps=1)
    o_reg = o_acc.load(reg_layout)  # shape [BM, BD] in regs

    # Store regs -> SMEM and TMA store -> global.
    smem = gl.allocate_shared_memory(o_desc.dtype, [BM, BD], o_desc.layout)

    # Optional split-N to encourage interleaving and avoid spills for wide BD.
    # (BD=128 in your current config, this is still fine; keeping the split helps as you scale.)
    if BD >= 128:
        o0, o1 = _split_n_once(o_reg.to(o_desc.dtype))
        smem.slice(0, BD // 2, dim=1).store(o0)
        fence_async_shared()
        tma.async_copy_shared_to_global(o_desc, [0, 0], smem.slice(0, BD // 2, dim=1))
        tma.store_wait(pendings=0)
        smem.slice(BD // 2, BD // 2, dim=1).store(o1)
        fence_async_shared()
        tma.async_copy_shared_to_global(o_desc, [0, BD // 2], smem.slice(BD // 2, BD // 2, dim=1))
        tma.store_wait(pendings=0)
    else:
        smem.store(o_reg.to(o_desc.dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(o_desc, [0, 0], smem)
        tma.store_wait(pendings=0)


# -----------------------------------------------------------------------------
# Kernel
# -----------------------------------------------------------------------------

@gluon.jit
def flash_mla_kernel(q_desc, k_desc, v_desc, o_desc,
                     n_tiles: gl.tensor,
                     BM: gl.constexpr, BN: gl.constexpr, BD: gl.constexpr,
                     num_stages: gl.constexpr):
    dtype: gl.constexpr = q_desc.dtype

    # Q tile in SMEM (single buffer; you can extend to a ring later)
    q_buf = gl.allocate_shared_memory(dtype, [BM, BD], q_desc.layout)

    # KV ring in SMEM
    k_bufs = gl.allocate_shared_memory(dtype, [num_stages, BN, BD], k_desc.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [num_stages, BN, BD], v_desc.layout)

    # KV ring barriers
    kv_ready = gl.allocate_shared_memory(gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())
    kv_empty = gl.allocate_shared_memory(gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())
    _init_ring(kv_empty, kv_ready, num_stages, num_consumers=1)

    # O-ready barrier (single)
    o_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(o_ready, count=1)

    # O accumulator in TMEM
    o_instr_m, o_instr_n, _ = get_mma_instr_shape([BM, BD], gl.float32)
    o_tmem_layout: gl.constexpr = TensorMemoryLayout([o_instr_m, o_instr_n], col_stride=1)
    o_acc = allocate_tensor_memory(gl.float32, [1, BM, BD], o_tmem_layout)

    # Load Q tile into SMEM
    q_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(q_bar, count=1)
    mbarrier.expect(q_bar, q_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(q_desc, [0, 0], q_bar, q_buf)
    mbarrier.wait(q_bar, 0)
    fence_async_shared()

    # Tuple args (your requested style)
    prod_args = (k_desc, v_desc, k_bufs, v_bufs, kv_empty, kv_ready)
    cons_args = (q_buf, k_bufs, v_bufs, kv_empty, kv_ready, o_acc, o_ready)
    epi_args = (o_desc, o_acc, o_ready)

    # We launch with num_warps=6 in Python:
    # - producer: 1 warp
    # - consumer: 4 warps
    # - default(epilogue): remaining 1 warp
    gl.warp_specialize(
        default_args=(epi_args, BM, BD),
        default_partition=epilogue_partition,
        worker_args=(prod_args, n_tiles, BN, num_stages),
        worker_partitions=[producer_kv_partition, consumer_mma_partition],
        worker_num_warps=[1, 4],
        worker_num_regs=[24, 128],
    )


# -----------------------------------------------------------------------------
# Python wrapper (toy “dense attention-like” skeleton; MLA compression not wired yet)
# -----------------------------------------------------------------------------

def torch_dtype_to_gl(dtype: torch.dtype):
    # minimal mapping for your current experiments
    if dtype == torch.float16:
        return gl.float16
    if dtype == torch.bfloat16:
        return gl.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def make_desc(x, shape, strides, block_shape):
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, torch_dtype_to_gl(x.dtype))
    return TensorDescriptor(x, shape=shape, strides=strides, block_shape=block_shape, layout=layout)


def flash_mla(Q, K, V, O, *, BM=128, BN=64, BD=128, num_stages=3):
    """
    This is a *scaffold*:
      - loads one Q tile [BM,BD]
      - streams K/V tiles [BN,BD]
      - does (Q K^T) (S V) without softmax
    It’s meant to validate your warp-specialized plumbing first.
    """
    assert Q.shape == (BM, BD)
    assert K.shape[1] == BD and V.shape[1] == BD
    assert K.shape[0] == V.shape[0]
    assert O.shape == (BM, BD)
    assert K.shape[0] % BN == 0

    n_tiles = K.shape[0] // BN

    q_desc = make_desc(Q, shape=[BM, BD], strides=[BD, 1], block_shape=[BM, BD])
    k_desc = make_desc(K, shape=[K.shape[0], BD], strides=[BD, 1], block_shape=[BN, BD])
    v_desc = make_desc(V, shape=[V.shape[0], BD], strides=[BD, 1], block_shape=[BN, BD])
    o_desc = make_desc(O, shape=[BM, BD], strides=[BD, 1], block_shape=[BM, BD])

    grid = (1,)
    flash_mla_kernel[grid](
        q_desc, k_desc, v_desc, o_desc,
        n_tiles,
        BM=BM, BN=BN, BD=BD,
        num_stages=num_stages,
        num_warps=6,
        maxnreg=128,
    )


if __name__ == "__main__":
    if not is_blackwell():
        print("Not on Blackwell (sm100); exiting.")
        raise SystemExit(0)

    torch.manual_seed(0)
    M, N, D = 128, 4096, 128
    Q = torch.randn(M, D, device="cuda", dtype=torch.float16)
    K = torch.randn(N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(N, D, device="cuda", dtype=torch.float16)
    O = torch.empty(M, D, device="cuda", dtype=torch.float16)

    flash_mla(Q, K, V, O)
    print("Ran scaffold kernel once.")