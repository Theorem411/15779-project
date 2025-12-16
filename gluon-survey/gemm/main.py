# %%
from __future__ import annotations

# Blackwell GEMM driver: Gluon (blocked/pipelined), Triton-TMA (tt-gemm),
# persistent GEMM (gemm-2), warp-specialized GEMM (gemm-3), plus optional torch baselines.

import os
import sys
import importlib.util
from dataclasses import dataclass
from typing import Callable, Optional

# -----------------------------------------------------------------------------
# IMPORTANT: avoid shadowing real PyTorch with local ./torch.py
# Put this directory at the END of sys.path so `import torch` resolves to the
# installed package, not a local file named torch.py.
# -----------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_clean = []
for p in sys.path:
    if p in ("", _THIS_DIR) or os.path.abspath(p) == _THIS_DIR:
        continue
    _clean.append(p)
_clean.append(_THIS_DIR)
sys.path[:] = _clean

import torch
import triton

# ============================================================
# CLI + device gating
# ============================================================

def _enabled(label: str) -> bool:
    return len(sys.argv) == 1 or label in sys.argv[1].split(",")


def is_blackwell() -> bool:
    return True
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError(
        f"Requires Blackwell (CC major == 10). Got {torch.cuda.get_device_name(0)} "
        f"CC={torch.cuda.get_device_capability(0)}"
    )

# ============================================================
# Local-file module loader
# ============================================================

def _load_py_file_module(module_name: str, filename: str):
    path = os.path.join(_THIS_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {filename}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ============================================================
# Optional imports of split-out implementations
#   - gemm0.py: blocked_matmul
#   - gemm1.py: blocked_matmul_pipelined
#   - tt-gemm.py: matmul_blackwell (now same signature)
#   - gemm-2.py: persistent_matmul + PersistentTileScheduler
#   - gemm-3.py: matmul_warp_specialized + PersistentTileScheduler
#   - torch.py: benchmark helpers (loaded as torch_baseline to avoid shadowing)
# ============================================================

gemm0 = _load_py_file_module("gemm0", "0-gemm.py")
gemm1 = _load_py_file_module("gemm1", "1-gemm.py")
tt_gemm = _load_py_file_module("tt_gemm", "tt-gemm.py")
gemm2 = _load_py_file_module("gemm2", "2-gemm.py")
gemm3 = _load_py_file_module("gemm3", "3-gemm.py")
torch_baseline = _load_py_file_module("torch_baseline", "torch-gemm.py")

# ============================================================
# Profiling utilities (bench ms + TFLOPs + printing)
# ============================================================

@dataclass(frozen=True)
class Problem:
    M: int
    N: int
    K: int


def bench_ms(fn: Callable[[], None], warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    return float(triton.testing.do_bench(fn))


def gemm_tflops(ms: float, prob: Problem) -> float:
    flops = 2.0 * prob.M * prob.N * prob.K
    return flops * 1e-12 / (ms * 1e-3)


def print_table_header() -> None:
    print("K, BLOCK_M, BLOCK_N, BLOCK_K, ms, TFLOPs/sec")


def print_row(prob: Problem, bm: int, bn: int, bk: int, ms: float, tflops: float) -> None:
    print(f"{prob.K}, {bm}, {bn}, {bk}, {ms:.4f}, {tflops:.2f}")


# ============================================================
# Benchmark sweep helper
# ============================================================

def run_gemm_sweep(
    mode: str,
    runner: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int], None],
    *,
    M: int = 8192,
    N: int = 8192,
    K_values=None,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    num_warps: int = 4,
    bench_warmup: int = 2,
) -> None:
    """
    runner(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps) should enqueue the GEMM.
    - A: [M, K], B: [K, N], C: [M, N]
    """
    if K_values is None:
        K_values = [2**i for i in range(9, 15)]

    print(f"Mode: {mode}")
    print_table_header()

    for K in K_values:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)

        prob = Problem(M, N, K)
        fn = lambda: runner(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)

        ms = bench_ms(fn, warmup=bench_warmup)
        tflops = gemm_tflops(ms, prob)
        print_row(prob, BLOCK_M, BLOCK_N, BLOCK_K, ms, tflops)

    print()


# ============================================================
# __main__ blocks
# ============================================================

if __name__ == "__main__" and _enabled("Blocked"):
    run_gemm_sweep(
        "Blocked",
        lambda A, B, C, bm, bn, bk, nw: gemm0.blocked_matmul(A, B, C, bm, bn, bk, False, nw),
    )

if __name__ == "__main__" and _enabled("Pipelined"):
    run_gemm_sweep(
        "Pipelined",
        lambda A, B, C, bm, bn, bk, nw: gemm1.blocked_matmul_pipelined(A, B, C, bm, bn, bk, nw),
    )

if __name__ == "__main__" and _enabled("triton"):
    run_gemm_sweep(
        "triton",
        lambda A, B, C, bm, bn, bk, nw: tt_gemm.matmul_blackwell(A, B, C, bm, bn, bk, nw),
    )

# ------------------------------------------------------------
# gemm-2: persistent_matmul_pipelined
# signature: persistent_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl)
# ------------------------------------------------------------
if __name__ == "__main__" and _enabled("persistent"):
    NUM_BUFFERS = 3
    SchedulerImpl = gemm2.GroupedPersistentTileScheduler(8)  # pass the CLASS

    def _persistent_runner(A, B, C, bm, bn, bk, nw):
        gemm2.persistent_matmul_pipelined(A, B, C, bm, bn, bk, NUM_BUFFERS, nw, SchedulerImpl)

    run_gemm_sweep(
        "persistent",
        _persistent_runner,
        # Feel free to override these on purpose if your gemm-2 kernel expects different tiling.
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=128,
        num_warps=4,
    )

# ------------------------------------------------------------
# gemm-3: warp-specialized (kept compatible with run_gemm_sweep by fixing extra args)
# signature: matmul_warp_specialized(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, SUBTILE_FACTOR, num_warps, SchedulerImpl)
# ------------------------------------------------------------
if __name__ == "__main__" and _enabled("warpspec"):
    NUM_BUFFERS = 4
    SUBTILE_FACTOR = 4
    SchedulerImpl = gemm3.PersistentTileScheduler  # pass the CLASS

    def _warpspec_runner(A, B, C, bm, bn, bk, nw):
        gemm3.matmul_warp_specialized(A, B, C, bm, bn, bk, NUM_BUFFERS, SUBTILE_FACTOR, nw, SchedulerImpl)

    # NOTE: gemm-3’s tutorial kernel is typically tuned for (BLOCK_M,BLOCK_N,BLOCK_K)=(128,256,64).
    run_gemm_sweep(
        "warpspec",
        _warpspec_runner,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        num_warps=4,
    )

# ------------------------------------------------------------
# Torch baselines (from ./torch.py loaded as torch_baseline)
#   - "torch": cuBLAS eager (no graph) vs eager (cudagraph)
#   - "torch.compile": torch.compile(torch.mm)
# ------------------------------------------------------------
if __name__ == "__main__" and _enabled("torch"):
    torch.set_float32_matmul_precision("high")
    M = N = 8192
    K_values = [2**i for i in range(9, 15)]

    print("Mode: torch")
    print_table_header()
    for K in K_values:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)

        prob = Problem(M, N, K)

        # no-graph eager (pre-allocated out)
        def fn_eager():
            torch.mm(A, B, out=C)

        ms = bench_ms(fn_eager, warmup=2)
        tflops = gemm_tflops(ms, prob)
        print_row(prob, 0, 0, 0, ms, tflops)
    print()

if __name__ == "__main__" and _enabled("torch.compile"):
    torch.set_float32_matmul_precision("high")

    # compile once (function-level compile)
    fast_mm = torch.compile(torch.mm, mode="max-autotune")

    def _compile_runner(A, B, C, bm, bn, bk, nw):
        fast_mm(A, B, out=C)

    run_gemm_sweep("torch.compile", _compile_runner)