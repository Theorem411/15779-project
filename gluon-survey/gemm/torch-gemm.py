import torch
import time

def benchmark_with_cudagraph(fn, warmup=25, rep=100):
    """
    Benchmarks a function using CUDA Graphs to eliminate CPU launch overhead,
    matching triton.testing.do_bench_cudagraph behavior.
    """
    # 1. Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 2. Capture CUDA Graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()

    # 3. Benchmark the Graph Replay
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        g.replay()
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / rep

def get_flops(M, N, K, ms):
    return (2 * M * N * K) / (ms * 1e-3) / 1e12

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    M, N = 8192, 8192
    
    print(f"Benchmarking PyTorch Baseline (M={M}, N={N})")
    print("=" * 65)
    print(f"{'K':>6} | {'cuBLAS (No Graph)':>18} | {'cuBLAS (Graph)':>18} | {'Inductor':>15}")
    print("-" * 65)

    K_values = [2**i for i in range(9, 15)] # 512 ... 16384

    for K in K_values:
        # Match Dtypes exactly (FP16)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        
        # FIXED: Pre-allocate Output to match Gluon's memory usage
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)

        # -------------------------------------------------------
        # 1. Standard Eager (with pre-allocation)
        # -------------------------------------------------------
        # Uses torch.mm with out=C to avoid allocation overhead
        fn_eager = lambda: torch.mm(A, B, out=C)
        
        # Standard benchmark (CPU overhead included)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warmup
        for _ in range(10): fn_eager()
        torch.cuda.synchronize()
        
        start_event.record()
        for _ in range(100): fn_eager()
        end_event.record()
        torch.cuda.synchronize()
        ms_eager_no_graph = start_event.elapsed_time(end_event) / 100
        tf_eager_no_graph = get_flops(M, N, K, ms_eager_no_graph)

        # -------------------------------------------------------
        # 2. CUDA Graph Eager (Strict Match to Gluon)
        # -------------------------------------------------------
        # This removes CPU launch latency, making it strictly comparable
        # to triton.testing.do_bench_cudagraph
        ms_eager_graph = benchmark_with_cudagraph(fn_eager)
        tf_eager_graph = get_flops(M, N, K, ms_eager_graph)

        # -------------------------------------------------------
        # 3. Inductor (Compiled)
        # -------------------------------------------------------
        tf_compile = 0.0
        try:
            # max-autotune will often use CUDA Graphs internally automatically
            fast_mm = torch.compile(torch.mm, mode="max-autotune")
            fast_mm(A, B, out=C) # Warmup / Compile trigger
            
            # Benchmark using standard timing (compiled kernel usually handles graph internally)
            # But we can wrap it just in case if safe, though usually we trust inductor's own graph.
            # We'll use the basic timer here to avoid double-graphing issues.
            start_event.record()
            for _ in range(100): fast_mm(A, B, out=C)
            end_event.record()
            torch.cuda.synchronize()
            ms_compile = start_event.elapsed_time(end_event) / 100
            tf_compile = get_flops(M, N, K, ms_compile)
        except Exception as e:
            pass

        print(f"{K:>6} | {tf_eager_no_graph:>18.2f} | {tf_eager_graph:>18.2f} | {tf_compile:>15.2f}")