# Gluon Survey

This repository contains benchmarks and examples for Triton's Gluon experimental features, focusing on GEMM (General Matrix Multiply) kernels for NVIDIA Blackwell GPUs.

**Our video demo can be found here: ** (https://drive.google.com/drive/folders/1XJUQBkOG8Ufg3N74uQZSvWUHqv9QyRSX?usp=sharing)[https://drive.google.com/drive/folders/1XJUQBkOG8Ufg3N74uQZSvWUHqv9QyRSX?usp=sharing]

## Prerequisites

- NVIDIA Blackwell GPU (Compute Capability 10.0)
- Python 3.12+
- PyTorch with CUDA support
- Triton built from source (see below)

## Installation

### 1. Clone the Repository

```bash
git clone --recurse-submodules <repository-url>
cd gluon-survey
```

### 2. Venv and pip dependencies
```bash
python -m venv .venv --prompt triton
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Build Triton from Source

**IMPORTANT:** As of 2025/12/15, you MUST build Triton from source to use Gluon kernels. The pre-built pip packages do not include the necessary experimental Gluon features required to compile these kernels.

To build and install Triton from source:

```bash
cd triton
pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```

This will build the Triton compiler with support for:
- `triton.experimental.gluon` module
- Blackwell-specific features (`triton.experimental.gluon.language.nvidia.blackwell`)
- TMA (Tensor Memory Accelerator) operations
- Gluon JIT compiler

**Note:** Building from source may take several minutes and requires a C++ compiler and CUDA toolkit installed on your system.

### 4. Verify Installation

After building Triton, verify that the Gluon module is available:

```bash
python -c "from triton.experimental import gluon; print('Gluon is available!')"
```

## Running GEMM Kernels

The `gemm/` directory contains several GEMM kernel implementations with varying optimization levels:

- `0-gemm.py`: Blocked GEMM kernel
- `1-gemm.py`: Pipelined GEMM kernel
- `2-gemm.py`: Persistent GEMM kernel
- `3-gemm.py`: Warp-specialized GEMM kernel
- `tt-gemm.py`: Triton-TMA GEMM implementation
- `torch-gemm.py`: PyTorch/cuBLAS baseline

### Basic Usage

Run all kernels with the main driver:

```bash
cd gluon-survey/gemm
python main.py
```

### Run Specific Kernels

You can run individual kernel modes by passing them as arguments:

```bash
# Run blocked GEMM only
python main.py Blocked

# Run Triton-TMA GEMM
python main.py triton

# Run PyTorch baseline
python main.py torch

# Run multiple modes (comma-separated)
python main.py Blocked,triton,torch
```

Available modes:
- `Blocked` - Basic blocked GEMM kernel
- `Pipelined` - Pipelined GEMM with async operations
- `triton` - Triton-TMA GEMM implementation
- `persistent` - Persistent GEMM kernel
- `warpspec` - Warp-specialized GEMM
- `torch` - PyTorch cuBLAS baseline (eager mode)
- `torch.compile` - PyTorch compiled with `torch.compile`

### Example Output

The benchmarks will sweep across different K dimensions (512 to 16384) and report:
- Block dimensions (BLOCK_M, BLOCK_N, BLOCK_K)
- Execution time in milliseconds
- Performance in TFLOPs/sec

```
Mode: Blocked
K, BLOCK_M, BLOCK_N, BLOCK_K, ms, TFLOPs/sec
512, 128, 128, 128, 0.1234, 543.21
1024, 128, 128, 128, 0.2456, 689.45
...
```

## Hardware Requirements

These kernels are designed specifically for NVIDIA Blackwell GPUs (Compute Capability 10.0). The code includes runtime checks and will raise an error if run on incompatible hardware.

## Troubleshooting

### Import Error: "No module named 'triton.experimental.gluon'"

This error indicates that Triton was not built from source or the build did not complete successfully. Make sure to:
1. Build Triton from source using `cd triton && pip install -e python`
2. Verify the build completed without errors
3. Check that you can import the Gluon module

### "Requires Blackwell (CC major == 10)" Error

These kernels require a Blackwell GPU. If you see this error, your GPU is not supported. The kernels use Blackwell-specific features like:
- TMA (Tensor Memory Accelerator)
- TCGEN05 MMA operations
- Blackwell tensor memory layout

## License

See LICENSE file for details.
