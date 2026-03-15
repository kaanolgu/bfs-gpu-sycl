# Build Instructions

## Table of Contents
- [Prerequisites](#prerequisites)
- [Option A: Intel oneAPI Prebuilt Compiler](#option-a-intel-oneapi-prebuilt-compiler)
  - [NVIDIA](#nvidia)
  - [AMD](#amd)
  - [Intel](#intel)
- [Option B: Build Intel LLVM from Source (HPC / RHEL)](#option-b-build-intel-llvm-from-source-hpc--rhel)
- [Running](#running)
- [Known Issues](#known-issues)

---

## Prerequisites

### Option A — Intel oneAPI
- Intel oneAPI 2025.0.0 installed via Spack with `+nvidia` and/or `+amd` plugin options
- For NVIDIA: CUDA 12.6.0+
- For AMD: ROCm 5.4.3+

### Option B — HPC / RHEL / Module-based systems
- GCC 14.2.0+
- CUDA 12.8.0+
- CMake 3.31+
- Ninja
- Python 3.11+

---

## Option A: Intel oneAPI Prebuilt Compiler

Suitable for Ubuntu/Debian systems where Intel oneAPI is installed.

### NVIDIA

See [here](https://developer.codeplay.com/products/oneapi/nvidia/2025.0.0/guides/get-started-guide-nvidia) for more options.
```bash
module load CUDA/12.6.0
module load CMake/3.27.6-GCCcore-13.2.0 binutils/2.40-GCCcore-13.2.0
source ~/spack/opt/spack/linux-rhel8-zen3/gcc-13.3.0/intel-oneapi-compilers-2025.0.0-gwzwv5l7t3jqv4aywexkknga4seygwbh/setvars.sh --force --include-intel-llvm
```
```bash
# ENABLE_NVIDIA_BACKEND = [ON/OFF]
# CUDA_ARCH             = [80 for A100, 90a for H100, 89 for L40S ..]
# GPU_TARGETS           = [ALL/"1;2;3"] number of GPU variants to generate
# ENABLE_VERBOSE        = [ON/OFF] for debugging purposes
# SM_FACTOR             = empirical value explained in paper

# LLB
cmake -Bbuild_local -H. \
    -DENABLE_NVIDIA_BACKEND=ON -DCUDA_ARCH=80 -DGPU_TARGETS=all \
    -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF \
    -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_local

# GLB
cmake -Bbuild_global -H. \
    -DENABLE_NVIDIA_BACKEND=ON -DCUDA_ARCH=80 -DGPU_TARGETS=all \
    -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=ON \
    -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_global

# SLB
cmake -Bbuild_stride_local -H. \
    -DENABLE_NVIDIA_BACKEND=ON -DCUDA_ARCH=80 -DGPU_TARGETS=all \
    -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF \
    -DUSE_STRIDED_LOCAL_LOAD_BALANCE=ON -DSM_FACTOR=48
cmake --build build_stride_local
```

### AMD

Tested with oneAPI Compiler 2025.0.0 with Codeplay plugin.
```bash
module load rocm/5.4.3
source ~/spack/opt/spack/linux-rhel8-zen3/gcc-13.3.0/intel-oneapi-compilers-2025.0.0-gwzwv5l7t3jqv4aywexkknga4seygwbh/setvars.sh --force --include-intel-llvm
```
```bash
# ENABLE_AMD_BACKEND = [ON/OFF]
# AMD_GPU_TARGET     = [gfx90a for MI210, gfx942 for MI300X ..]
# GPU_TARGETS        = [ALL/"1;2;3"] number of GPU variants to generate
# ENABLE_VERBOSE     = [ON/OFF] for debugging purposes
# SM_FACTOR          = empirical value explained in paper

# LLB
cmake -Bbuild_local -H. \
    -DENABLE_AMD_BACKEND=ON -DAMD_GPU_TARGET=gfx90a -DGPU_TARGETS=all \
    -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF \
    -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_local

# GLB
cmake -Bbuild_global -H. \
    -DENABLE_AMD_BACKEND=ON -DAMD_GPU_TARGET=gfx90a -DGPU_TARGETS=all \
    -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=ON \
    -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_global

# SLB
cmake -Bbuild_stride_local -H. \
    -DENABLE_AMD_BACKEND=ON -DAMD_GPU_TARGET=gfx90a -DGPU_TARGETS=all \
    -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF \
    -DUSE_STRIDED_LOCAL_LOAD_BALANCE=ON -DSM_FACTOR=48
cmake --build build_stride_local
```

### Intel
```bash
module load CMake/3.27.6-GCCcore-13.2.0 binutils/2.40-GCCcore-13.2.0
source ~/spack/opt/spack/linux-rhel8-zen3/gcc-13.3.0/intel-oneapi-compilers-2025.0.0-gwzwv5l7t3jqv4aywexkknga4seygwbh/setvars.sh --force --include-intel-llvm
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
```
```bash
# GPU_TARGETS    = [ALL/"1;2;3"] number of GPU variants to generate
# ENABLE_VERBOSE = [ON/OFF] for debugging purposes
# SM_FACTOR      = empirical value explained in paper

# LLB
cmake -Bbuild_local -H. \
    -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF \
    -DUSE_GLOBAL_LOAD_BALANCE=OFF -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF \
    -DSM_FACTOR=48
cmake --build build_local

# GLB
cmake -Bbuild_global -H. \
    -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF \
    -DUSE_GLOBAL_LOAD_BALANCE=ON -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF \
    -DSM_FACTOR=48
cmake --build build_global

# SLB
cmake -Bbuild_stride_local -H. \
    -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF \
    -DUSE_GLOBAL_LOAD_BALANCE=OFF -DUSE_STRIDED_LOCAL_LOAD_BALANCE=ON \
    -DSM_FACTOR=48
cmake --build build_stride_local
```

---

## Option B: Build Intel LLVM from Source (HPC / RHEL)

Use this if prebuilt oneAPI binaries are unavailable or incompatible with your
system's glibc — common on RHEL/Rocky Linux HPC clusters.

### 1. Load modules

Adjust module names to match your HPC environment:
```bash
module load GCC/14.2.0
module load CUDA/12.8.0
module load Ninja/1.12.1-GCCcore-14.2.0
module load CMake/3.31.3-GCCcore-14.2.0
module load Python/3.13.1-GCCcore-14.2.0
module load binutils/2.42-GCCcore-14.2.0
```

### 2. Clone and build Intel LLVM
```bash
git clone https://github.com/intel/llvm.git --branch sycl
cd llvm
git checkout nightly-2026-03-12  # or latest stable nightly tag

export CUDADIR=/path/to/CUDA     # e.g. /opt/software/software/CUDA/12.8.0
export CUDA_PATH=$CUDADIR
export CXX=g++
export CC=gcc

CUDA_LIB_PATH=${CUDADIR}/stubs/lib64 CC=gcc CXX=g++ python ./buildbot/configure.py \
    --cuda --native_cpu \
    --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=${CUDADIR}" \
    --cmake-opt="-DLLVM_ENABLE_ZSTD=OFF" \
    -t Release

CUDA_LIB_PATH=${CUDADIR}/stubs/lib64 CC=gcc CXX=g++ python ./buildbot/compile.py
```

> **Note:** `-DLLVM_ENABLE_ZSTD=OFF` is required when the system zstd provides
> only a non-PIC static library, which causes linker errors when building shared
> objects. This is common with EasyBuild-installed zstd.

### 3. Set up runtime environment
```bash
export DPCPP_HOME=/path/to/llvm/parent
export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH

# Required to use the real CUDA driver instead of toolkit stubs
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
```

Verify your GPUs are visible:
```bash
sycl-ls
```

### 4. Build the project
```bash
# CUDA_ARCH  = [80 for A100, 90 for H100, 89 for L40S ..]
# SM_FACTOR  = empirical value explained in paper

# LLB
cmake -Bbuild_local -H. \
    -DCMAKE_CXX_COMPILER=$DPCPP_HOME/llvm/build/bin/clang++ \
    -DENABLE_NVIDIA_BACKEND=ON -DCUDA_ARCH=80 -DGPU_TARGETS=all \
    -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF \
    -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_local -j$(nproc)

# GLB
cmake -Bbuild_global -H. \
    -DCMAKE_CXX_COMPILER=$DPCPP_HOME/llvm/build/bin/clang++ \
    -DENABLE_NVIDIA_BACKEND=ON -DCUDA_ARCH=80 -DGPU_TARGETS=all \
    -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=ON \
    -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_global -j$(nproc)

# SLB
cmake -Bbuild_stride_local -H. \
    -DCMAKE_CXX_COMPILER=$DPCPP_HOME/llvm/build/bin/clang++ \
    -DENABLE_NVIDIA_BACKEND=ON -DCUDA_ARCH=80 -DGPU_TARGETS=all \
    -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF \
    -DUSE_STRIDED_LOCAL_LOAD_BALANCE=ON -DSM_FACTOR=48
cmake --build build_stride_local -j$(nproc)
```

---

## Running
```bash
# Set your dataset path and root vertex
dataset=rmat-19-16-nnz
root=0

for j in {1..8}; do
    ./build_local/bfs_${j}.gpu \
        --dataset=$dataset --root=$root \
        --num_runs=20 --output=output_local.json
    ./build_global/bfs_${j}.gpu \
        --dataset=$dataset --root=$root \
        --num_runs=20 --output=output_global.json
    ./build_stride_local/bfs_${j}.gpu \
        --dataset=$dataset --root=$root \
        --num_runs=20 --output=output_stride_local.json
done
```

The `j` index in `bfs_${j}.gpu` corresponds to the number of GPUs used.
For example `bfs_1.gpu` runs on 1 GPU, `bfs_2.gpu` on 2 GPUs, and so on.

---

## Known Issues

### Multi-GPU incorrect results on PCIe-only systems

Multi-GPU variants (`bfs_2.gpu` and above) produce incorrect BFS results on
systems where GPUs are connected via PCIe without NVLink. This is due to
unsupported P2P atomics — the kernel uses cross-GPU atomic operations that
silently fail on PCIe-only interconnects.

You can check your system's GPU topology with:
```bash
nvidia-smi topo -m
```

If the output shows `SYS` or `PHB` between GPUs (rather than `NV#`), P2P
atomics are not supported and multi-GPU execution will produce wrong results.
Single GPU execution works correctly on all supported hardware.

See the related [issue](#) for details and progress on a fix.
