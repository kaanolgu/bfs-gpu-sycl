# BUILD & RUN COMMANDS

## AMD
Tested with OneAPI Compiler 2025.0.0 with Codeplay plugin 
```
module load rocm/5.4.3
source ~/spack/opt/spack/linux-rhel8-zen3/gcc-13.3.0/intel-oneapi-compilers-2025.0.0-gwzwv5l7t3jqv4aywexkknga4seygwbh/setvars.sh --force --include-intel-llvm

# ENABLE_AMD_BACKEND = [ON/OFF] 
# AMD_GPU_TARGET = [gfx90a for MI210 ]
# GPU TARGETS = [ALL/"1;2;3"] generating multiGPU versions for how many GPUs
# ENABLE_VERBOSE = [ON/OFF] for debugging purposes
# SM_FACTOR = empirical value explained in paper

# LLB
cmake -Bbuild_local -H.  -DENABLE_AMD_BACKEND=ON -DAMD_GPU_TARGET=80 -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_local

# GLB
cmake -Bbuild_global -H.  -DENABLE_AMD_BACKEND=ON -DAMD_GPU_TARGET=80 -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=ON -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_global

# SLB
cmake -Bbuild_stride_local -H.  -DENABLE_AMD_BACKEND=ON -DAMD_GPU_TARGET=80 -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF -DUSE_STRIDED_LOCAL_LOAD_BALANCE=ON -DSM_FACTOR=48
cmake --build build_stride_local

# RUN
for j in {1..8}; do
./build_local/bfs_${j}.gpu --dataset=$dataset --root=$root --num_runs=20 --output=output_nvidia.json
./build_global/bfs_${j}.gpu --dataset=$dataset --root=$root --num_runs=20 --output=output_nvidia.json
./build_stride_local/bfs_${j}.gpu --dataset=$dataset --root=$root --num_runs=20 --output=output_nvidia.json
done
```
## NVIDIA
[Here](https://developer.codeplay.com/products/oneapi/nvidia/2025.0.0/guides/get-started-guide-nvidia) is the link for more options.
```
module load CUDA/12.6.0
module load CMake/3.27.6-GCCcore-13.2.0 binutils/2.40-GCCcore-13.2.0
source ~/spack/opt/spack/linux-rhel8-zen3/gcc-13.3.0/intel-oneapi-compilers-2025.0.0-gwzwv5l7t3jqv4aywexkknga4seygwbh/setvars.sh --force --include-intel-llvm

# ENABLE_NVIDIA_BACKEND = [ON/OFF] 
# CUDA_ARCH = [80 for A100, 90a for H100 .. ]
# GPU TARGETS = [ALL/"1;2;3"] generating multiGPU versions for how many GPUs
# ENABLE_VERBOSE = [ON/OFF] for debugging purposes
# SM_FACTOR = empirical value explained in paper

# LLB
cmake -Bbuild_local -H.  -DENABLE_NVIDIA_BACKEND=ON -DCUDA_ARCH=80 -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_local

# GLB
cmake -Bbuild_global -H.  -DENABLE_NVIDIA_BACKEND=ON -DCUDA_ARCH=80 -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=ON -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_global

# SLB
cmake -Bbuild_stride_local -H.  -DENABLE_NVIDIA_BACKEND=ON -DCUDA_ARCH=80 -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF -DUSE_STRIDED_LOCAL_LOAD_BALANCE=ON -DSM_FACTOR=48
cmake --build build_stride_local

# RUN
for j in {1..8}; do
./build_local/bfs_${j}.gpu --dataset=$dataset --root=$root --num_runs=20 --output=output_nvidia.json
./build_global/bfs_${j}.gpu --dataset=$dataset --root=$root --num_runs=20 --output=output_nvidia.json
./build_stride_local/bfs_${j}.gpu --dataset=$dataset --root=$root --num_runs=20 --output=output_nvidia.json
done
```

## INTEL
```
module load CMake/3.27.6-GCCcore-13.2.0 binutils/2.40-GCCcore-13.2.0
source ~/spack/opt/spack/linux-rhel8-zen3/gcc-13.3.0/intel-oneapi-compilers-2025.0.0-gwzwv5l7t3jqv4aywexkknga4seygwbh/setvars.sh --force --include-intel-llvm
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

# GPU TARGETS = [ALL/"1;2;3"] generating multiGPU versions for how many GPUs
# ENABLE_VERBOSE = [ON/OFF] for debugging purposes
# SM_FACTOR = empirical value explained in paper

# LLB
cmake -Bbuild_local -H.  -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_local

# GLB
cmake -Bbuild_global -H.   -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=ON -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_global

# SLB
cmake -Bbuild_stride_local -H.  -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF -DUSE_STRIDED_LOCAL_LOAD_BALANCE=ON -DSM_FACTOR=48
cmake --build build_stride_local

# RUN
for j in {1..8}; do
./build_local/bfs_${j}.gpu --dataset=$dataset --root=$root --num_runs=20 --output=output_intel.json
./build_global/bfs_${j}.gpu --dataset=$dataset --root=$root --num_runs=20 --output=output_intel.json
./build_stride_local/bfs_${j}.gpu --dataset=$dataset --root=$root --num_runs=20 --output=output_intel.json
done
```

# GENERATE DATASETS
The dataset `rmat-19-16` provided for up to 4 GPU files. Best way is to generate your own RMAT dataset via scripts in the `scripts` folder or converting your already available dataset to binary format. The python might require missing packages that could be installed via ` pip install xxx` 
```
$python --version
Python 3.12.5


python genGraph.py rmat ${scale} ${factor}
python generator.py rmat-${scale}-${factor} nnz

# Example : 
python generator.py rmat-19-16 nnz $((2**19))
