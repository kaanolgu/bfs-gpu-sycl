# Towards Efficient Load Balancing on GPUs: One Source Code for All Major Vendor GPUs with SYCL
This work has been submitted to IEEE Cluster 2025 which is the the 27th edition of the IEEE Cluster.

This work is tested on AMD MI210,AMD MI300X, Intel Max 1550, Nvidia A100, and Nvidia GH200 GPUS.

- We have 3 different load balancing approaches that works best in different scenarios:

  - __Local Load Balancing(LLB)__ 
distributes work efficiently within each work-group, ensuring that individual work-items share the load evenly.

  - __Global Load Balancing(GLB)__
extends load balancing across the entire device by redistributing work between work-groups.

  - __Strided Local Load Balancing(SLB)__ similar to LLB but assigns work-items using a strided mapping based on the number of work-groups.


Authors: [Kaan Olgu](https://research-information.bris.ac.uk/en/persons/kaan-olgu-2) & [Tobias Kenter](https://www.uni-paderborn.de/en/person/3145)

## Build & Run Commands
- For the Intel OneAPI Compiler Spack Package with `+amd` and `+nvidia` plugin options enabled
### AMD
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
cmake -Bbuild_local -H.  -DENABLE_AMD_BACKEND=ON -DAMD_GPU_TARGET=gfx90a -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_local

# GLB
cmake -Bbuild_global -H.  -DENABLE_AMD_BACKEND=ON -DAMD_GPU_TARGET=gfx90a -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=ON -DUSE_STRIDED_LOCAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_global

# SLB
cmake -Bbuild_stride_local -H.  -DENABLE_AMD_BACKEND=ON -DAMD_GPU_TARGET=gfx90a -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF -DUSE_GLOBAL_LOAD_BALANCE=OFF -DUSE_STRIDED_LOCAL_LOAD_BALANCE=ON -DSM_FACTOR=48
cmake --build build_stride_local

# RUN
for j in {1..8}; do
./build_local/bfs_${j}.gpu --dataset=$dataset --root=$root --num_runs=20 --output=output_nvidia.json
./build_global/bfs_${j}.gpu --dataset=$dataset --root=$root --num_runs=20 --output=output_nvidia.json
./build_stride_local/bfs_${j}.gpu --dataset=$dataset --root=$root --num_runs=20 --output=output_nvidia.json
done
```
### NVIDIA
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

### INTEL
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

## Generate Datasets
The dataset `rmat-19-16` provided for up to 4 GPU files. Best way is to generate your own RMAT dataset via scripts in the `scripts` folder or converting your already available dataset to binary format. The python might require missing packages that could be installed via ` pip install xxx` 
```
$python --version
Python 3.12.5


python genGraph.py rmat ${scale} ${factor}
python generator.py rmat-${scale}-${factor} nnz

# Example : 
python generator.py rmat-19-16 nnz $((2**19))
```


## Performance Results
Here is a table that we captured the throughput values in **GTEPS**
<img width="810" alt="image" src="https://github.com/user-attachments/assets/9c7bf23f-691e-431e-a6c5-34aa45a53592" />



## Cite

## Acknowledgments
> The authors gratefully acknowledge the computing time provided to them on the high-performance computers Noctua2 at the NHR Center PC2. These are funded by the Federal Ministry of Education and Research and the state governments participating on the basis of the resolutions of the GWK for the national highperformance computing at universities (www.nhr-verein.de/unsere-partner).

> (Intel Tiber AI Cloud)[https://www.intel.com/content/www/us/en/developer/tools/tiber/ai-cloud.html]

> This work used the DiRAC@Durham facility managed by the Institute for Computational Cosmology on behalf of the STFC DiRAC HPC Facility (www.dirac.ac.uk). The equipment was funded by BEIS capital funding via STFC capital grants ST/P002293/1, ST/R002371/1 and ST/S002502/1, Durham University and STFC operations grant ST/R000832/1. DiRAC is part of the National e-Infrastructure.
