# BUILD 

```
module load fpga intel/oneapi-llvm/24.2.1 system/CUDA/12.6.0
module load devel/CMake/3.27.6-GCCcore-13.2.0 tools/binutils/2.40-GCCcore-13.2.0

# NVIDIA GPU
# Generate ALL 8 GPUS for DGX
rm -rf build;mkdir build;cd build;cmake .. -DENABLE_NVIDIA=ON -DCUDA_ARCH=80 -DGPU_TARGETS=all -DENABLE_VERBOSE=ON
# If you prefer you could generate 1 to 4 instead of 1 to 8 
rm -rf build;mkdir build;cd build;cmake .. -DENABLE_NVIDIA=ON -DCUDA_ARCH=80 -DGPU_TARGETS="1;2;3;4" -DENABLE_VERBOSE=ON
make
```
or rely on the provided spack environment :
```
cd path/to/project
source ../spack/share/spack/setup-env.sh 
spack env activate -p environment

rm -rf build
cmake -Bbuild-spack -H.  -DENABLE_NVIDIA=ON -DCUDA_ARCH=80 -DGPU_TARGETS=all -DENABLE_VERBOSE=OFF
cmake --build build-spack
```

# RUN 
```
module load fpga intel/oneapi-llvm/24.2.1 system/CUDA/12.6.0
module load devel/CMake/3.27.6-GCCcore-13.2.0

export SYCL_PI_TRACE=1
export SYCL_PRINT_EXECUTION_GRAPH="always"

echo "RMAT-21-64 Test all GPU combinations"
for j in {1..8}; do
  ./build/bfs_${j}.gpu --dataset=rmat-21-64 --root=0 --num_runs=100
done
echo "LJ Test ALL GPU Combinations"
for j in {1..8}; do
    ./build/bfs_${j}.gpu --dataset=lj --root=10009 --num_runs=100
done
```

or rely on the alternative spack loaded packages
```
cd path/to/project
source ../spack/share/spack/setup-env.sh 
spack env activate -p environment

export SYCL_PI_TRACE=1
export SYCL_PRINT_EXECUTION_GRAPH="always"

echo "RMAT-21-64 Test all GPU combinations"
for j in {1..8}; do
  ./build/bfs_${j}.gpu --dataset=rmat-21-64 --root=0 --num_runs=100
done
echo "LJ Test ALL GPU Combinations"
for j in {1..8}; do
    ./build/bfs_${j}.gpu --dataset=lj --root=10009 --num_runs=100
done
```