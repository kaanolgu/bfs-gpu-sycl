# Towards Efficient Load Balancing on GPUs: One Source Code for All Major Vendor GPUs with SYCL
This work has been accepted to WACCPD 2025 which is the the Twelfth Workshop on Accelerator Programming and Directives.

The paper: [https://dl.acm.org/doi/10.1145/3731599.3767570](https://dl.acm.org/doi/10.1145/3731599.3767570)

This work is tested on AMD MI210,AMD MI300X, Intel Max 1550, Nvidia A100, and Nvidia GH200 GPUS.

- We have 3 different load balancing approaches that works best in different scenarios:

  - __Local Load Balancing(LLB)__ 
distributes work efficiently within each work-group, ensuring that individual work-items share the load evenly.

  - __Global Load Balancing(GLB)__
extends load balancing across the entire device by redistributing work between work-groups.

  - __Strided Local Load Balancing(SLB)__ similar to LLB but assigns work-items using a strided mapping based on the number of work-groups.


Authors: [Kaan Olgu](https://research-information.bris.ac.uk/en/persons/kaan-olgu-2) & [Tobias Kenter](https://www.uni-paderborn.de/en/person/3145)

## Build & Run

Build instructions vary depending on your system environment (Ubuntu/Debian with
prebuilt oneAPI, or RHEL/HPC systems requiring a source build). 

See [BUILD.md](BUILD.md) for full instructions covering:
- Intel oneAPI prebuilt (AMD, NVIDIA, Intel GPUs)
- Building Intel LLVM from source (HPC / RHEL / module-based systems)
- Runtime environment setup
- Known issues

### Quick start (NVIDIA, oneAPI prebuilt)
```bash
source setvars.sh --force --include-intel-llvm

cmake -Bbuild_local -H. -DENABLE_NVIDIA_BACKEND=ON -DCUDA_ARCH=80 \
    -DGPU_TARGETS=all -DUSE_GLOBAL_LOAD_BALANCE=OFF -DSM_FACTOR=48
cmake --build build_local

./build_local/bfs_1.gpu --dataset=$dataset --root=$root \
    --num_runs=20 --output=output.json
```

> For AMD, Intel GPU, HPC clusters, or multi-GPU setups — see [BUILD.md](BUILD.md).

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
