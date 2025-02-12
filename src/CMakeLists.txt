set(SOURCE_FILE host.cpp)
set(TARGET_NAME bfs)

 
# Define a backend
option(ENABLE_NVIDIA_BACKEND "Enable the NVIDIA backend" OFF)
option(ENABLE_AMD_BACKEND "Enable the AMD backend" OFF)
option(ENABLE_VERBOSE "Enable the verbose version of the execution" OFF)
set(CXX_EXTRA_FLAGS "" CACHE STRING "Define Extra flags here like --cuda-path etc.")
option(USE_GLOBAL_LOAD_BALANCE "Enable the Global load balance, if OFF it will use local load balancing" OFF)
option(USE_STRIDED_LOCAL_LOAD_BALANCE "Enable the strided version of the execution which performs better with RMAT graphs" OFF)
set(SM_FACTOR 96 CACHE STRING "Define SM_FACTOR for the global load balance")

# endif()
if(ENABLE_NVIDIA_BACKEND)
    message(STATUS "Configuring the design to run on NVIDIA_GPU device")
    set(CUDA_ARCH "80" CACHE STRING "Set the number of CUDA_ARCH to use (e.g. 80 for sm_80)")
    set(SYCL_TARGET "nvptx64-nvidia-cuda" CACHE STRING "Set the device")
    set(SYCL_BACKEND "-Xsycl-target-backend=${SYCL_TARGET}" CACHE STRING "Set the device")
    set(SYCL_BACKEND_DEVICE "--cuda-gpu-arch=sm_${CUDA_ARCH}" CACHE STRING "Set the device")
elseif(ENABLE_AMD_BACKEND)
    message(STATUS "AMD backend is enabled.")
    set(AMD_GPU_TARGET "gfx90a" CACHE STRING "Set the number of AMD_GPU_TARGET to use (e.g. gfx90a)")
    set(SYCL_TARGET "amdgcn-amd-amdhsa" CACHE STRING "Set the device")
    set(SYCL_BACKEND "-Xsycl-target-backend=${SYCL_TARGET}" CACHE STRING "Set the device")
    set(SYCL_BACKEND_DEVICE "--offload-arch=${AMD_GPU_TARGET}" CACHE STRING "Set the device")
else()
    set(SYCL_TARGET "spir64" CACHE STRING "Set the device")
    message(STATUS "Backend is disabled.")
endif()


# set(COMPILE_LINK_FLAGS "-fopenmp -I/opt/software/FPGA/IntelFPGA/oneapi/24.2.1/compiler/latest/opt/compiler/include/ -fsycl -fsycl-targets=${SYCL_TARGET} ${SYCL_BACKEND} ${SYCL_BACKEND_DEVICE} ${CXX_EXTRA_FLAGS}")
set(COMPILE_LINK_FLAGS "-fsycl -fsycl-targets=${SYCL_TARGET} ${SYCL_BACKEND} ${SYCL_BACKEND_DEVICE} ${CXX_EXTRA_FLAGS}")


# Set GPU_TARGETS option: can be 'single', 'all', or a custom list
set(GPU_TARGETS "single" CACHE STRING "Set GPU targets: 'single', 'all', or a list (e.g., '1;2;3')")

# Set the number of available GPU targets (adjust this as needed)
set(AVAILABLE_GPU_TARGETS "1;2;3;4;5;6;7;8")

# Determine the list of targets based on the input
if(GPU_TARGETS STREQUAL "single")
    set(GPU_TARGET_LIST "1")  # Default to single GPU target '1'
elseif(GPU_TARGETS STREQUAL "all")
    set(GPU_TARGET_LIST ${AVAILABLE_GPU_TARGETS})  # Use all available targets
else()
    # Assume the user provided a custom list, split into a list
    set(GPU_TARGET_LIST ${GPU_TARGETS})
endif()

# Loop through the list of GPU targets
foreach(GPU_NUMBER IN LISTS GPU_TARGET_LIST)
    set(GPU_TARGET ${TARGET_NAME}_${GPU_NUMBER}.gpu)

    add_executable(${GPU_TARGET} ${SOURCE_FILE})
    target_include_directories(${GPU_TARGET} PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(${GPU_TARGET} PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_compile_definitions(${GPU_TARGET} PUBLIC NUM_GPU=${GPU_NUMBER})
    target_compile_definitions(${GPU_TARGET} PUBLIC SM_FACTOR=${SM_FACTOR})
    target_compile_definitions(${GPU_TARGET} PUBLIC VERBOSE=$<IF:$<BOOL:${ENABLE_VERBOSE}>,1,0>)
    target_compile_definitions(${GPU_TARGET} PUBLIC USE_GLOBAL_LOAD_BALANCE=$<IF:$<BOOL:${USE_GLOBAL_LOAD_BALANCE}>,1,0>)
    target_compile_definitions(${GPU_TARGET} PUBLIC USE_STRIDED_LOCAL_LOAD_BALANCE=$<IF:$<BOOL:${USE_STRIDED_LOCAL_LOAD_BALANCE}>,1,0>)
    set_target_properties(${GPU_TARGET} PROPERTIES COMPILE_FLAGS "${COMPILE_LINK_FLAGS}")
    set_target_properties(${GPU_TARGET} PROPERTIES LINK_FLAGS "${COMPILE_LINK_FLAGS}")

    # Define a custom target for each GPU
    add_custom_target(gpu${GPU_NUMBER} DEPENDS ${GPU_TARGET})
endforeach()

message(STATUS "GPU targets: ${GPU_TARGET_LIST}")
