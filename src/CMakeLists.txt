set(SOURCE_FILE host.cpp)
set(TARGET_NAME bfs)


# Define the GPU targets and corresponding GPU numbers
set(GPU_TARGETS "1;2;3;4;5;6;7;8")

    
# endif()
if("${DEVICE}" STREQUAL "INTEL_MAX_GPU")
    message(STATUS "Configuring the design to run on INTEL_MAX_GPU device ${DEVICE}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=intel_gpu_pvc -D INTEL_MAX_GPU")
elseif("${DEVICE}" STREQUAL "NVIDIA_GPU")
    message(STATUS "Configuring the design to run on NVIDIA_GPU device ${DEVICE}")
    set(COMPILE_FLAGS_GPU "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -O1 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80")
    set(HARDWARE_LINK_FLAGS_GPU "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -O1 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 ${USER_HARDWARE_FLAGS}")
else()
    message(STATUS "Configuring the design to run on FPGA device ${DEVICE}")
endif()



# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.


foreach(GPU_NUMBER IN LISTS GPU_TARGETS)
    set(GPU_TARGET ${TARGET_NAME}_${GPU_NUMBER}.gpu)

    add_executable(${GPU_TARGET} ${SOURCE_FILE})
    target_include_directories(${GPU_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(${GPU_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_compile_definitions(${GPU_TARGET} PUBLIC NUM_GPU=${GPU_NUMBER})
    target_compile_definitions(${GPU_TARGET} PUBLIC VERBOSE=${VERBOSE_INFO})
    set_target_properties(${GPU_TARGET} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS_GPU}")
    set_target_properties(${GPU_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS_GPU}")

    # Define a custom target for each GPU
    add_custom_target(gpu${GPU_NUMBER} DEPENDS ${GPU_TARGET})
endforeach()
