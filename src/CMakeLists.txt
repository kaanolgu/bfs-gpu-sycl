set(SOURCE_FILE host.cpp)
set(TARGET_NAME bfs)
# set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
# set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
# set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(GPU_TARGET_1 ${TARGET_NAME}_1.gpu)
set(GPU_TARGET_2 ${TARGET_NAME}_2.gpu)
set(GPU_TARGET_3 ${TARGET_NAME}_3.gpu)
set(GPU_TARGET_4 ${TARGET_NAME}_4.gpu)

# # FPGA device selection
# if(NOT DEFINED DEVICE)
#     set(DEVICE "intel_a10gx_pac:pac_a10")
#     message(STATUS "DEVICE was not specified.\
#                     \nConfiguring the design to run on the default FPGA device ${DEVICE} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
#                     \nPlease refer to the README for information on device selection.")
# else()
    
# endif()
if("${DEVICE}" STREQUAL "INTEL_MAX_GPU")
    message(STATUS "Configuring the design to run on INTEL_MAX_GPU device ${DEVICE}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=intel_gpu_pvc -D INTEL_MAX_GPU")
elseif("${DEVICE}" STREQUAL "NVIDIA_GPU")
    message(STATUS "Configuring the design to run on NVIDIA_GPU device ${DEVICE}")
    
else()
    message(STATUS "Configuring the design to run on FPGA device ${DEVICE}")
endif()


# NUMBER OF COMPUTE UNITS
if(NOT DEFINED NUMBER_OF_WORKGROUPS)
    set(NUMBER_OF_WORKGROUPS "16")
    message(STATUS "NUMBER_OF_WORKGROUPS was not specified.\
                    \nConfiguring the design to run on the default ${NUMBER_OF_WORKGROUPS} \
                    \nPlease refer to the README for information on NUMBER_OF_WORKGROUPS selection.")
else()
    message(STATUS "WORKGROUP-SIZE: ${NUMBER_OF_WORKGROUPS}")
endif()

# NUMBER OF COMPUTE UNITS
if(NOT DEFINED NUM_GPU)
    set(NUM_GPU "1")
    message(STATUS "NUM_GPU was not specified.\
                    \nConfiguring the design to run on the default ${NUM_GPU} \
                    \nPlease refer to the README for information on NUM_GPU selection.")
else()
    message(STATUS "NUMBER OF GPUs: ${NUM_GPU}")
endif()

# These are Windows-specific flags:
# 1. /EHsc This is a Windows-specific flag that enables exception handling in host code
# 2. /Qactypes Include ac_types headers and link against ac_types emulation libraries
if(WIN32)
    set(WIN_FLAG "/EHsc")
    set(AC_TYPES_FLAG "/Qactypes")
else()
    set(AC_TYPES_FLAG "-qactypes")
endif()


# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS_GPU "-fsycl -fsycl-targets=nvptx64-nvidia-cuda ")
set(HARDWARE_LINK_FLAGS_GPU "-fsycl -fsycl-targets=nvptx64-nvidia-cuda ${USER_HARDWARE_FLAGS}")



# set(EMULATOR_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fsycl -fintelfpga ${AC_TYPES_FLAG} -DFPGA_EMULATOR -DNUM_COMPUTE_UNITS=${NUM_COMPUTE_UNITS}")
# set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG}")
# set(HARDWARE_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fsycl -fintelfpga ${AC_TYPES_FLAG} -O3 -DNUM_COMPUTE_UNITS=${NUM_COMPUTE_UNITS}")
# set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG} -Xsauto-pipeline -Xshardware -Xssfc-exit-fifo-type=zero-latency  -Xsnum-reorder=1 -Xstarget=${DEVICE} -Xsprofile ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation



###############################################################################
### GPU (NVIDIA)
###############################################################################
add_executable(${GPU_TARGET_1} ${SOURCE_FILE})
target_include_directories(${GPU_TARGET_1} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${GPU_TARGET_1} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(${GPU_TARGET_1} PUBLIC NUM_GPU=1)
set_target_properties(${GPU_TARGET_1} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS_GPU}")
set_target_properties(${GPU_TARGET_1} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS_GPU}")
add_custom_target(gpu1 DEPENDS ${GPU_TARGET_1})
###############################################################################
### GPU (NVIDIA)
###############################################################################
add_executable(${GPU_TARGET_2} ${SOURCE_FILE})
target_include_directories(${GPU_TARGET_2} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${GPU_TARGET_2} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(${GPU_TARGET_2} PUBLIC NUM_GPU=2)
set_target_properties(${GPU_TARGET_2} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS_GPU}")
set_target_properties(${GPU_TARGET_2} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS_GPU}")
add_custom_target(gpu2 DEPENDS ${GPU_TARGET_2})
###############################################################################
### GPU (NVIDIA)
###############################################################################
add_executable(${GPU_TARGET_3} ${SOURCE_FILE})
target_include_directories(${GPU_TARGET_3} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${GPU_TARGET_3} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(${GPU_TARGET_3} PUBLIC NUM_GPU=3)
set_target_properties(${GPU_TARGET_3} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS_GPU}")
set_target_properties(${GPU_TARGET_3} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS_GPU}")
add_custom_target(gpu3 DEPENDS ${GPU_TARGET_3})
###############################################################################
### GPU (NVIDIA)
###############################################################################
add_executable(${GPU_TARGET_4} ${SOURCE_FILE})
target_include_directories(${GPU_TARGET_4} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${GPU_TARGET_4} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(${GPU_TARGET_4} PUBLIC NUM_GPU=4)
set_target_properties(${GPU_TARGET_4} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS_GPU}")
set_target_properties(${GPU_TARGET_4} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS_GPU}")
add_custom_target(gpu4 DEPENDS ${GPU_TARGET_4})
# ###############################################################################
# ### FPGA Emulator
# ###############################################################################
# # To compile in a single command:
# #    icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -DFPGA_EMULATOR fpga_compile.cpp -o fpga_compile.fpga_emu
# # CMake executes:
# #    [compile] icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -DFPGA_EMULATOR -o fpga_compile.cpp.o -c fpga_compile.cpp
# #    [link]    icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} fpga_compile.cpp.o -o fpga_compile.fpga_emu
# add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
# target_include_directories(${EMULATOR_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/include)
# target_include_directories(${EMULATOR_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)
# target_include_directories(${EMULATOR_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/benchmarks/${NUMBER_CU}cu)
# set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
# set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
# add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

# ###############################################################################
# ### Generate Report
# ###############################################################################
# # To compile manually:
# #   icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -Xshardware -Xstarget=<DEVICE> -fsycl-link=early fpga_compile.cpp -o fpga_compile_report.a
# set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# # The compile output is not an executable, but an intermediate compilation result unique to SYCL.
# add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
# target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ${PROJECT_SOURCE_DIR}/include)
# add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
# set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DCACHE_ENABLED")
# set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsread-only-cache-size=4096 -Xsno-interleaving=default -fsycl-link=early")
# # fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

# ###############################################################################
# ### FPGA Hardware
# ###############################################################################
# # To compile in a single command:
# #   icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -Xshardware -Xstarget=<DEVICE> fpga_compile.cpp -o fpga_compile.fpga
# # CMake executes:
# #   [compile] icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -o fpga_compile.cpp.o -c fpga_compile.cpp
# #   [link]    icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -Xshardware -Xstarget=<DEVICE> fpga_compile.cpp.o -o fpga_compile.fpga
# add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
# target_include_directories(${FPGA_TARGET} PRIVATE ../include)
# add_custom_target(fpga DEPENDS ${FPGA_TARGET})
# set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
# set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# # The -reuse-exe flag enables rapid recompilation of host-only code changes.
# # See C++SYCL_FPGA/GettingStarted/fast_recompile for details.



