cmake_minimum_required(VERSION 3.10)

project(depth_detection)

add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

option(ENABLE_TIMER "Enable function timing" ON)
if(ENABLE_TIMER)
    message(STATUS "Enabling function timing")
    add_definitions(-DENABLE_TIMER)
endif()

# ==================== Platform Detect ====================
# 可选值: AUTO / NANO2GB / TX2 / X86_64
set(TARGET_PLATFORM "AUTO" CACHE STRING "Build target platform")
set_property(CACHE TARGET_PLATFORM PROPERTY STRINGS AUTO NANO2GB TX2 X86_64)

if(TARGET_PLATFORM STREQUAL "AUTO")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(_jetson_hint "")
    set(_jetson_hint_lc "")

    if(EXISTS "/proc/device-tree/model")
      execute_process(
        COMMAND sh -c "tr -d '\\000' < /proc/device-tree/model"
        OUTPUT_VARIABLE _jetson_model_raw
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      string(APPEND _jetson_hint " ${_jetson_model_raw}")
    endif()

    if(EXISTS "/etc/nv_tegra_release")
      file(READ "/etc/nv_tegra_release" _tegra_release_raw)
      string(APPEND _jetson_hint " ${_tegra_release_raw}")
    endif()

    string(TOLOWER "${_jetson_hint}" _jetson_hint_lc)

    # TX2 关键字: quill / tx2 / t186
    if(_jetson_hint_lc MATCHES "quill|tx2|t186")
      set(TARGET_PLATFORM "TX2")
    # Nano 关键字: nano / p3448 / t210
    elseif(_jetson_hint_lc MATCHES "nano|p3448|t210")
      set(TARGET_PLATFORM "NANO2GB")
    else()
      # 未识别型号时，ARM64 默认按 TX2 处理
      set(TARGET_PLATFORM "TX2")
    endif()
  else()
    set(TARGET_PLATFORM "X86_64")
  endif()
endif()

message(STATUS "TARGET_PLATFORM = ${TARGET_PLATFORM}")

# ==================== Platform Config ====================
set(IS_JETSON OFF)

if(TARGET_PLATFORM STREQUAL "NANO2GB")
  set(IS_JETSON ON)
  set(CUDA_ARCH "-gencode arch=compute_53,code=sm_53")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=cortex-a57")
  if(NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
    set(CMAKE_BUILD_PARALLEL_LEVEL 2)
  endif()
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -maxrregcount=32")
  message(STATUS "Jetson Nano 2GB config enabled")

elseif(TARGET_PLATFORM STREQUAL "TX2")
  set(IS_JETSON ON)
  set(CUDA_ARCH "-gencode arch=compute_62,code=sm_62")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=cortex-a57")
  if(NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
    set(CMAKE_BUILD_PARALLEL_LEVEL 4)
  endif()
  message(STATUS "Jetson TX2 config enabled")

elseif(TARGET_PLATFORM STREQUAL "X86_64")
  # 按需增删架构
  set(CUDA_ARCH "-gencode arch=compute_120,code=sm_120")
  if(NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
    set(CMAKE_BUILD_PARALLEL_LEVEL 16)
  endif()
  message(STATUS "x86_64 config enabled")

else()
  message(FATAL_ERROR "Unknown TARGET_PLATFORM: ${TARGET_PLATFORM}")
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")


include_directories("/usr/include/eigen3")

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# ============= cuda ============
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# CUDA compilation options
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math")

# ============= tensorrt ============
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Embed_platform on - Jetson")
  include_directories(/usr/include/aarch64-linux-gnu)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/cub)
  link_directories(/usr/lib/aarch64-linux-gnu)
  
  # Jetson specific TensorRT path detection
  if(EXISTS "/usr/local/TensorRT")
    include_directories(/usr/local/TensorRT/include)
    link_directories(/usr/local/TensorRT/lib)
  endif()
else()
  message(STATUS "Embed_platform off - x86_64")
  include_directories(/usr/include/x86_64-linux-gnu)
  link_directories(/usr/lib/x86_64-linux-gnu)
endif()

# ============ opencv ============
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CPP_SOURCE_DIR ${PROJECT_SOURCE_DIR}/cpp)
include_directories(${CPP_SOURCE_DIR}/include)

# =========== bytetrack lib ===========
include_directories(${CPP_SOURCE_DIR}/bytetrack/include)
file(GLOB_RECURSE BYTETRACK_SRCS ${CPP_SOURCE_DIR}/bytetrack/src/*.cpp)
add_library(bytetrack SHARED ${BYTETRACK_SRCS})
target_link_libraries(bytetrack ${OpenCV_LIBS})

# =========== cuda operator library ===========
include_directories(${CPP_SOURCE_DIR}/op_kernel/include)
file(GLOB_RECURSE CUDA_KERNEL_SRCS ${CPP_SOURCE_DIR}/op_kernel/src/*.cu)
cuda_add_library(op_kernel SHARED ${CUDA_KERNEL_SRCS})

# Jetson specific optimizations
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  # Add FP16 support option
  target_compile_definitions(op_kernel PRIVATE ENABLE_FP16=1)
endif()

target_link_libraries(op_kernel cudart nvinfer nvonnxparser ${OpenCV_LIBS})

# ============= yolo lib =============
include_directories(${CPP_SOURCE_DIR}/detect/include)
file(GLOB_RECURSE YOLO_SRCS ${CPP_SOURCE_DIR}/detect/src/*.cpp ${CPP_SOURCE_DIR}/detect/src/*.cu)
add_library(yolo_infer SHARED ${YOLO_SRCS})
target_link_libraries(yolo_infer op_kernel ${OpenCV_LIBS})

# ============= depth lib =============
include_directories(${CPP_SOURCE_DIR}/depth/include)
file(GLOB_RECURSE DEPTH_SRCS ${CPP_SOURCE_DIR}/depth/src/*.cpp ${CPP_SOURCE_DIR}/depth/src/*.cu)
cuda_add_library(depth SHARED ${DEPTH_SRCS})
target_link_libraries(depth op_kernel cudart nvinfer ${OpenCV_LIBS})

# ============= core lib =============
include_directories(${CPP_SOURCE_DIR}/core/include)
file(GLOB_RECURSE CORE_SRCS ${CPP_SOURCE_DIR}/core/src/*.cpp)
add_library(core SHARED ${CORE_SRCS})
target_link_libraries(core yolo_infer depth ${OpenCV_LIBS})

# ============= tools lib =============
include_directories(${CPP_SOURCE_DIR}/tools/include)
file(GLOB_RECURSE TOOL_SRCS ${CPP_SOURCE_DIR}/tools/src/*.cpp)
add_library(tools SHARED ${TOOL_SRCS})
target_link_libraries(tools ${OpenCV_LIBS})

# ======== main execute file ========
find_package(yaml-cpp REQUIRED)
add_executable(main ${CPP_SOURCE_DIR}/main.cpp)
target_link_libraries(main core bytetrack tools yaml-cpp)

# ==================== Google Benchmark ====================
option(ENABLE_BENCHMARK "Build benchmark tests" ON)

if(ENABLE_BENCHMARK)
    message(STATUS "Building benchmark tests")
    # find Google benchmark
    find_package(benchmark REQUIRED)
    set(BENCHMARK_SRCS ${PROJECT_SOURCE_DIR}/benchmark/)
    # test pipeline benchmark
    add_executable(test_pipeline ${BENCHMARK_SRCS}/test_pipeline.cpp)
    target_link_libraries(test_pipeline PRIVATE 
        core bytetrack yolo_infer depth tools yaml-cpp benchmark::benchmark
    )
    message(STATUS "Registered benchmark: test_pipeline")
    # test depth preprocess benchmark
    add_executable(test_depth_preprocess ${BENCHMARK_SRCS}/test_depth_preprocess.cpp)
    target_link_libraries(test_depth_preprocess PRIVATE
        core bytetrack yolo_infer depth tools yaml-cpp benchmark::benchmark
    )
    message(STATUS "Registered benchmark: test_depth_preprocess")

else()
  message(STATUS "ENABLE_BENCHMARK=OFF")
endif()

# ==================== pybind11 (Jetson 禁用) ====================
option(ENABLE_PYBIND "Build python pybind module" OFF)
if(IS_JETSON)
  set(ENABLE_PYBIND OFF CACHE BOOL "Build python pybind module" FORCE)
endif()

if(ENABLE_PYBIND)
  find_package(pybind11 REQUIRED)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/cvnp)
  pybind11_add_module(depth_detection ${CPP_SOURCE_DIR}/python_wrapper/depth_detect_py.cpp)
  target_include_directories(depth_detection PRIVATE
      ${CPP_SOURCE_DIR}/core/include
      ${CPP_SOURCE_DIR}/detect/include
      ${CPP_SOURCE_DIR}/bytetrack/include
      ${CPP_SOURCE_DIR}/depth/include
      ${CPP_SOURCE_DIR}/tools/include
  )
  target_link_libraries(depth_detection PRIVATE
      ${OpenCV_LIBS}
      ${CUDA_LIBRARIES}
      ${TRT_LIBRARIES}
      cvnp
      bytetrack
      yolo_infer
      depth
      core
      tools
      yaml-cpp
  )
  set_target_properties(depth_detection PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python
      RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python
  )
else()
  message(STATUS "ENABLE_PYBIND=OFF")
endif()
