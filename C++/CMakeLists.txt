cmake_minimum_required(VERSION 3.10)

project(depth_detection)

add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# ============= Jetson Nano Optimization Configuration ============
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Jetson Nano 2GB platform detected - enabling optimizations")
  
  # CUDA architecture configuration (Jetson Nano uses sm_53, Maxwell architecture)
  set(CUDA_ARCH "-gencode arch=compute_53,code=sm_53")
  
  # Compiler optimization options (for 2GB memory constraints)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=cortex-a57")
  
  # Reduce parallel compilation tasks to avoid memory overflow
  if(NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
    set(CMAKE_BUILD_PARALLEL_LEVEL 2)
  endif()
  
  # Add low-memory optimization options
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections -Os")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  
else()
  message(STATUS "Non-ARM64 platform - using default configuration")
  set(CUDA_ARCH "-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120")
endif()

include_directories("/usr/include/eigen3")

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# ============= cuda ============
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# CUDA compilation options
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math")

# Options to reduce VRAM usage for Jetson Nano
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -maxrregcount=32")
endif()

# ============= tensorrt ============
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Embed_platform on - Jetson Nano")
  include_directories(/usr/include/aarch64-linux-gnu)
  link_directories(/usr/lib/aarch64-linux-gnu)
  
  # Jetson Nano specific TensorRT path detection
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

# =========== bytetrack lib ===========
include_directories(${PROJECT_SOURCE_DIR}/bytetrack/include)
file(GLOB_RECURSE SRCS01 ${PROJECT_SOURCE_DIR}/bytetrack/src/*.cpp)
add_library(bytetrack SHARED ${SRCS01})
target_link_libraries(bytetrack cudart nvinfer ${OpenCV_LIBS})

# ============= yolov8 lib =============
include_directories(${PROJECT_SOURCE_DIR}/detect/include)
file(GLOB_RECURSE SRCS02 ${PROJECT_SOURCE_DIR}/detect/src/*.cpp ${PROJECT_SOURCE_DIR}/detect/src/*.cu)
cuda_add_library(yolo_infer SHARED ${SRCS02})

# Jetson Nano specific optimizations
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  # Add FP16 support option
  target_compile_definitions(yolo_infer PRIVATE ENABLE_FP16=1)
endif()

target_link_libraries(yolo_infer nvinfer cudart nvonnxparser ${OpenCV_LIBS})

# ============= depth lib =============
include_directories(${PROJECT_SOURCE_DIR}/depth/include)
file(GLOB_RECURSE SRCS03 ${PROJECT_SOURCE_DIR}/depth/src/*.cpp)
add_library(depth SHARED ${SRCS03})
target_link_libraries(depth cudart nvinfer ${OpenCV_LIBS})


# ======== main execute file ========
find_package(yaml-cpp REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/tools/include)
add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(main bytetrack yolo_infer depth yaml-cpp nvidia-ml)