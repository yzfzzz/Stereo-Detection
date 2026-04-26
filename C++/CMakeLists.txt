cmake_minimum_required(VERSION 3.10)

project(yolov8_bytetrack)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories("/usr/include/eigen3")

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# ============= cuda ============
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# ============= tensorrt ============
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("Embed_platform on")
  include_directories(/usr/include/aarch64-linux-gnu)
  link_directories(/usr/lib/aarch64-linux-gnu)
else()
  message("Embed_platform off")
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
target_link_libraries(yolo_infer nvinfer cudart nvonnxparser ${OpenCV_LIBS})

# ============= depth lib =============
include_directories(${PROJECT_SOURCE_DIR}/depth/include)
file(GLOB_RECURSE SRCS03 ${PROJECT_SOURCE_DIR}/depth/src/*.cpp)
add_library(depth SHARED ${SRCS03})
target_link_libraries(depth cudart nvinfer ${OpenCV_LIBS})


# ======== main execute file ========
add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(main bytetrack yolo_infer depth)
