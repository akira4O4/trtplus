cmake_minimum_required(VERSION 3.6)
project(trtplus VERSION 1.0.0 LANGUAGES C CXX)

set(TARGET_NAME "main.bin")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 75)

configure_file(version.h.in "${PROJECT_SOURCE_DIR}/src/version.h" @ONLY)
message("TARGET_NAME:          " ${TARGET_NAME})
message("PROJECT_VERSION:       " ${PROJECT_VERSION})
message("PROJECT_VERSION_MAJOR: " ${PROJECT_VERSION_MAJOR})
message("PROJECT_VERSION_MINOR: " ${PROJECT_VERSION_MINOR})
message("PROJECT_VERSION_PATH:  " ${PROJECT_VERSION_PATCH})

# SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)   
set(OpenCV_HOME "/home/seeking/llf/opencv")
set(TRT_HOME "/home/seeking/llf/TensorRT")
set(CUDA_HOME "/usr/local/cuda")

set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")

set(OpenCV_DIR "${OpenCV_HOME}/share/OpenCV")
find_package(OpenCV)
message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
message("OpenCV_LIBS: " ${OpenCV_LIBS})
message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})

include_directories(
        ${PROJECT_SOURCE_DIR}/src
        ${TRT_HOME}/include
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_HOME}/include
)

link_directories(
        ${OpenCV_HOME}/lib
        ${TRT_HOME}/lib
        ${CUDA_HOME}/lib64
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")

file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)

message(STATUS "Cpp source files: ${cpp_srcs}")
message(STATUS "Cuda source files: ${cuda_srcs}")

add_executable(${TARGET_NAME} ${cpp_srcs} ${cuda_srcs} main.cpp)

target_link_libraries(${TARGET_NAME} pthread)
target_link_libraries(${TARGET_NAME} nvinfer nvinfer_plugin)
target_link_libraries(${TARGET_NAME} cuda cublas cudart)
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
