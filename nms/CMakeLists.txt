project(nms)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(CUDA)
include_directories (${CUDA_INCLUDE_DIRS})


cuda_add_library(${PROJECT_NAME} SHARED
    rotate_nms.cu
    rotate_nms.cuh    
    rotate_nms_utils.h    
)




