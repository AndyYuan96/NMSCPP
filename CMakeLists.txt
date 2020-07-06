cmake_minimum_required(VERSION 3.0)
project(faf)

SET(CMAKE_BUILD_TYPE "Release")

## Compile as C++11
add_compile_options(-std=c++11) # failed to work on ubuntu 14.04 (TLS error)

## append external lib
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

## find PCL
find_package(PCL REQUIRED)
include_directories (${PCL_INCLUDE_DIRS})

find_package(catkin REQUIRED
        roscpp
        cv_bridge
        pcl_conversions
        pcl_ros
        )

## find opencv
#find_package(OpenCV REQUIRED)

include_directories ("/usr/include/eigen3")

## find CUDA
find_package(CUDA)
include_directories (${CUDA_INCLUDE_DIRS})

include_directories(include)

add_subdirectory(rs_cnn_base)
include_directories(rs_cnn_base/include)

add_subdirectory(nms)
set(EXTRA_LIBS ${EXTRA_LIBS} nms)

add_library(${PROJECT_NAME} SHARED
        src/util/util.cpp
        include/preprocessor/util/util.h
        src/feature/pixor_feature.cpp
        include/preprocessor/feature/pixor_feature.h
        src/label/anchor_label.cpp
        include/preprocessor/label/anchor_label.h
        src/loader/multi_frame_loader.cpp
        include/preprocessor/loader/multi_frame_loader.h
        src/preprocess/multi_frame_pixor_feature_anchor_label_preprocess.cpp
        include/preprocessor/preprocess/multi_frame_pixor_feature_anchor_label_preprocess.h
        demo/faf.h
        demo/faf.cpp)

# set prefix otherwise the output will be libfoo.so instead of foo.so
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")

target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} rs_cnn_base ${EXTRA_LIBS} ${catkin_LIBRARIES}) # ${OpenCV_LIBS}${OpenCV_LIBS}

add_executable(test_faf demo/test_faf.cpp)
target_link_libraries(test_faf ${PROJECT_NAME})
