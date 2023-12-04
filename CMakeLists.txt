cmake_minimum_required(VERSION 3.5)
project(bevdet)

add_compile_options(-W)
add_compile_options(-std=c++14)
set(GENCODE ${GENCODE} -gencode arch=compute_86,code=compute_86)
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O3")
set(CMAKE_BUILD_TYPE "RELEASE")
# set(CMAKE_BUILD_TYPE "DEBUG")

if(${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()


set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include)
set(CUDA_LIB_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)


if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")       # orin
  set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu/)
  set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu/)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(TENSORRT_INCLUDE_DIRS ~/software/TensorRT-8.5.3.1/include)  # your tensorrt path
  set(TENSORRT_LIBRARY_DIRS ~/software/TensorRT-8.5.3.1/lib)
endif()



find_library(NVJPEG_LIBRARY nvjpeg ${CUDA_LIB_DIRS})
if(NVJPEG_LIBRARY)
  add_definitions(-D__HAVE_NVJPEG__)
  link_libraries(${NVJPEG_LIBRARY})
  message(STATUS ${NVJPEG_LIBRARY})
endif()


find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  message_filters

  sensor_msgs
  pcl_conversions
  pcl_ros 
  cv_bridge
  jsk_recognition_msgs
)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  include_ros
)

catkin_package()



link_directories(
	${TENSORRT_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
)

# ------------------

cuda_add_executable(ros_node
    test_ros/bevdet_ros.cpp
    
    src/bevdet.cpp 
    src/preprocess.cu 
    src/bevpool.cu 
    src/iou3d_nms.cu
    src/postprocess.cu
    src/data.cpp
    src/grid_sampler.cu
    src/cpu_jpegdecoder.cpp
    src/nvjpegdecoder.cpp
)

target_link_libraries(ros_node
    yaml-cpp 
    libnvinfer.so
    libjpeg.so
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
)
