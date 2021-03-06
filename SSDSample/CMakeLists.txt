cmake_minimum_required(VERSION 2.8.3)
project(SSD)

set(USE_CUDA True)

SET(SAMPLE_SOURCES
    sampleSSD.cpp
)

include_directories(
  ../common
)
# set(SAMPLES_COMMON_SOURCES
#   ../common/logger.cpp
# )

# set flags for CUDA availability
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
message("CUDA dir paths", ${CUDA_LIBRARIES})

include_directories( 
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  )

if (USE_CUDA)
  message("CUDA is available!")
  message("CUDA Libs: ${CUDA_LIBRARIES}")
  message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  set(CUDA_AVAIL ON)
else()
  message("CUDA NOT FOUND OR INCOMPATIBLE CMAKE VERSION FOUND")
  set(CUDA_AVAIL OFF)
endif ()

# set flags for TensorRT availability
include_directories(/home/zhangcheng/Downloads/TensorRT-8.2.3.0/include/)
link_directories(/home/zhangcheng/Downloads/TensorRT-8.2.3.0/lib/)

# set flags for CUDNN availability
option(CUDNN_AVAIL "CUDNN available" OFF)
# try to find the CUDNN module
find_library(CUDNN_LIBRARY
  NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
  PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
  PATH_SUFFIXES lib lib64 bin
  DOC "CUDNN library."
)

if(CUDNN_LIBRARY)
  message("CUDNN is available!")
  message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
  set(CUDNN_AVAIL ON)
else()
  message("CUDNN is NOT Available")
  set(CUDNN_AVAIL OFF)
endif()

find_package(OpenCV REQUIRED)

#添加OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

#显示OpenCV_INCLUDE_DIRS的值
message(${OpenCV_INCLUDE_DIRS})

add_library(TRTModel SHARED ../common/TRTModelBase.cpp)
set(CUDA_LIB_PATH /usr/local/cuda/lib64/)

add_executable(SSD SSDSample.cpp)
target_link_libraries(SSD 
                         ${CUDA_LIBRARY} 
                         ${CUDA_RUNTIME_LIBRARY} 
                         TRTModel
                         nvinfer_plugin
                         nvonnxparser
                         nvcaffe_parser
                         nvinfer    
                         ${OpenCV_LIBS}             
                         ${CUDA_LIB_PATH}libcudart.so                         
                         )
                         
 


