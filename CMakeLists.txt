cmake_minimum_required(VERSION 3.29)

project(multiple_project C CXX)

# find py
find_package(Python QUIET)

if (NOT DEFINED ENV{TENSORRT_ROOTDIR})
  message(FATAL_ERROR "Failed to find tensorRT lib")
endif()

find_package(CUDAToolkit REQUIRED)

# torch
if (NOT DEFINED ENV{TORCH_INSTALL_PREFIX})
  message(FATAL_ERROR "Don`t find libtorch lib")
else()
  list(APPEND CMAKE_PREFIX_PATH "$ENV{TORCH_INSTALL_PREFIX}\\share\\cmake\\Torch")
  find_package(Torch REQUIRED)
endif()

#message(STATUS "ENV: $ENV{ONNXRUNTIME_ROOTDIR}")
if (NOT DEFINED ENV{ONNXRUNTIME_ROOTDIR})
  message(FATAL_ERROR "Failed to find onnxruntime lib")
endif()

set(CMAKE_CXX_STANDARD 17)

add_compile_options("/source-charset:utf-8")

set(MODEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/res/model)
set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/res/data)
set(RESULT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/res/result)




add_subdirectory(common)
add_subdirectory(eeg_data)
add_subdirectory(butterworth)

include_directories(
"$ENV{TENSORRT_ROOTDIR}\\include"
"$ENV{ONNXRUNTIME_ROOTDIR}\\include"
)

link_libraries(
"$ENV{TENSORRT_ROOTDIR}\\lib\\nvinfer.lib"
"$ENV{ONNXRUNTIME_ROOTDIR}\\lib\\onnxruntime.lib" 
CUDA::cudart
${TORCH_LIBRARIES}
util
eeg_data
butterworth)

add_subdirectory(libtorch)
add_subdirectory(onnxruntime)
add_subdirectory(tensorRT)
add_subdirectory(test)