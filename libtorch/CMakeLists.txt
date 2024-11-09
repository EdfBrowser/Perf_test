cmake_minimum_required(VERSION 3.29)
project(libtorch C CXX) 

option(libtorch_USE_CUDA "Build with CUDA support" OFF)

add_library(${PROJECT_NAME} SHARED torch_model.cc)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE DLL_EXPORTS)

if (libtorch_USE_CUDA)
  target_compile_definitions(${PROJECT_NAME} PRIVATE
    USE_CUDA
  )
endif()
