cmake_minimum_required(VERSION 3.29)

project(trt)

add_library(${PROJECT_NAME} SHARED infer.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE DLL_EXPORTS)