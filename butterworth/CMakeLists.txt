cmake_minimum_required(VERSION 3.29)

project(butterworth C CXX)

add_library(${PROJECT_NAME} SHARED
biquad.cpp
butterworth.cpp
filter_design.cpp
utils.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE util)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE DLL_EXPORTS)
