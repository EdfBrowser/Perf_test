cmake_minimum_required(VERSION 3.29 FATAL_ERROR)

project(eeg_data C CXX)

add_library(${PROJECT_NAME} SHARED
eeg_data.cc)
target_link_libraries(${PROJECT_NAME} PRIVATE util)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE DLL_EXPORTS)