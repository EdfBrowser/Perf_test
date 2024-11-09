cmake_minimum_required(VERSION 3.29)

project(util C CXX)

# build som util lib
add_library(util SHARED util.cc)
target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(util PRIVATE DLL_EXPORTS)