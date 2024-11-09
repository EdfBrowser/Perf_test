cmake_minimum_required(VERSION 3.29)

project(test)

add_executable(test main.cc ResourceMonitor.cc)

target_link_libraries(test PRIVATE onnx libtorch onnx trt Psapi)
target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(test
PRIVATE MODEL_DIR="${MODEL_DIR}"
PRIVATE DATA_DIR="${DATA_DIR}"
PRIVATE RESULT_DIR="${RESULT_DIR}")

function(copy)
  foreach(dll ${ARGV})
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different 
      ${dll}  
      $<TARGET_FILE_DIR:${PROJECT_NAME}>)
  endforeach()
endfunction()

if (MSVC)
  file(GLOB TORCH_DLLS "$ENV{TORCH_INSTALL_PREFIX}/lib/*.dll")
  file(GLOB ORT_DLLS "$ENV{ONNXRUNTIME_ROOTDIR}/lib/*.dll")
  file(GLOB TRT_DLLS "$ENV{TENSORRT_ROOTDIR}/lib/*.dll")
  
  copy(${TORCH_DLLS})
  copy(${ORT_DLLS})
  copy(${TRT_DLLS})
  copy(
    $<TARGET_FILE:eeg_data>
    $<TARGET_FILE:butterworth>
    $<TARGET_FILE:util>
    $<TARGET_FILE:libtorch>
    $<TARGET_FILE:onnx>
    $<TARGET_FILE:trt>)
endif()