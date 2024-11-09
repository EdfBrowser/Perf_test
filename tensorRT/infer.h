#pragma once
#include <NvInfer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "defines.h"
#include "util.h"

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    // suppress info-level messages
    if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
  }
} logger;

class DLL_API trt_infer {
 public:
  trt_infer::trt_infer(std::string model_file);

  template <typename T>
  std::vector<float> run_model(std::vector<T> batch);

 private:
  // 释放顺序
  std::shared_ptr<nvinfer1::IRuntime> runtime;
  std::shared_ptr<nvinfer1::ICudaEngine> engine;
  std::shared_ptr<nvinfer1::IExecutionContext> context;
  int32_t trt_num;
  std::string in_name, out_name;
  nvinfer1::DataType in_type, out_type;
  nvinfer1::Dims in_dims, out_dims;
  std::uint64_t in_size, out_size;
};

template <typename T>
std::vector<float> trt_infer::run_model(std::vector<T> batch) {
  // malloc memory
  void *in_mem{nullptr}, *out_mem{nullptr};
  cudaStream_t stream(nullptr);

  cudaError_t result;
  // malloc memory
  result = cudaMalloc(&in_mem, in_size);
  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate input memory: "
              << cudaGetErrorString(result) << std::endl;
    return {};
  }

  result = cudaMalloc(&out_mem, out_size);
  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate output memory: "
              << cudaGetErrorString(result) << std::endl;
    cudaFree(in_mem);
    return {};
  }

  result = cudaStreamCreate(&stream);
  if (result != cudaSuccess) {
    std::cerr << "Failed to create stream: " << cudaGetErrorString(result)
              << std::endl;
    cudaFree(in_mem);
    cudaFree(out_mem);
    return {};
  }

  // copy input to device
  result = cudaMemcpyAsync(in_mem, batch.data(), in_size,
                           cudaMemcpyHostToDevice, stream);
  if (result != cudaSuccess) {
    std::cerr << "Failed to copy input memory: " << cudaGetErrorString(result)
              << std::endl;
    cudaFree(in_mem);
    cudaFree(out_mem);
    cudaStreamDestroy(stream);
    return {};
  }

  // 等待流完成
  cudaStreamSynchronize(stream);

  void* binding[] = {in_mem, out_mem};

  bool isinferred = context->enqueueV2(binding, stream, nullptr);

  std::vector<float> out_buffer(out_size);
  // copy out from device
  result = cudaMemcpyAsync(out_buffer.data(), out_mem, out_size,
                           cudaMemcpyDeviceToHost, stream);
  if (result != cudaSuccess) {
    std::cerr << "Failed to copy output memory: " << cudaGetErrorString(result)
              << std::endl;
    cudaFree(in_mem);
    cudaFree(out_mem);
    cudaStreamDestroy(stream);
    return {};
  }

  // 等待输出复制完成
  cudaStreamSynchronize(stream);

  // 释放内存
  cudaFree(in_mem);
  cudaFree(out_mem);
  cudaStreamDestroy(stream);

  auto x = softmax(out_buffer, 3600, 3, 1);

  return x;
}
