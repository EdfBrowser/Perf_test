#include "onnx_model.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>

int64_t multiplyVectorElements(const std::vector<int64_t>& vec) {
  // 使用 std::accumulate 和 lambda 表达式来实现乘法
  return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<int>());
}

onnx_model::onnx_model(const std::wstring& model_path)
    : model_file(model_path.begin(), model_path.end()),
      env(ORT_LOGGING_LEVEL_WARNING, "default"),
      session(nullptr) {
  // Configure session options for CUDA
  Ort::SessionOptions session_opt;
#ifdef USE_CUDA
  try {
    // session_opt.SetIntraOpNumThreads(
    //     1);  // Set the number of threads for intra-op parallelism
    session_opt.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);  // Set optimization level

    // OrtCUDAProviderOptions cuda_opts;
    //  cuda_opts.device_id = 0;

    // Enable CUDA
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_opt, 0);
    // session_opt.AppendExecutionProvider_CUDA(cuda_opts);

  } catch (const Ort::Exception& e) {
    std::cerr << "Failed to create ONNX Runtime GPU session: " << e.what()
              << std::endl;
    throw;
  }
#endif

  // Create the session with the model file and session options
  session = Ort::Session(env, model_file.c_str(), session_opt);
}

/// @brief 模型预测
/// @param model_path
/// @param result_file
std::vector<float> onnx_model::run_model(const std::vector<float>& input_data) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::int64_t> input_shapes;

  for (std::size_t i = 0; i < session.GetInputCount(); i++) {
    input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    input_shapes =
        session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
  }
  // some models might have negative shape values to indicate dynamic shape,
  // e.g., for variable batch size.
  for (auto& s : input_shapes) {
    if (s < 0) {
      s = 1;
    }
  }
  std::vector<std::string> output_names;
  std::vector<std::int64_t> output_shapes;
  for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
    output_names.emplace_back(
        session.GetOutputNameAllocated(i, allocator).get());
    output_shapes =
        session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
  }

  int64_t out_shape_size = multiplyVectorElements(output_shapes);
  std::vector<float> output_buf(out_shape_size);

  // Assume model has 1 input node and 1 output node.
  assert(input_names.size() == 1 && output_names.size() == 1);

  auto input_shape = input_shapes;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;
  input_tensors.emplace_back(vec_to_tensor<float>(input_data, input_shape));

  // double-check the dimensions of the input tensor
  assert(input_tensors[0].IsTensor() &&
         input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() ==
             input_shape);

  // pass data through model
  std::vector<const char*> input_names_char(input_names.size(), nullptr);
  std::transform(std::begin(input_names), std::end(input_names),
                 std::begin(input_names_char),
                 [&](const std::string& str) { return str.c_str(); });

  std::vector<const char*> output_names_char(output_names.size(), nullptr);
  std::transform(std::begin(output_names), std::end(output_names),
                 std::begin(output_names_char),
                 [&](const std::string& str) { return str.c_str(); });

  try {
    auto output_tensors =
        session.Run(Ort::RunOptions{nullptr}, input_names_char.data(),
                    input_tensors.data(), input_names_char.size(),
                    output_names_char.data(), output_names_char.size());

    // double-check the dimensions of the output tensors
    assert(output_tensors.size() == output_names.size() &&
           output_tensors[0].IsTensor());

    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    std::copy(floatarr, floatarr + out_shape_size, output_buf.data());

    assert(output_buf.size() != 0);

  } catch (const Ort::Exception& exception) {
    std::cout << "ERROR running model inference: " << exception.what()
              << std::endl;
    exit(-1);
  }

  std::vector<float> softmax_vec = softmax(output_buf, 3600, 3, 1);

  return softmax_vec;
}

/// @brief 数组转成tensor
/// @tparam T
/// @param data
/// @param shape
/// @return
template <typename T>
Ort::Value onnx_model::vec_to_tensor(const std::vector<T>& data,
                                     const std::vector<int64_t>& shape) {
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  return Ort::Value::CreateTensor<T>(mem_info, const_cast<T*>(data.data()),
                                     data.size(), shape.data(), shape.size());
}