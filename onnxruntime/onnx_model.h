#ifndef ONNX_MODEL_H
#define ONNX_MODEL_H

#include <string>
#include <vector>

#include "defines.h"
#include "onnxruntime_cxx_api.h"
#include "util.h"

class DLL_API onnx_model {
 public:
  onnx_model(const std::wstring& model_path);

  std::vector<float> run_model(const std::vector<float>& input_data);

 private:
  std::basic_string<wchar_t> model_file;
  Ort::Env env;
  Ort::Session session;

  template <typename T>
  Ort::Value vec_to_tensor(const std::vector<T>& data,
                           const std::vector<int64_t>& shape);
};

#endif  // ONNX_MODEL_H
