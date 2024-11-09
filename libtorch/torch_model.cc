#include "torch_model.h"

std::vector<float> tensor_vector(const at::Tensor& tensor) {
  // 确保 tensor 是在 CPU 上并且是连续的
  at::Tensor tensor_contiguous =
      (tensor.is_cpu() ? tensor : tensor.to(at::kCPU))
          .to(at::kFloat)
          .contiguous();

  // 创建 vector 并复制数据
  std::vector<float> vec(
      tensor_contiguous.data_ptr<float>(),
      tensor_contiguous.data_ptr<float>() + tensor_contiguous.numel());

  return vec;
}

torch_model::torch_model(const std::string& model_path)
    : model_file(model_path) {
  module = torch::jit::load(model_file);

  module.to(dev);

  module.eval();
}

std::vector<float> torch_model::run_model(std::vector<float>& input_data,
                                          const std::vector<int64_t>& shape) {
  auto options = c10::TensorOptions().dtype(at::kFloat).requires_grad(true);
  auto lead_x = torch::from_blob(input_data.data(), shape, options);
  lead_x = lead_x.contiguous().to(dev);

  // Disable gradient calculation
  torch::NoGradGuard no_grad;

  at::Tensor output;
  {
    // Get model prediction
    output = module.forward({lead_x}).toTensor();
    // .to(at::Half);  // float16
    // output = torch::nn::functional::softmax(output, 1);
  }

  std::vector<float> vec = tensor_vector(output);

  return softmax(vec, 3600, 3, 1);
}
