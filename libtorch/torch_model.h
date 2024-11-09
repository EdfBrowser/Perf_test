#pragma once

#include <fstream>
#include <iostream>
#include <vector>

// #include "tcn_spike.h"
#include "defines.h"
#include "torch/cuda.h"
#include "torch/data.h"
#include "torch/script.h"
#include "torch/serialize.h"
#include "torch/torch.h"
#include "torch/types.h"
#include "torch/utils.h"
#include "util.h"

class DLL_API torch_model {
 public:
  torch_model(const std::string& model_path);

  std::vector<float> run_model(std::vector<float>& input_data,
                               const std::vector<int64_t>& shape);

 private:
  std::string model_file;
  torch::jit::script::Module module;
  // TODO USE MACRO TO CONTROL DEVICE TYPE
  c10::DeviceType dev = torch::kCUDA;
};