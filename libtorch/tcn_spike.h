#pragma once

#include "torch/torch.h"
#include "torch/utils.h"

// Chomp1d layer
class Chomp1dImpl : public torch::nn::Module {
 public:
  Chomp1dImpl(int64_t size) { chomp_size = size; }

  torch::Tensor forward(torch::Tensor x);

  int64_t chomp_size;
};
TORCH_MODULE(Chomp1d);

// TemporalBlock layer
class TemporalBlockImpl : public torch::nn::Module {
 public:
  TemporalBlockImpl(int64_t n_inputs, int64_t n_outputs, int64_t kernel_size,
                    int64_t stride, int64_t dilation, int64_t padding,
                    double dropout) {
    // TODO: weight_norm函数没找到
    conv1 = torch::nn::Conv1d(
        torch::nn::Conv1dOptions(n_inputs, n_outputs, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation));
    conv2 = torch::nn::Conv1d(
        torch::nn::Conv1dOptions(n_outputs, n_outputs, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation));
    chomp1 = Chomp1d(padding);
    chomp2 = Chomp1d(padding);
    relu1 = torch::nn::ReLU();
    relu2 = torch::nn::ReLU();
    dropout1 = torch::nn::Dropout(dropout);
    dropout2 = torch::nn::Dropout(dropout);

    net = torch::nn::Sequential(conv1, chomp1, relu1, dropout1, conv2, chomp2,
                                relu2, dropout2);

    downsample =
        torch::nn::Conv1d(torch::nn::Conv1dOptions(n_inputs, n_outputs, 1));
    relu = torch::nn::ReLU();

    register_module("downsample", downsample);
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("chomp1", chomp1);
    register_module("chomp2", chomp2);
    register_module("relu1", relu1);
    register_module("relu2", relu2);
    register_module("dropout1", dropout1);
    register_module("dropout2", dropout2);
    register_module("net", net);
    register_module("relu", relu);

    init_weights();
  }

 public:
  void init_weights();
  torch::Tensor forward(torch::Tensor x);

  torch::nn::Conv1d conv1{nullptr}, conv2{nullptr}, downsample{nullptr};
  Chomp1d chomp1{nullptr}, chomp2{nullptr};
  torch::nn::ReLU relu1{nullptr}, relu2{nullptr}, relu{nullptr};
  torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
  torch::nn::Sequential net{nullptr};
};
TORCH_MODULE(TemporalBlock);

// TemporalConvNet layer
class TemporalConvNetImpl : public torch::nn::Module {
 public:
  TemporalConvNetImpl(int64_t num_inputs, std::vector<int64_t> num_channels,
                      int64_t kernel_size, double dropout) {
    int64_t num_levels = 8;
    std::vector<int64_t> nf(8, 25);

    for (int64_t i = 0; i < num_levels; ++i) {
      int64_t dilation_size = 1 << i;
      int64_t in_channels = i == 0 ? num_inputs : num_channels[i - 1];
      int64_t out_channels = nf[i];
      network->push_back(TemporalBlock(
          in_channels, out_channels, kernel_size, 1, dilation_size,
          (kernel_size - 1) * dilation_size, dropout));
    }

    register_module("network", network);
  }

 public:
  torch::Tensor forward(torch::Tensor x);

  torch::nn::Sequential network;
};
TORCH_MODULE(TemporalConvNet);

// TCN model
class TCNImpl : public torch::nn::Module {
 public:
  TCNImpl(int64_t input_size, int64_t output_size,
          std::vector<int64_t> num_channels, int64_t kernel_size,
          double dropout) {
    tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout);
    linear = torch::nn::Linear(num_channels.back(), output_size);
    fc = torch::nn::Linear(25, output_size);
    gap = torch::nn::AdaptiveAvgPool1d(1);
    fc_do = torch::nn::Dropout(fc_dropout);

    register_module("tcn", tcn);
    register_module("linear", linear);
    register_module("fc", fc);
    register_module("gap", gap);
    register_module("fc_do", fc_do);
  }

 public:
  torch::Tensor forward(torch::Tensor x);

  double fc_dropout = 0.15;
  TemporalConvNet tcn{nullptr};
  torch::nn::Linear linear{nullptr}, fc{nullptr};
  torch::nn::AdaptiveAvgPool1d gap{nullptr};
  torch::nn::Dropout fc_do{nullptr};
};
TORCH_MODULE(TCN);