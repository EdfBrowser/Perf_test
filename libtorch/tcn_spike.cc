#include "tcn_spike.h"

void TemporalBlockImpl::init_weights() {
    conv1->weight.data().normal_(0, 0.01);
    conv2->weight.data().normal_(0, 0.01);
    if (downsample) {
        downsample->weight.data().normal_(0, 0.01);
    }
}

torch::Tensor Chomp1dImpl::forward(torch::Tensor x) {
    return torch::narrow(x, 2,0,x.size(2) - chomp_size).contiguous();
}

torch::Tensor TemporalBlockImpl::forward(torch::Tensor x) {
    auto out = net->forward(x);
    auto res = downsample ? downsample->forward(x) : x;
    return relu->forward(out + res);
}

torch::Tensor TemporalConvNetImpl::forward(torch::Tensor x) {
    return network->forward(x);
}

torch::Tensor TCNImpl::forward(torch::Tensor x) {
    x = tcn->forward(x);
    x = gap->forward(x).squeeze(-1);
    if (fc_do) 
        x = fc_do->forward(x);
    return fc->forward(x);
}