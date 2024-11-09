#include "infer.h"

uint64_t get_memorysize(const nvinfer1::Dims& dims, const int32_t elem_size) {
  return std::accumulate(dims.d, dims.d + dims.nbDims, 1,
                         std::multiplies<int64_t>()) *
         elem_size;
}

int64_t open_model(std::string file, std::vector<char>& engine_data) {
  std::ifstream fs(file, std::ios::binary);
  if (!fs.is_open()) {
    std::cerr << "Failed to open the model!" << "\n";
    exit(-1);
  }

  // file_size
  fs.seekg(0, std::ios::end);
  int64_t file_size = fs.tellg();
  fs.seekg(0);
  engine_data.resize(file_size);
  // read to a vector
  fs.read(engine_data.data(), file_size);
  // close fs
  fs.close();
  return file_size;
}

trt_infer::trt_infer(std::string model_file) {
  std::vector<char> engine_data;
  int64_t file_size;
  file_size = open_model(model_file, engine_data);
  assert(engine_data.size() == file_size);

  runtime =
      std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  engine = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engine_data.data(), file_size));
  context = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine->createExecutionContext());

  trt_num = engine->getNbIOTensors();
  assert(trt_num == 2);
  // TODO: iterator
  in_name = engine->getBindingName(0);
  out_name = engine->getBindingName(1);

  in_type = engine->getTensorDataType(in_name.data());
  out_type = engine->getTensorDataType(out_name.data());

  // assume the data type is float
  assert(in_type == nvinfer1::DataType::kFLOAT &&
         out_type == nvinfer1::DataType::kFLOAT);

  in_dims = engine->getTensorShape(in_name.data());
  out_dims = engine->getTensorShape(out_name.data());

  in_size = get_memorysize(in_dims, sizeof(float));
  out_size = get_memorysize(out_dims, sizeof(float));
}

template std::vector<float> trt_infer::run_model(std::vector<float> batch);