#include <chrono>
#include <codecvt>
#include <locale>

#include "ResourceMonitor.h"
#include "butterworth.h"
#include "eeg_data.h"
#include "infer.h"
#include "onnx_model.h"
#include "torch_model.h"
#include "util.h"

eeg_handle eeg;
// std::chrono::high_resolution_clock::time_point start;
// std::chrono::high_resolution_clock::time_point finish;

void test_trt(std::vector<std::vector<int16_t>>& lead_data, butterworth& filter,
              std::string& model_file, std::string& result_file,
              std::vector<double>& freq) {
  trt_infer infer(model_file);
  printer p(result_file);

  // start = std::chrono::high_resolution_clock::now();  // 开始计时

  // TODO: 优化
  for (int i = 0; i < lead_data.size(); i++) {
    std::vector<double> lead_data_float(lead_data[i].begin(),
                                        lead_data[i].end());
    std::vector<double> process_data = filter.process(lead_data_float);
    std::vector<float> flat_data(process_data.begin(), process_data.end());
    std::vector<float> sofx = infer.run_model(flat_data);
    p.dump_csv(sofx);
  }

  // finish = std::chrono::high_resolution_clock::now();  // 结束计时
  // std::cout << "耗时为:"
  //           << std::chrono::duration_cast<std::chrono::seconds>(finish -
  //           start)
  //                  .count()
  //           << "s.\n";
}

void test_onnx(std::vector<std::vector<int16_t>>& lead_data,
               butterworth& filter, std::wstring& model_file,
               std::string& result_file, std::vector<double>& freq) {
  // Create and run model
  onnx_model model(model_file);
  printer p(result_file);
  // start = std::chrono::high_resolution_clock::now();  // 开始计时

  for (int i = 0; i < lead_data.size(); i++) {
    std::vector<double> lead_data_float(lead_data[0].begin(),
                                        lead_data[0].end());
    std::vector<double> process_data = filter.process(lead_data_float);
    std::vector<float> flat_data(process_data.begin(), process_data.end());
    std::vector<float> sofx = model.run_model(flat_data);
    p.dump_csv(sofx);
  }

  // finish = std::chrono::high_resolution_clock::now();  // 结束计时
  // std::cout << "耗时为:"
  //           << std::chrono::duration_cast<std::chrono::seconds>(finish -
  //           start)
  //                  .count()
  //           << "s.\n";
}

void test_torch(std::vector<std::vector<int16_t>>& lead_data,
                butterworth& filter, std::string& model_file,
                std::string& result_file, std::vector<double>& freq) {
  // Create and run model
  torch_model model(model_file);
  printer p(result_file);
  // start = std::chrono::high_resolution_clock::now();  // 开始计时

  for (int i = 0; i < lead_data.size(); i++) {
    std::vector<double> lead_data_float(lead_data[0].begin(),
                                        lead_data[0].end());
    std::vector<double> process_data = filter.process(lead_data_float);
    std::vector<float> flat_data(process_data.begin(), process_data.end());
    std::vector<float> sofx = model.run_model(flat_data, {3600, 1, 256});
    p.dump_csv(sofx);
  }

  // finish = std::chrono::high_resolution_clock::now();  // 结束计时
  // std::cout << "耗时为:"
  //           << std::chrono::duration_cast<std::chrono::seconds>(finish -
  //           start)
  //                  .count()
  //           << "s.\n";
}
/*
void test_torch(std::vector<std::vector<int16_t>>& lead_data,
                std::string& model_file, std::string& result_file,
                std::vector<double>& freq) {
  torch_model model(model_file);
  start = std::chrono::high_resolution_clock::now();  // 开始计时

  for (int i = 0; i < 3600; i++) {
    auto flat_data = filter_process(lead_data, freq);
    auto sofx = model.run_model(flat_data, {3600, 1, 256});
    printer p(result_file);
    p.dump_csv(sofx);
  }

  finish = std::chrono::high_resolution_clock::now();  // 结束计时
  printer_time();
}
*/

int main() {
  std::string model_folder = MODEL_DIR;
  std::string result_folder = RESULT_DIR;
  std::string eeg_folder = DATA_DIR;
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

  std::string trt_model = model_folder + "/best.trt";
  std::string torch_model = model_folder + "/best.pt";
  std::wstring wf = converter.from_bytes("/best.onnx");
  std::wstring wd = converter.from_bytes(model_folder);
  std::wstring onnx_model = wd + wf;

  std::string trt_result = result_folder + "/trt_result.txt";
  std::string onnx_result = result_folder + "/onnx_result.txt";
  std::string torch_result = result_folder + "/torch_result.txt";

  std::string eeg_path = eeg_folder + "/1.eeg";
  // Open eeg data
  std::vector<std::vector<int16_t>> lead_data = eeg.get_lead_data(eeg_path);

  // Process data
  std::vector<double> freq = {0.8, 35};
  butterworth filter(5, freq, filter_design::filter_type::bandpass, 256);

  ResourceMonitor monitor(result_folder + "/resource_usage_log.txt");

  std::cout << "Running\n";

  // 测量test_trt的运行时间和资源使用情况
  monitor.start();
  test_trt(lead_data, filter, trt_model, trt_result, freq);
  monitor.record();
  monitor.stop("test_trt");

  // 测量test_onnx的运行时间和资源使用情况
  monitor.start();
  test_onnx(lead_data, filter, onnx_model, onnx_result, freq);
  monitor.record();
  monitor.stop("test_onnx");

  // 测量test_torch的运行时间和资源使用情况
  monitor.start();
  test_torch(lead_data, filter, torch_model, torch_result, freq);
  monitor.record();
  monitor.stop("test_torch");

  std::cout << "Done\n";

  std::cin.get();
  return 0;
}