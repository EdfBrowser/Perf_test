#pragma once
#include <fstream>
#include <iostream>
#include <vector>

#include "defines.h"

// Function to calculate softmax along a specific axis
DLL_API std::vector<float> softmax(const std::vector<float>& x, int rows,
                                   int cols, int axis = -1);

class DLL_API printer {
 public:
  ~printer() = default;  // TODO: 不知道要不要手动关闭
  printer(std::string& file) : fs(file, std::ios_base::out) {
    if (!fs.is_open()) {
      std::cerr << "Failed to open " << file << std::endl;
    }
  }

  template <typename T>
  void dump_csv(const std::vector<T>& predictions);  // dump data to file

 private:
  std::fstream fs;
};

template <typename T>
void printer::dump_csv(const std::vector<T>& predictions) {
  for (const auto& pred : predictions) {
    fs << pred << ",";
  }

  fs << std::endl;
}
