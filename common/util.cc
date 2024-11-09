#include "util.h"

std::vector<float> softmax(const std::vector<float>& x, int rows, int cols,
                           int axis) {
  std::vector<float> result;

  // Perform softmax along rows (axis = 0) or along columns (axis = 1)
  if (axis == 0) {
    // Softmax along columns
    for (int col = 0; col < cols; ++col) {
      float max_val = x[col];
      for (int row = 1; row < rows; ++row) {
        if (x[row * cols + col] > max_val) {
          max_val = x[row * cols + col];
        }
      }

      std::vector<float> exp_sum(rows, 0.0);
      float sum_exp = 0.0;

      for (int row = 0; row < rows; ++row) {
        exp_sum[row] = exp(x[row * cols + col] - max_val);
        sum_exp += exp_sum[row];
      }

      for (int row = 0; row < rows; ++row) {
        result.emplace_back(exp_sum[row] / sum_exp);
      }
    }
  } else if (axis == 1 || axis == -1) {
    // Softmax along rows (default or specified explicitly)
    for (int row = 0; row < rows; ++row) {
      float max_val = x[row * cols];
      for (int col = 1; col < cols; ++col) {
        if (x[row * cols + col] > max_val) {
          max_val = x[row * cols + col];
        }
      }

      std::vector<float> exp_sum(cols, 0.0);
      float sum_exp = 0.0;

      for (int col = 0; col < cols; ++col) {
        exp_sum[col] = exp(x[row * cols + col] - max_val);
        sum_exp += exp_sum[col];
      }

      for (int col = 0; col < cols; ++col) {
        result.emplace_back(exp_sum[col] / sum_exp);
      }
    }
  } else {
    // Invalid axis
    std::cerr << "Invalid axis specified for softmax." << std::endl;
  }

  return result;
}

// Explicit instantiation for the types you need
template <>
void printer::dump_csv(const std::vector<float>& predictions);