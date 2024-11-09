#include "eeg_data.h"

std::pair<int32_t, int32_t> eeg_handle::get_eeg_pack_len(int32_t input_len) {
  int one_pack_binary_len = 100;
  if (input_len % 100 != 0) {
    one_pack_binary_len = 98;
  }

  int pre_len = input_len / one_pack_binary_len;
  if (pre_len >= ONE_HOUR_PACKS) {
    pre_len = ONE_HOUR_PACKS;
  }

  return {pre_len, one_pack_binary_len};
}
/// @brief 切片
/// @param v
/// @param start
/// @param end
/// @return
std::vector<char> eeg_handle::slice(const std::vector<char> &v, int32_t start,
                                    int32_t end) {
  if (start >= v.size()) {
    throw std::out_of_range("Start index is out of range!");
  }
  if (end > v.size()) {
    throw std::out_of_range("End index is out of range!");
  }

  return std::vector<char>(v.begin() + start, v.begin() + end);
}

/// @brief 互换行和列，同时赋为负值
/// @param matrix
/// @return
std::vector<std::vector<int16_t>> eeg_handle::transpose(
    const std::vector<std::vector<int16_t>> &matrix) {
  if (matrix.empty()) return {};

  int32_t rowCount = matrix.size();
  int32_t colCount = matrix[0].size();

  std::vector<std::vector<int16_t>> transposed(colCount,
                                               std::vector<int16_t>(rowCount));

  for (size_t i = 0; i < rowCount; ++i) {
    for (size_t j = 0; j < colCount; ++j) {
      transposed[j][i] = -matrix[i][j];
    }
  }

  return transposed;
}

/// @brief 读取eeg数据
/// @param buf
/// @param pack_num
/// @param e_pack_len
/// @return
std::vector<std::vector<int16_t>> eeg_handle::process_eeg_data(
    const std::vector<char> &buf, int32_t pack_num, int32_t e_pack_len) {
  std::vector<char> eeg_body = slice(buf, head_size, buf_size);

  std::vector<std::vector<int16_t>> epoch_list;
  epoch_list.reserve(pack_num);  // 提前分配空间

  for (int i = 0; i < pack_num; ++i) {
    int start_idx = i * e_pack_len + start_offset;

    std::vector<int16_t> block_vec(read_count);
    std::memcpy(block_vec.data(), eeg_body.data() + start_idx,
                read_count * SHORT_SIZE);

    epoch_list.emplace_back(block_vec);
  }

  return transpose(epoch_list);
}

/// @brief 打开eeg文件
/// @param file_path
/// @return
bool eeg_handle::open_eeg(const std::string &file_path) {
  std::ifstream file;
  file.open(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "文件打开失败" << std::endl;
    return false;
    file.close();
  }

  buf_size = file.tellg();
  file.seekg(0, std::ios::beg);

  buf.resize(buf_size);
  file.read(buf.data(), buf.size());

  file.close();

  std::vector<char> head_vec = slice(buf, 0, 4);
  // TODO: 合适的转换
  head_size = *reinterpret_cast<int *>(head_vec.data());

  return true;
}

/// @brief 总入口
/// @param file_path
/// @return
std::vector<std::vector<int16_t>> eeg_handle::get_lead_data(
    const std::string &file_path) {
  if (!open_eeg(file_path)) {
    return {};
  }

  auto [pack_num, e_pack_len] = get_eeg_pack_len(buf_size - head_size);
  return process_eeg_data(buf, pack_num, e_pack_len);
}