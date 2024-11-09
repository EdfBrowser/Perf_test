#ifndef EEG_DATA_H
#define EEG_DATA_H

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "defines.h"

class DLL_API eeg_handle {
 private:
  bool open_eeg(const std::string &file_path);
  std::pair<int32_t, int32_t> get_eeg_pack_len(int32_t input_len);
  std::vector<char> slice(const std::vector<char> &v, int32_t start,
                          int32_t end);
  std::vector<std::vector<int16_t>> eeg_handle::transpose(
      const std::vector<std::vector<int16_t>> &matrix);
  std::vector<std::vector<int16_t>> process_eeg_data(
      const std::vector<char> &buf, int32_t pack_num, int32_t e_pack_len);

 public:
  std::vector<std::vector<int16_t>> get_lead_data(const std::string &file_path);

 private:
  std::vector<char> buf;
  int32_t buf_size;
  int32_t head_size;

 public:
  static const int start_offset = 30;  // 30 hz
  static const int read_count = 34;    // channel
  static const int32_t SAMPLE_RATE = 256;
  static const int32_t ONE_HOUR_PACKS = SAMPLE_RATE * 60 * 60;
  static const int SHORT_SIZE = sizeof(int16_t);
};

#endif  // EEG_DATA_H