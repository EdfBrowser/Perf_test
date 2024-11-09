#ifndef RESOURCEMONITOR_H
#define RESOURCEMONITOR_H

// 在包含 windows.h 之前定义 NOMINMAX
#define NOMINMAX

#include <cuda_runtime.h>
#include <windows.h>
#define PSAPI_VERSION 1
#include <psapi.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

class ResourceMonitor {
 public:
  ResourceMonitor(const std::string& logFile);
  void start();
  void record();
  void stop(const std::string& sectionName);
  ~ResourceMonitor();

 private:
  double getCpuTime();
  void getMemoryUsage(SIZE_T& physicalMemUsed, SIZE_T& virtualMemUsed);
  void getCudaMemoryUsage(size_t& freeMem, size_t& totalMem);

  std::ofstream logfile;
  double cpuTimeStart;
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
  std::chrono::time_point<std::chrono::high_resolution_clock> recordStartTime;
  std::vector<std::string> sectionNames;
  std::vector<std::chrono::duration<double>> elapsedTimes;
  std::vector<double> cpuTimes;
  std::vector<SIZE_T> physicalMemoryUsed;
  std::vector<SIZE_T> virtualMemoryUsed;
  std::vector<size_t> cudaMemoryUsed;
  std::vector<size_t> cudaTotalMemory;
};

#endif  // RESOURCEMONITOR_H