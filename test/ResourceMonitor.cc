#include "ResourceMonitor.h"

ResourceMonitor::ResourceMonitor(const std::string& logFile) {
  logfile.open(logFile, std::ios::out | std::ios::app);
  if (!logfile) {
    throw std::runtime_error("Unable to open log file");
  }
  // Write CSV header
  logfile
      << "Section,Elapsed Time (s),CPU Time (s),Physical Memory (KB),Virtual "
         "Memory (KB),CUDA Memory Used (KB),CUDA Total Memory (KB)\n";
}

void ResourceMonitor::start() {
  cpuTimeStart = getCpuTime();
  startTime = std::chrono::high_resolution_clock::now();
  recordStartTime = startTime;
}

void ResourceMonitor::record() {
  auto now = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = now - recordStartTime;
  SIZE_T physicalMemUsed, virtualMemUsed;
  getMemoryUsage(physicalMemUsed, virtualMemUsed);
  size_t freeMem, totalMem;
  getCudaMemoryUsage(freeMem, totalMem);

  elapsedTimes.push_back(elapsed);
  cpuTimes.push_back((getCpuTime() - cpuTimeStart) / 1000.0);
  physicalMemoryUsed.push_back(physicalMemUsed / 1024);
  virtualMemoryUsed.push_back(virtualMemUsed / 1024);
  cudaMemoryUsed.push_back((totalMem - freeMem) / 1024);
  cudaTotalMemory.push_back(totalMem / 1024);

  recordStartTime = now;  // Reset for the next record
}

void ResourceMonitor::stop(const std::string& sectionName) {
  logfile << sectionName << "," << elapsedTimes.back().count() << ","
          << cpuTimes.back() << "," << physicalMemoryUsed.back() << ","
          << virtualMemoryUsed.back() << "," << cudaMemoryUsed.back() << ","
          << cudaTotalMemory.back() << "\n";
}

ResourceMonitor::~ResourceMonitor() {
  if (logfile.is_open()) {
    logfile.close();
  }
}

double ResourceMonitor::getCpuTime() {
  FILETIME creationTime, exitTime, kernelTime, userTime;
  GetProcessTimes(GetCurrentProcess(), &creationTime, &exitTime, &kernelTime,
                  &userTime);

  ULARGE_INTEGER kTime;
  kTime.LowPart = kernelTime.dwLowDateTime;
  kTime.HighPart = kernelTime.dwHighDateTime;

  ULARGE_INTEGER uTime;
  uTime.LowPart = userTime.dwLowDateTime;
  uTime.HighPart = userTime.dwHighDateTime;

  return (kTime.QuadPart + uTime.QuadPart) / 10000.0;  // 转换为毫秒
}

void ResourceMonitor::getMemoryUsage(SIZE_T& physicalMemUsed,
                                     SIZE_T& virtualMemUsed) {
  PROCESS_MEMORY_COUNTERS_EX pmc;
  GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc,
                       sizeof(pmc));
  physicalMemUsed = pmc.WorkingSetSize;
  virtualMemUsed = pmc.PrivateUsage;
}

void ResourceMonitor::getCudaMemoryUsage(size_t& freeMem, size_t& totalMem) {
  cudaMemGetInfo(&freeMem, &totalMem);
}
