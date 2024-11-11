@echo off
setlocal

:: 设置 Visual Studio 的安装路径，这个路径可能需要根据你的安装进行调整
set "VS_PATH=C:\\Program Files\\Microsoft Visual Studio\\2022\\Community"

:: 设置 Visual Studio 工具的路径
call "%VS_PATH%\\VC\\Auxiliary\\Build\\vcvars64.bat"  -vcvars_ver=14.2

:: 设置工作区目录
set "WORKSPACE=%~dp0"

:: 配置
cmake -B "%WORKSPACE%\\build" ^
      -DCMAKE_CXX_COMPILER=cl ^
      -DCMAKE_C_COMPILER=cl ^
      -Donnxruntime_USE_CUDA=ON ^
      -Dlibtorch_USE_CUDA=ON ^
      -DCAFFE2_USE_CUDNN=ON ^
      -DCMAKE_BUILD_TYPE=Release ^
      -G Ninja ^
      -S "%WORKSPACE%"

:: 构建
cmake --build "%WORKSPACE%\\build"

endlocal