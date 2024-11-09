param (
    [string]$OnnxUri = "https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-win-x64-gpu-1.17.0.zip",
    [string]$OnnxZipPath = "onnxruntime.zip",
    [string]$OnnxDestinationPath = "C:\\ONNXRUNTIME_PACKAGE",

    [string]$LibtorchUri = "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.3.0%2Bcu118.zip",
    [string]$LibtorchZipPath = "libtorch.zip",
    [string]$LibtorchDestinationPath = "C:\\LIBTORCH_PACKAGE",

    [string]$CuDNNUri = "https://developer.download.nvidia.cn/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.5.0.96_cuda11-archive.zip",
    [string]$CuDNNZipPath = "cudnn.zip",
    [string]$CuDNNDestinationPath = "C:\\CUDNN_PACKAGE",

    [string]$ZlibUri = "http://www.winimage.com/zLibDll/zlib123dllx64.zip",
    [string]$ZlibZipPath = "zlib123dllx64.zip",
    [string]$ZlibDestinationPath = "C:\\ZLIB_PACKAGE"
)

# 下载压缩文件的函数
function Download-Zip {
    param (
        [string]$uri,
        [string]$path
    )
    Write-Host "Downloading file from $uri to $path"
    Invoke-WebRequest -Uri $uri -OutFile $path
}

# 解压缩文件的函数
function Extract-Zip {
    param (
        [string]$zipPath,
        [string]$destinationPath
    )
    Write-Host "Extracting $zipPath to $destinationPath"
    Expand-Archive -Path $zipPath -DestinationPath $destinationPath
}

# 处理解压后内容的函数
function Handle-ExtractedContents {
    param (
        [string]$destinationPath
    )
    $items = Get-ChildItem -Path $destinationPath

    if ($items.Count -eq 1 -and $items[0].PSIsContainer) {
        $innerFolder = $items[0].FullName
        Write-Host "The zip contains an extra layer of folder: $($items[0].Name)"
        
        # 移动内部文件到目的地
        $innerItems = Get-ChildItem -Path $innerFolder
        foreach ($item in $innerItems) {
            Move-Item -Path $item.FullName -Destination $destinationPath
        }

        # 删除多余的文件夹
        Remove-Item -Path $innerFolder -Recurse
    } else {
        Write-Host "The zip does not contain an extra layer of folder."
    }
}

# 自动生成libraries数组
$libraries = @()
$parameterGroups = @{}

# 遍历所有绑定的参数
foreach ($param in $PSBoundParameters.GetEnumerator()) {
    $key = $param.Key
    $value = $param.Value

    if ($key -match "^(.*)(Uri|ZipPath|DestinationPath)$") {
        $baseName = $matches[1]
        if (-not $parameterGroups.ContainsKey($baseName)) {
            $parameterGroups[$baseName] = @{}
        }
        $parameterGroups[$baseName][$matches[2]] = $value
    }
}

# 将参数组转换为自定义对象并添加到libraries数组中
foreach ($group in $parameterGroups.GetEnumerator()) {
    $libraries += [PSCustomObject]@{
        Uri = $group.Value["Uri"]
        ZipPath = $group.Value["ZipPath"]
        DestinationPath = $group.Value["DestinationPath"]
    }
}

# 主脚本执行
foreach ($lib in $libraries) {
    Download-Zip -uri $lib.Uri -path $lib.ZipPath
    Extract-Zip -zipPath $lib.ZipPath -destinationPath $lib.DestinationPath
    Handle-ExtractedContents -destinationPath $lib.DestinationPath
}
