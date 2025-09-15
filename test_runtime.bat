@echo off
echo === Testing Perdix Runtime Compilation ===
echo.

REM Set CUDA environment
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set PATH=%CUDA_PATH%\bin;%PATH%

echo Checking CUDA installation...
where nvcc >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] nvcc found
    nvcc --version | findstr "release"
) else (
    echo [WARNING] nvcc not found in PATH
)

echo.
echo Checking NVRTC library...
if exist "%CUDA_PATH%\bin\nvrtc64_120_0.dll" (
    echo [OK] NVRTC DLL found
) else (
    echo [ERROR] NVRTC DLL not found
    exit /b 1
)

echo.
echo Building with runtime compilation...

REM Rename build scripts temporarily
if exist build.rs (
    echo Temporarily disabling build.rs...
    ren build.rs build.rs.bak
)
copy build_runtime.rs build.rs >nul

echo.
echo Compiling Perdix with runtime compilation support...
cargo build --release 2>&1

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Build completed!
    echo.
    echo Running runtime compilation example...
    cargo run --release --example runtime_compile
) else (
    echo.
    echo [ERROR] Build failed
)

REM Restore original build.rs
if exist build.rs.bak (
    del build.rs
    ren build.rs.bak build.rs
)

echo.
echo === Test Complete ===