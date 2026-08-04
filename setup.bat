@echo off
rem ============================================
rem  Setup drone-human-detection (terminal)
rem  uv dipakai secara default, pip sebagai fallback
rem  GPU NVIDIA -> onnxruntime-gpu (CUDA 13/cuDNN 9 via pip, tanpa menyentuh toolkit sistem)
rem  Non-NVIDIA -> onnxruntime (CPU)
rem  Idempotent: aman dijalankan ulang.
rem ============================================
setlocal
cd /d "%~dp0"

echo ==========================================
echo  drone-human-detection setup
echo ==========================================
echo.

set HAS_UV=0
where uv >nul 2>nul && set HAS_UV=1

set HAS_NVIDIA=0
where nvidia-smi >nul 2>nul && set HAS_NVIDIA=1

if "%HAS_NVIDIA%"=="1" (
    echo [GPU] NVIDIA terdeteksi - runtime CUDA.
    set EXTRA=gpu
) else (
    echo [GPU] NVIDIA tidak terdeteksi - runtime CPU.
    set EXTRA=cpu
)

if "%HAS_UV%"=="1" (
    echo [uv] uv sync --extra %EXTRA%
    uv sync --extra %EXTRA%
    if errorlevel 1 (
        echo.
        echo [FAIL] uv sync gagal. Coba lagi dengan verbose: uv sync -v
        exit /b 1
    )
    echo.
    echo ==========================================
    echo  Selesai. Jalankan:  uv run python main.py --source tello
    echo  Tanpa drone:       uv run python main.py --source phone
    echo ==========================================
    exit /b 0
)

echo [pip] uv tidak ditemukan - pakai venv lokal + pip.
if not exist .venv\Scripts\python.exe (
    echo [pip] Membuat venv lokal (.venv)...
    python -m venv .venv
    if errorlevel 1 exit /b 1
)
call .venv\Scripts\activate.bat
if "%EXTRA%"=="gpu" (
    python -m pip install -r requirements-gpu.txt
) else (
    python -m pip install -r requirements.txt
)
if errorlevel 1 exit /b 1
echo.
echo ==========================================
echo  Selesai. Jalankan:  .venv\Scripts\python main.py --source tello
echo  Tanpa drone:        .venv\Scripts\python main.py --source phone
echo ==========================================
endlocal
