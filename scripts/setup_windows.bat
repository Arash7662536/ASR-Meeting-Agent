@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM Setup script — Windows 10/11
REM Run once from a normal Command Prompt (not PowerShell):
REM   scripts\setup_windows.bat
REM
REM Prerequisites (install manually before running):
REM   1. Python 3.10+ from https://www.python.org/downloads/
REM      Make sure "Add Python to PATH" is checked.
REM   2. ffmpeg from https://ffmpeg.org/download.html
REM      Extract and add the bin\ folder to your PATH.
REM   3. (optional) CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
REM ─────────────────────────────────────────────────────────────────────────────
setlocal enabledelayedexpansion

SET REPO_ROOT=%~dp0..
SET VENV_DIR=%REPO_ROOT%\.venv

echo [INFO] Repository root: %REPO_ROOT%

REM ── Check Python ─────────────────────────────────────────────────────────────
python --version >NUL 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found in PATH.
    echo         Download from https://www.python.org/downloads/
    pause & exit /b 1
)
FOR /F "tokens=2" %%v IN ('python --version') DO SET PY_VER=%%v
echo [INFO] Found Python %PY_VER%

REM ── Check ffmpeg ──────────────────────────────────────────────────────────────
ffmpeg -version >NUL 2>&1
IF ERRORLEVEL 1 (
    echo [WARN] ffmpeg not found in PATH.
    echo        Download from https://ffmpeg.org/download.html and add bin\ to PATH.
    echo        Continuing setup — app will fail at runtime without ffmpeg.
)

REM ── Create virtual environment ────────────────────────────────────────────────
IF NOT EXIST "%VENV_DIR%" (
    echo [INFO] Creating virtual environment at %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
) ELSE (
    echo [INFO] Virtual environment already exists.
)

CALL "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip wheel setuptools --quiet

REM ── PyTorch (GPU or CPU) ──────────────────────────────────────────────────────
nvidia-smi >NUL 2>&1
IF ERRORLEVEL 1 (
    echo [INFO] No NVIDIA GPU detected. Installing CPU-only PyTorch.
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
) ELSE (
    echo [INFO] NVIDIA GPU detected. Installing CUDA 12.1 PyTorch.
    echo        If your CUDA version is different, edit this script and change cu121.
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
)

REM ── Project dependencies ──────────────────────────────────────────────────────
echo [INFO] Installing project dependencies...
pip install -r "%REPO_ROOT%\requirements.txt" --quiet

REM ── .env file ────────────────────────────────────────────────────────────────
IF NOT EXIST "%REPO_ROOT%\.env" (
    COPY "%REPO_ROOT%\.env.example" "%REPO_ROOT%\.env" >NUL
    echo [WARN] .env created from .env.example
    echo        Edit it and set HF_TOKEN=hf_your_token_here before launching.
) ELSE (
    echo [INFO] .env already exists.
)

REM ── Data directories ──────────────────────────────────────────────────────────
IF NOT EXIST "%REPO_ROOT%\data\output"        MKDIR "%REPO_ROOT%\data\output"
IF NOT EXIST "%REPO_ROOT%\data\voice_samples" MKDIR "%REPO_ROOT%\data\voice_samples"

echo.
echo [INFO] Setup complete!
echo.
echo   Next steps:
echo   1. Edit .env and set HF_TOKEN=hf_your_token_here
echo   2. Activate venv:   .venv\Scripts\activate
echo   3. Launch app:      python run.py
echo.
pause
