@echo off
setlocal

REM --- move to project root (rag_app) ---
cd /d %~dp0\..


REM --- create venv if not exists ---
if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating venv...
  python -m venv .venv
) else (
  echo [1/4] venv already exists.
)

REM --- activate venv ---
echo [2/4] Activating venv...
call .\.venv\Scripts\activate.bat

REM --- upgrade pip ---
echo [3/4] Upgrading pip...
python -m pip install -U pip

REM --- install deps ---
echo [4/4] Installing requirements...
if exist "requirements.txt" (
  python -m pip install -r requirements.txt
) else (
  echo requirements.txt not found. Installing minimal deps...
  python -m pip install fastapi "uvicorn[standard]" openai python-dotenv numpy sentence-transformers faiss-cpu pypdf python-docx
)

echo.
echo Setup complete.
echo Next: scripts\run.bat
pause
endlocal
