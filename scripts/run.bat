@echo off
setlocal

REM --- move to project root (rag_app) ---
cd /d %~dp0\..

REM --- activate venv ---
if not exist ".venv\Scripts\activate.bat" (
  echo venv not found. Run scripts\setup.bat first.
  pause
  exit /b 1
)
call .\.venv\Scripts\activate.bat

REM --- start server ---
echo Starting API: http://127.0.0.1:8000/docs
python -m uvicorn api:app --reload

endlocal
