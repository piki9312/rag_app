@echo off
echo [SETUP] create venv
python -m venv .venv

echo [SETUP] activate venv
call .\.venv\Scripts\activate

echo [SETUP] upgrade pip
python -m pip install --upgrade pip

echo [SETUP] install requirements
pip install -r requirements.txt

echo [SETUP] done
pause
