@echo off
REM setup.bat - Windows setup script for InteractiveTorqueProject

REM Create and activate virtual environment
python -m venv venv
call venv\Scripts\activate

REM Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

REM Copy env sample if no .env exists
if not exist .env (
    copy .env.sample .env
    echo .env file created from .env.sample. Please edit it with your paths and credentials.
) else (
    echo .env already exists. Skipping copy.
)

echo Setup complete. To activate virtual environment:
    PowerShell: .\venv\Scripts\Activate.ps1
    CMD: call venv\Scripts\activate
