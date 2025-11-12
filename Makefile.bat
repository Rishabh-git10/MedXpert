@echo off
REM MedXpert - Common Commands

if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="test" goto test
if "%1"=="clean" goto clean
if "%1"=="run-api" goto run-api
if "%1"=="run-app" goto run-app
if "%1"=="docker-up" goto docker-up
if "%1"=="docker-down" goto docker-down
goto help

:help
echo Available commands:
echo   Makefile.bat install      - Install dependencies
echo   Makefile.bat test         - Run tests
echo   Makefile.bat clean        - Clean cache files
echo   Makefile.bat run-api      - Run API locally
echo   Makefile.bat run-app      - Run Streamlit locally
echo   Makefile.bat docker-up    - Start Docker containers
echo   Makefile.bat docker-down  - Stop Docker containers
goto end

:install
pip install -r requirements.txt
goto end

:test
cd tests && pytest
goto end

:clean
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
del /s /q *.pyc 2>nul
for /d /r . %d in (.pytest_cache) do @if exist "%d" rd /s /q "%d"
if exist htmlcov rd /s /q htmlcov
goto end

:run-api
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
goto end

:run-app
streamlit run src/app.py
goto end

:docker-up
docker-compose up -d
goto end

:docker-down
docker-compose down
goto end

:end
