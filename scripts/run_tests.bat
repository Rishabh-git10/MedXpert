@echo off
echo Running MedXpert Test Suite...
cd tests
pytest -v --cov=../src --cov-report=html --cov-report=term
cd ..
echo.
echo Test complete htmlcov\index.html
