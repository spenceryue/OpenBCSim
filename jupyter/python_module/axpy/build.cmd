python setup.py build -c msvc > build.txt 2>&1

@echo off
if errorlevel 1 (
  <nul set /p =Build failed. Tests not run. See logs at build.txt.
  exit /b %errorlevel%
) else (
  <nul set /p =Build succeeded.
)
@echo on

python setup.py test > test.txt 2>&1

@echo off
if errorlevel 1 (
  <nul set /p =Test failed. See logs at test.txt.
  exit /b %errorlevel%
) else (
  <nul set /p =Tests passed.
)
@echo on