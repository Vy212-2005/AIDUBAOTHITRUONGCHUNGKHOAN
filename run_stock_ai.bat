@echo off
echo Starting Stock Market Prediction AI...
set CONDA_ENV_PATH=C:\Users\LOQ\anaconda3\envs\stock_ai
set STREAMLIT_EXE=%CONDA_ENV_PATH%\Scripts\streamlit.exe
set APP_PATH=c:\Users\LOQ\Documents\AI DỰ BÁO THỊ TRƯỜNG CHỨNG KHOÁN\app_chat.py

if exist "%STREAMLIT_EXE%" (
    "%STREAMLIT_EXE%" run "%APP_PATH%"
) else (
    echo Error: Could not find Streamlit at %STREAMLIT_EXE%
    echo Please make sure the 'stock_ai' environment is installed.
    pause
)
