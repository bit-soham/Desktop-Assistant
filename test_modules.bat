@echo off
echo Testing Python modules...
echo.

REM Check syntax compilation
echo Checking syntax compilation...
python -m py_compile models/audio_processing.py models/text_processing.py models/note_management.py models/rag_llm.py models/model_setup.py models/utils.py
if %errorlevel% neq 0 (
    echo Syntax compilation failed!
    pause
    exit /b 1
)
echo Syntax compilation successful!
echo.

REM Test imports
echo Testing imports...
python -c "import models.audio_processing; import models.text_processing; import models.note_management; import models.rag_llm; import models.model_setup; import models.utils; print('All imports successful!')"
if %errorlevel% neq 0 (
    echo Import test failed!
    pause
    exit /b 1
)
echo Import test successful!
echo.
echo All tests passed! Ready to run main.py
pause