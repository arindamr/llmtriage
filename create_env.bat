:: filepath: c:\Users\arind\OneDrive\Development\LLMTriage\create_env.bat
@echo off
echo Creating Conda environment...
conda create --name llmtriage python=3.9 -y

echo Activating Conda environment...
conda activate llmtriage

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Environment setup complete. To activate the environment, run:
echo conda activate llmtriage