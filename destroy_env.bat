:: filepath: c:\Users\arind\OneDrive\Development\LLMTriage\destroy_env.bat
@echo off
echo Deactivating Conda environment (if active)...
conda deactivate

echo Removing Conda environment...
conda remove --name llmtriage --all -y

echo Environment destroyed.