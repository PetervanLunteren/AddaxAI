# This is a placeholder script for PyInstaller to create the main executable 
# https://addaxdatascience.com/addaxai/
# Created by Peter van Lunteren
# Latest edit by Peter van Lunteren on 12 Jan 2024

import os
import subprocess
import sys
import platform

print("\n")

system = platform.system()

# clean path
if getattr(sys, 'frozen', False):
    AddaxAI_files = os.path.dirname(sys.executable)
else:
    AddaxAI_files = os.path.dirname(os.path.abspath(__file__))

if AddaxAI_files.endswith("main.app/Contents/MacOS"):
    AddaxAI_files = AddaxAI_files.replace("main.app/Contents/MacOS", "")

if AddaxAI_files.endswith(".app/Contents/MacOS"):
    AddaxAI_files = os.path.dirname(os.path.dirname(os.path.dirname(AddaxAI_files)))

# init paths
GUI_script = os.path.join(AddaxAI_files, "AddaxAI", "AddaxAI_GUI.py")
first_startup_file = os.path.join(AddaxAI_files, "first-startup.txt")

# log
print(f"        AddaxAI_files: {AddaxAI_files}")
print(f"       sys.executable: {sys.executable.replace(AddaxAI_files, '.')}")
print(f"           GUI_script: {GUI_script.replace(AddaxAI_files, '.')}")
print(f"    python_executable: using current interpreter")

#cuda toolkit
cuda_toolkit_path = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
print(f"    cuda_toolkit_path: {cuda_toolkit_path}")

# run the GUI script using CURRENT Python environment
print("\nOpening application...")

run_cmd = [sys.executable, GUI_script]

if system == 'Windows':
    if sys.executable.endswith("debug.exe"):
        subprocess.run(run_cmd)
        input("Press [Enter] to close console window...")
    else:
        subprocess.run(run_cmd, creationflags=subprocess.CREATE_NO_WINDOW)
else:
    subprocess.run(run_cmd)