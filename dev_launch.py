# Dev launcher: runs AddaxAI_GUI.py with the correct AddaxAI_files path
# pointing at the installed app, so all models/envs/deps resolve correctly.
# Usage: C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe dev_launch.py

import os
import sys

ADDAXAI_FILES = r"C:\Users\Topam\AddaxAI_files"

# Inject paths that the installed GUI sets up via sys.path.insert
paths_to_add = [
    ADDAXAI_FILES,
    os.path.join(ADDAXAI_FILES, "cameratraps"),
    os.path.join(ADDAXAI_FILES, "cameratraps", "megadetector"),
]
for p in paths_to_add:
    if p not in sys.path:
        sys.path.insert(0, p)

# Point __file__ resolution to the repo's GUI, but override AddaxAI_files
# by monkey-patching os.path so dirname(dirname(realpath(__file__))) = ADDAXAI_FILES
# Simpler: just exec the file with a patched global.
gui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AddaxAI_GUI.py")

with open(gui_path, "r", encoding="utf-8") as f:
    source = f.read()

# Override the AddaxAI_files line before executing
source = source.replace(
    "AddaxAI_files = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))",
    f"AddaxAI_files = r'{ADDAXAI_FILES}'"
)

exec(compile(source, gui_path, "exec"), {"__file__": gui_path, "__name__": "__main__"})
