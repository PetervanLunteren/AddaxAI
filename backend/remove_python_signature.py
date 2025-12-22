#!/usr/bin/env python3
"""
Post-build script to remove signatures from the Python shared library
in the PyInstaller bundle. This prevents Team ID mismatch errors when
the Electron app is signed with a different Developer ID certificate.
"""
import os
import subprocess
import sys
from pathlib import Path

def remove_signatures(dist_dir):
    """Remove code signatures from Python libraries in the dist directory."""
    backend_dir = Path(dist_dir) / "backend" / "_internal"

    if not backend_dir.exists():
        print(f"Error: {backend_dir} does not exist")
        return 1

    # Find the Python shared library
    python_lib = None
    for item in backend_dir.rglob("Python"):
        # Look for the actual Python shared library (not symlinks)
        if item.is_file() and not item.is_symlink():
            python_lib = item
            break

    if python_lib:
        print(f"Removing signature from: {python_lib}")
        try:
            subprocess.run(
                ["codesign", "--remove-signature", str(python_lib)],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✓ Successfully removed signature from {python_lib}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not remove signature: {e.stderr}")
            # Don't fail the build, just warn
    else:
        print("Warning: Could not find Python shared library")

    # Also remove signatures from any .dylib or .so files
    for ext in ["*.dylib", "*.so"]:
        for lib in backend_dir.rglob(ext):
            if lib.is_file() and not lib.is_symlink():
                try:
                    # Check if it's signed
                    result = subprocess.run(
                        ["codesign", "--verify", str(lib)],
                        capture_output=True
                    )
                    if result.returncode == 0:  # It's signed
                        print(f"Removing signature from: {lib.name}")
                        subprocess.run(
                            ["codesign", "--remove-signature", str(lib)],
                            check=True,
                            capture_output=True
                        )
                except subprocess.CalledProcessError:
                    # Not signed or couldn't remove, continue
                    pass

    return 0

if __name__ == "__main__":
    dist_dir = sys.argv[1] if len(sys.argv) > 1 else "dist"
    sys.exit(remove_signatures(dist_dir))
