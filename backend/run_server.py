#!/usr/bin/env python3
"""
Entry point for PyInstaller-bundled backend server.

This script starts the uvicorn server with the FastAPI app.
The app configuration now has sensible defaults, so no
environment setup is needed.
"""

import sys
from pathlib import Path

# When running as PyInstaller bundle, add bundle directory to Python path
if getattr(sys, 'frozen', False):
    bundle_dir = Path(sys._MEIPASS)
    sys.path.insert(0, str(bundle_dir))

if __name__ == "__main__":
    import uvicorn

    # Start uvicorn server
    # Configuration is handled by app/core/config.py with sensible defaults
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,
    )
