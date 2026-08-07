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
    # Scripted setup for IT deployments: `backend --setup [MODEL_ID ...]`
    # or `backend --list-models`. Runs the first-launch setup from the
    # command line and exits, no server. See app/setup_cli.py.
    if "--setup" in sys.argv or "--list-models" in sys.argv:
        from app.setup_cli import run_cli

        sys.exit(run_cli(sys.argv[1:]))

    import uvicorn

    from app.core.config import get_settings

    # Start uvicorn server
    # Configuration is handled by app/core/config.py with sensible defaults.
    #
    # The port MUST come from settings rather than a literal. Electron picks
    # the port and passes it down as API_PORT (see spawnBackend in
    # electron/src/main.ts), then waits for /health on that same port. A
    # hardcoded 8000 here meant the packaged backend ignored the setting, so
    # the app could not be moved off a port another application already held.
    settings = get_settings()

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=settings.api_port,
        log_level="info",
        reload=False,
    )
