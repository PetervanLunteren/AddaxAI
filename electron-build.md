# Electron Build Guide

This guide explains how to build the AddaxAI desktop application on different platforms.

## Overview

The desktop app uses:
- **Electron** for the desktop shell
- **PyInstaller** to bundle the Python backend into a single executable
- **electron-builder** to create platform-specific installers

Each platform builds its own PyInstaller executable, resulting in ~100-120MB installers.

## Prerequisites

All platforms need:
- Node.js 18+ and npm
- Git

Platform-specific:
- **macOS**: Xcode Command Line Tools (`xcode-select --install`)
- **Windows**: Python 3.11+ installed
- **Linux**: Python 3.11+ installed

## Building on macOS

### 1. Install Dependencies

```bash
# Install backend dependencies
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pyinstaller

# Install Electron dependencies
cd ../electron
npm install
```

### 2. Build Backend Executable

```bash
cd ../backend
source venv/bin/activate
pyinstaller backend.spec --clean
```

This creates `backend/dist/backend` (26MB executable)

### 3. Build Electron App

```bash
cd ../electron
npm run build    # Compile TypeScript
npm run package  # Build DMG and ZIP
```

**Output:**
- `electron/dist-build/AddaxAI-0.1.0-arm64.dmg` (~116MB)
- `electron/dist-build/AddaxAI-0.1.0-arm64-mac.zip` (~113MB)

### 4. Test the Installer

```bash
# Open the DMG
open dist-build/AddaxAI-0.1.0-arm64.dmg

# Or run the packaged app directly
open dist-build/mac-arm64/AddaxAI.app
```

## Building on Windows

### 1. Install Dependencies

```powershell
# Install backend dependencies
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller

# Install Electron dependencies
cd ..\electron
npm install
```

### 2. Build Backend Executable

```powershell
cd ..\backend
.\venv\Scripts\activate
pyinstaller backend.spec --clean
```

This creates `backend\dist\backend.exe` (~26MB executable)

### 3. Build Electron App

```powershell
cd ..\electron
npm run build    # Compile TypeScript
npm run package  # Build NSIS installer and ZIP
```

**Output:**
- `electron\dist-build\AddaxAI Setup 0.1.0.exe` (~120MB installer)
- `electron\dist-build\AddaxAI-0.1.0-win.zip` (~115MB)

### 4. Test the Installer

```powershell
# Run the installer
.\dist-build\AddaxAI Setup 0.1.0.exe

# Or run the packaged app directly
.\dist-build\win-unpacked\AddaxAI.exe
```

**Note:** On Windows, you may see a SmartScreen warning since the app isn't code-signed. Click "More info" → "Run anyway" for testing.

## Building on Linux

### 1. Install Dependencies

```bash
# Install backend dependencies
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pyinstaller

# Install Electron dependencies (requires some system packages)
cd ../electron
npm install
```

**Ubuntu/Debian additional packages:**
```bash
sudo apt-get install -y rpm fakeroot dpkg
```

### 2. Build Backend Executable

```bash
cd ../backend
source venv/bin/activate
pyinstaller backend.spec --clean
```

This creates `backend/dist/backend` (~26MB executable)

### 3. Build Electron App

```bash
cd ../electron
npm run build    # Compile TypeScript
npm run package  # Build AppImage and DEB
```

**Output:**
- `electron/dist-build/AddaxAI-0.1.0.AppImage` (~120MB)
- `electron/dist-build/addaxai_0.1.0_amd64.deb` (~115MB)

### 4. Test the Installer

```bash
# Make AppImage executable and run
chmod +x dist-build/AddaxAI-0.1.0.AppImage
./dist-build/AddaxAI-0.1.0.AppImage

# Or install the DEB package
sudo dpkg -i dist-build/addaxai_0.1.0_amd64.deb
addaxai
```

## Development Mode

For development (all platforms), you can run without building:

```bash
# Terminal 1 - Start backend
cd backend
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
python -m uvicorn app.main:app --reload

# Terminal 2 - Start Electron
cd electron
npm run dev
```

This starts Electron with the development backend, including hot reload.

## Troubleshooting

### Backend Executable Fails to Start

Check the logs at `~/AddaxAI/logs/backend.log` (or `%USERPROFILE%\AddaxAI\logs\backend.log` on Windows)

### PyInstaller Missing Modules

If you see "ModuleNotFoundError" when running the executable:
1. Add the module to `hiddenimports` in `backend/backend.spec`
2. Rebuild: `pyinstaller backend.spec --clean`

### Electron Build Fails

- **macOS**: Make sure Xcode Command Line Tools are installed
- **Windows**: Run as Administrator if permission errors occur
- **Linux**: Install `rpm` and `fakeroot` packages

### Large Installer Size

The installer includes:
- Electron framework (~90MB)
- PyInstaller backend (~26MB)
- Node modules and dependencies

This is normal for self-contained desktop apps. The alternative (asking users to install Python) is less user-friendly.

## Clean Build

To start fresh:

```bash
# Clean backend build artifacts
cd backend
rm -rf build/ dist/ __pycache__/
find . -name "*.pyc" -delete

# Clean Electron build artifacts
cd ../electron
rm -rf dist/ dist-build/ node_modules/
npm install
```

## Code Signing (Optional)

For production distribution:

### macOS
1. Get an Apple Developer account
2. Create a Developer ID Application certificate
3. Add to `electron/package.json`:
```json
"mac": {
  "identity": "Developer ID Application: Your Name (TEAM_ID)"
}
```

### Windows
1. Get a code signing certificate
2. Use `electron-builder` with certificate:
```json
"win": {
  "certificateFile": "path/to/cert.pfx",
  "certificatePassword": "password"
}
```

## CI/CD with GitHub Actions

See `.github/workflows/build.yml` (TODO) for automated multi-platform builds.
