# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AddaxAI backend.

This builds a single-file executable that includes:
- FastAPI application
- All Python dependencies
- Database migrations (alembic)
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Get the backend directory
backend_dir = Path.cwd()

block_cipher = None

# Collect all data from key packages
datas = [
    # Include the app package
    ('app', 'app'),
    # Include alembic for database migrations
    ('alembic', 'alembic'),
    ('alembic.ini', '.'),
    # Include frontend static files
    ('../frontend/dist', 'frontend/dist'),
]

# Collect data files from packages
datas += collect_data_files('huggingface_hub')
datas += collect_data_files('pydantic')
datas += collect_data_files('fastapi')

# Comprehensive hidden imports - collect ALL submodules
hiddenimports = []
hiddenimports += collect_submodules('fastapi')
hiddenimports += collect_submodules('starlette')
hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('pydantic')
hiddenimports += collect_submodules('pydantic_core')
hiddenimports += collect_submodules('pydantic_settings')
hiddenimports += collect_submodules('sqlalchemy')
hiddenimports += collect_submodules('alembic')
hiddenimports += collect_submodules('huggingface_hub')
hiddenimports += collect_submodules('PIL')
hiddenimports += collect_submodules('multipart')
hiddenimports += collect_submodules('websockets')
hiddenimports += collect_submodules('httpx')
hiddenimports += collect_submodules('redis')
hiddenimports += collect_submodules('requests')  # Required by huggingface_hub
hiddenimports += ['yaml', 'yaml.loader', 'yaml.dumper']

a = Analysis(
    ['run_server.py'],
    pathex=[str(backend_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest',
        'mypy',
        'ruff',
        'tkinter',
        'matplotlib',
        'numpy',  # Not used, reduces size
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    # Use --collect-all for problematic packages
    module_collection_mode={
        'fastapi': 'py',
        'starlette': 'py',
        'uvicorn': 'py',
        'pydantic': 'py',
        'sqlalchemy': 'py',
        'huggingface_hub': 'py',
    }
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],  # Remove binaries, zipfiles, datas - will be in COLLECT
    exclude_binaries=True,  # Important: don't bundle everything into one file
    name='backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX - can cause issues with code signing
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # These options help with macOS code signing
    bundle_identifier='com.addaxai.cameratrap.backend',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='backend',
)
