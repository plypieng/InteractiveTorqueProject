# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Interactive Torque Application
Creates a standalone Windows desktop application
"""

block_cipher = None

# Collect all data files
from PyInstaller.utils.hooks import collect_data_files
imblearn_datas = collect_data_files('imblearn')

datas = [
    ('app', 'app'),  # Include the entire app package
    ('trained_models', 'trained_models'),  # Include trained ML models
    ('assets', 'assets'),  # Include CSS and other assets
    ('data', 'data'),  # Include data directory structure
    ('.env.sample', '.'),  # Include sample environment file
] + imblearn_datas  # Include imblearn package data (VERSION.txt, etc.)

# Hidden imports for Dash and Plotly
hiddenimports = [
    'dash',
    'dash_bootstrap_components',
    'dash.dependencies',
    'dash_auth',
    'plotly',
    'plotly.graph_objs',
    'flask',
    'sqlalchemy',
    'sqlalchemy.ext.declarative',
    'alembic',
    'pandas',
    'numpy',
    'scipy',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.tree',
    'sklearn.linear_model',
    'imblearn',
    'joblib',
    'webview',
    'run_migrations',
    'logging.config',  # Required by alembic env.py
]

a = Analysis(
    ['standalone_launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['celery', 'redis'],  # Explicitly exclude removed dependencies
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='InteractiveTorqueApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Enable console for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='InteractiveTorqueApp',
)
