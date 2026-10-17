# PyInstaller spec — localist-mcp (port 8003), base + `[ocr]` build.
# See localist-backend.spec for the shared reasoning (onedir, no MLX,
# uvicorn hook auto-discovery, why mcp's submodules aren't force-collected).
#
# Unlike localist-backend, this bundles the `[ocr]` extra (Apple Vision via
# pyobjc, pypdfium2 + its PDFium binary) — ocr_extract runs here. Its
# original exclusion was for license footprint (PyMuPDF, AGPL); PyMuPDF
# was replaced by pypdfium2 (Apache-2.0/BSD-3) in §22.13 and pyobjc is MIT,
# so that reason no longer applies. Requires the packaging venv to have
# `[ocr]` installed — see backend/packaging/README.md.
#
# Build: pyinstaller localist-mcp.spec  (from backend/packaging/)
# Run:   ./dist/localist-mcp/localist-mcp

block_cipher = None

a = Analysis(
    ['run_mcp_server.py'],
    pathex=['../src'],
    binaries=[],
    datas=[],
    # pypdfium2's bitmap.to_pil() (ocr.py's scanned-PDF fallback) imports
    # PIL lazily in a way static analysis doesn't follow — without this the
    # frozen build fails scanned PDFs with "No module named 'PIL'".
    hiddenimports=['PIL.Image'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='localist-mcp',
    debug=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='localist-mcp',
)
