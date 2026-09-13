# Build from the source directory using PyInstaller, preserving SDK files alongside the executable.
from pathlib import Path
icon_path = str(Path(SPECPATH) / 'starfield.ico')
a = Analysis(['main.py'], pathex=[], binaries=[], datas=[(icon_path, '.')], hiddenimports=[],
             hookspath=[], runtime_hooks=[], excludes=[], noarchive=False)
# Avoid bundling unrelated ICU libraries discovered through the builder's PATH.
a.binaries = [x for x in a.binaries if x[0].lower() not in ('icuuc.dll', 'icudt78.dll')]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='StarfieldStudio-2.6',
          debug=False, strip=False, upx=True, console=False, icon=icon_path)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name='StarfieldStudio')
