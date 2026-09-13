# Build from the source directory using PyInstaller, preserving SDK files alongside the executable.
a = Analysis(['main.py'], pathex=[], binaries=[], datas=[], hiddenimports=[],
             hookspath=[], runtime_hooks=[], excludes=[], noarchive=False)
# Avoid bundling unrelated ICU libraries discovered through the builder's PATH.
a.binaries = [x for x in a.binaries if x[0].lower() not in ('icuuc.dll', 'icudt78.dll')]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='StarfieldStudio-2.4',
          debug=False, strip=False, upx=True, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name='StarfieldStudio')
