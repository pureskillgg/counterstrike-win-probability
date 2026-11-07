"""Verifier for E2-1 (swarm2): which cache-dir API the LightGBM DLL uses (read-only string search).

Usage: cs2gpu_alt python.exe -I -X utf8 verify_E2_1_dllstr2.py
"""
import os
import sys

DLL = r"C:\Users\<user>\.pixi\workspace-envs\cs2gpu_alt-16300978804608365045\envs\default\Library\bin\lib_lightgbm.dll"
with open(DLL, "rb") as fh:
    data = fh.read()
print("VERIFY_E2_1 DLL cache-dir string probe", sys.version.split()[0])
for text in ["APPDATA", "SHGetFolderPath", "SHGetKnownFolderPath", "shell32.dll", "getenv",
             "boost_compute", "user_cache_dir", "XDG_CACHE_HOME", "LOCALAPPDATA"]:
    print(f"  {text!r}: ascii={data.count(text.encode('ascii'))} utf16={data.count(text.encode('utf-16-le'))}")
print("DONE")
