"""Verifier for E2-1 (swarm2): static check of the LightGBM build in cs2gpu_alt.

Read-only. Searches lib_lightgbm.dll for the build-time messages and linked
runtime names that decide which LightGBM device paths were compiled in:
  - CUDA path: message "CUDA Tree Learner was not enabled" and CUDA runtime imports
  - OpenCL path: message "GPU Tree Learner was not enabled", OpenCL.dll import, boost_compute strings
Also reads the Khronos OpenCL vendor registry key (read-only) and the conda-meta records.

Usage (cs2gpu_alt interpreter only, never the PATH python):
  C:\\Users\\13144\\.pixi\\workspace-envs\\cs2gpu_alt-16300978804608365045\\envs\\default\\python.exe -I -X utf8 verify_E2_1_static.py
Writes nothing to disk; the caller redirects stdout to logs\\swarm2\\verify_E2_1.log.
"""
import glob
import hashlib
import json
import os
import sys
import winreg

ENV = r"C:\Users\13144\.pixi\workspace-envs\cs2gpu_alt-16300978804608365045\envs\default"
DLL = os.path.join(ENV, "Library", "bin", "lib_lightgbm.dll")
OPENCL_DLL = os.path.join(ENV, "Library", "bin", "OpenCL.dll")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_both(data, text):
    ascii_hits = data.count(text.encode("ascii"))
    utf16_hits = data.count(text.encode("utf-16-le"))
    return ascii_hits, utf16_hits


print("VERIFY_E2_1 static check")
print("python:", sys.version.split()[0], sys.executable)
print("lib_lightgbm.dll exists:", os.path.exists(DLL), DLL)
if not os.path.exists(DLL):
    raise SystemExit("missing lib_lightgbm.dll")
print("lib_lightgbm.dll bytes:", os.path.getsize(DLL))
print("lib_lightgbm.dll sha256:", sha256(DLL))
print("OpenCL.dll in Library/bin exists:", os.path.exists(OPENCL_DLL), "bytes:",
      os.path.getsize(OPENCL_DLL) if os.path.exists(OPENCL_DLL) else None)
print("OpenCL.dll in Library/bin sha256:", sha256(OPENCL_DLL) if os.path.exists(OPENCL_DLL) else None)

with open(DLL, "rb") as fh:
    data = fh.read()

probes = [
    "CUDA Tree Learner was not enabled in this build",
    "GPU Tree Learner was not enabled in this build",
    "Please recompile with CMake option -DUSE_CUDA=1",
    "Please recompile with CMake option -DUSE_GPU=1",
    "USE_GPU",
    "OpenCL.dll",
    "boost_compute",
    "boost_filesystem.dll",
    "cudart",
    "nvcuda.dll",
    "cublas",
    "Using sparse features with CUDA is currently not supported",
]
print("\nSTRING PROBES (ascii hits / utf-16-le hits in lib_lightgbm.dll)")
for text in probes:
    a, u = count_both(data, text)
    print(f"  {text!r}: ascii={a} utf16={u}")

print("\nCONDA RECORDS")
for pattern in ["liblightgbm-*.json", "lightgbm-*.json", "khronos-opencl-icd-loader-*.json"]:
    for path in glob.glob(os.path.join(ENV, "conda-meta", pattern)):
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        print(f"  {os.path.basename(path)}: name={rec.get('name')} version={rec.get('version')} "
              f"build={rec.get('build')} depends={rec.get('depends')}")

print("\nKHRONOS OPENCL VENDOR REGISTRY (read-only)")
for hive_name, hive in [("HKLM", winreg.HKEY_LOCAL_MACHINE)]:
    for sub in [r"SOFTWARE\Khronos\OpenCL\Vendors", r"SOFTWARE\WOW6432Node\Khronos\OpenCL\Vendors"]:
        try:
            key = winreg.OpenKey(hive, sub, 0, winreg.KEY_READ)
        except OSError as exc:
            print(f"  {hive_name}\\{sub}: ABSENT ({exc.__class__.__name__}: {exc})")
            continue
        vals = []
        i = 0
        while True:
            try:
                name, value, _ = winreg.EnumValue(key, i)
                vals.append((name, value))
                i += 1
            except OSError:
                break
        winreg.CloseKey(key)
        print(f"  {hive_name}\\{sub}: PRESENT values={vals}")

print("\nSTATIC CHECK DONE")
