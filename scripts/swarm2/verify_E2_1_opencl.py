"""Verifier for E2-1 (swarm2): OpenCL platform and device probe, read-only.

Question: does the OpenCL path that LightGBM device_type=gpu uses see any platform or device
in this process, and where does the in-process OpenCL.dll come from?

Steps:
  1. import lightgbm (loads lib_lightgbm.dll, which imports OpenCL.dll) and report which
     OpenCL.dll the process actually mapped (GetModuleHandle + GetModuleFileName).
  2. Call clGetPlatformIDs / clGetDeviceIDs / clGetDeviceInfo through that loaded module.
  3. Load the NVIDIA ICD nvopencl64.dll from the driver store directly and repeat (bypasses
     the Khronos vendor registry), to separate "loader found no ICD" from "driver fails".

Usage (cs2gpu_alt interpreter only):
  C:\\Users\\13144\\.pixi\\workspace-envs\\cs2gpu_alt-16300978804608365045\\envs\\default\\python.exe -I -X utf8 verify_E2_1_opencl.py
Writes nothing to disk; the caller redirects stdout to logs\\swarm2\\verify_E2_1.log.
"""
import ctypes
import glob
import os
import sys
from ctypes import wintypes

print("VERIFY_E2_1 OpenCL probe")
print("python:", sys.executable)

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetModuleHandleW.restype = wintypes.HMODULE
k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
k32.GetModuleFileNameW.restype = wintypes.DWORD
k32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]


def module_path(name):
    h = k32.GetModuleHandleW(name)
    if not h:
        return None
    buf = ctypes.create_unicode_buffer(1024)
    k32.GetModuleFileNameW(h, buf, 1024)
    return buf.value


# Step 1: importing lightgbm loads lib_lightgbm.dll and its dependency OpenCL.dll.
import lightgbm  # noqa: E402

print("lightgbm", lightgbm.__version__, "imported from", lightgbm.__file__)
print("OpenCL.dll mapped in process after import lightgbm:", module_path("OpenCL.dll"))
print("lib_lightgbm.dll mapped:", module_path("lib_lightgbm.dll"))
print("System32 OpenCL.dll exists:", os.path.exists(r"C:\Windows\System32\OpenCL.dll"))
path_env = os.environ.get("PATH", "")
print("PATH contains env Library\\bin:",
      any(p.lower().rstrip("\\") == r"c:\users\13144\.pixi\workspace-envs\cs2gpu_alt-16300978804608365045\envs\default\library\bin"
          for p in path_env.split(";")))

CL_SUCCESS = 0
CL_PLATFORM_VERSION = 0x0901
CL_PLATFORM_NAME = 0x0902
CL_PLATFORM_VENDOR = 0x0903
CL_DEVICE_TYPE = 0x1000
CL_DEVICE_VENDOR = 0x102C
CL_DEVICE_NAME = 0x102B
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_VERSION = 0x102F
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF


def bind(lib):
    lib.clGetPlatformIDs.restype = ctypes.c_int32
    lib.clGetPlatformIDs.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    lib.clGetPlatformInfo.restype = ctypes.c_int32
    lib.clGetPlatformInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t,
                                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.clGetDeviceIDs.restype = ctypes.c_int32
    lib.clGetDeviceIDs.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32,
                                   ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    lib.clGetDeviceInfo.restype = ctypes.c_int32
    lib.clGetDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t,
                                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    return lib


def info_str(lib, getter, handle, param):
    size = ctypes.c_size_t(0)
    rc = getter(handle, param, 0, None, ctypes.byref(size))
    if rc != CL_SUCCESS:
        return f"<rc={rc}>"
    buf = ctypes.create_string_buffer(size.value)
    rc = getter(handle, param, size.value, buf, None)
    if rc != CL_SUCCESS:
        return f"<rc={rc}>"
    raw = buf.raw[:size.value]
    try:
        text = raw.rstrip(b"\x00").decode("ascii")
        return f"{text!r} (ascii ok, {len(raw)} bytes)"
    except UnicodeDecodeError:
        return f"<non-ascii bytes {raw[:60]!r}, {len(raw)} bytes>"


def enumerate_platforms(label, lib):
    lib = bind(lib)
    n = ctypes.c_uint32(0)
    rc = lib.clGetPlatformIDs(0, None, ctypes.byref(n))
    print(f"[{label}] clGetPlatformIDs(count) rc={rc} num_platforms={n.value}")
    if rc != CL_SUCCESS or n.value == 0:
        return 0
    plats = (ctypes.c_void_p * n.value)()
    rc = lib.clGetPlatformIDs(n.value, plats, None)
    print(f"[{label}] clGetPlatformIDs(list) rc={rc}")
    total_devices = 0
    for idx, p in enumerate(plats):
        print(f"[{label}]   platform {idx}: name={info_str(lib, lib.clGetPlatformInfo, p, CL_PLATFORM_NAME)} "
              f"vendor={info_str(lib, lib.clGetPlatformInfo, p, CL_PLATFORM_VENDOR)} "
              f"version={info_str(lib, lib.clGetPlatformInfo, p, CL_PLATFORM_VERSION)}")
        nd = ctypes.c_uint32(0)
        rc = lib.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, None, ctypes.byref(nd))
        print(f"[{label}]   platform {idx}: clGetDeviceIDs(count) rc={rc} num_devices={nd.value}")
        if rc != CL_SUCCESS or nd.value == 0:
            continue
        devs = (ctypes.c_void_p * nd.value)()
        rc = lib.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, nd.value, devs, None)
        for jdx, d in enumerate(devs):
            dtype = ctypes.c_uint64(0)
            lib.clGetDeviceInfo(d, CL_DEVICE_TYPE, 8, ctypes.byref(dtype), None)
            print(f"[{label}]     device {jdx}: type_bits={dtype.value:#x} "
                  f"name={info_str(lib, lib.clGetDeviceInfo, d, CL_DEVICE_NAME)} "
                  f"vendor={info_str(lib, lib.clGetDeviceInfo, d, CL_DEVICE_VENDOR)} "
                  f"version={info_str(lib, lib.clGetDeviceInfo, d, CL_DEVICE_VERSION)} "
                  f"driver={info_str(lib, lib.clGetDeviceInfo, d, CL_DRIVER_VERSION)}")
        total_devices += nd.value
    return total_devices


# Step 2: the OpenCL.dll that lightgbm actually mapped.
mapped = module_path("OpenCL.dll")
if mapped:
    print("\nSTEP 2: enumerate through the mapped OpenCL.dll:", mapped)
    try:
        lib = ctypes.WinDLL(mapped)
        ndev = enumerate_platforms("mapped", lib)
        print(f"[mapped] total devices visible: {ndev}")
    except OSError as exc:
        print("[mapped] load failed:", repr(exc))
else:
    print("\nSTEP 2: OpenCL.dll not mapped after import lightgbm; skipped")

# Step 3: NVIDIA ICD loaded directly. An ICD exports clIcdGetPlatformIDsKHR, not clGetPlatformIDs,
# so this only counts platforms the driver reports on its own (bypasses the Khronos loader and registry).
icds = glob.glob(r"C:\Windows\System32\DriverStore\FileRepository\nv_dispig.inf_amd64_*\nvopencl64.dll")
print("\nSTEP 3: NVIDIA ICD candidates:", icds)
for icd in icds:
    try:
        nv = ctypes.WinDLL(icd)
        print(f"[nv-direct] loaded {icd}")
        f = nv.clIcdGetPlatformIDsKHR
        f.restype = ctypes.c_int32
        f.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        n = ctypes.c_uint32(0)
        rc = f(0, None, ctypes.byref(n))
        print(f"[nv-direct] clIcdGetPlatformIDsKHR(count) rc={rc} num_platforms={n.value}")
    except (OSError, AttributeError) as exc:
        print(f"[nv-direct] failed for {icd}: {exc!r}")

print("\nOPENCL PROBE DONE")
