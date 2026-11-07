"""Independent static check for E2-1 (LightGBM device paths in cs2gpu_alt).

Read-only and stdlib-only. Writes no files; the caller redirects stdout.
Sections:
  1. PE import table of lib_lightgbm.dll (which runtimes the DLL links).
  2. Build-message strings in lib_lightgbm.dll (ASCII and UTF-16LE hit counts).
  3. Khronos OpenCL vendor registry (read-only), 64-bit and 32-bit views.
  4. OpenCL ICD loader queries (clGetPlatformIDs, clGetPlatformInfo, clGetDeviceIDs).
  5. Audit of the E2 logs cited by the claim (sizes, hashes, markers, driver exit codes).
"""
import ctypes
import datetime
import glob
import hashlib
import os
import struct
import winreg

ENV = r"C:\Users\13144\.pixi\workspace-envs\cs2gpu_alt-16300978804608365045\envs\default"
DLL = os.path.join(ENV, "Library", "bin", "lib_lightgbm.dll")
LOGDIR = r"G:\pureskill\csds-cs2\logs\swarm2"
SYS32 = r"C:\Windows\System32"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_mtime(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%H:%M:%S")


def section(title):
    print("")
    print("=== " + title + " ===")


def parse_pe_imports(data):
    """Return (normal, delay) dicts: dll name -> list of imported symbol names."""
    if data[:2] != b"MZ":
        raise ValueError("not an MZ file")
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if data[e_lfanew:e_lfanew + 4] != b"PE\x00\x00":
        raise ValueError("PE signature missing")
    coff = e_lfanew + 4
    _machine, nsec, _ts, _ptr, _nsyms, opt_size, _chars = struct.unpack_from("<HHIIIHH", data, coff)
    opt = coff + 20
    magic = struct.unpack_from("<H", data, opt)[0]
    if magic == 0x20B:
        dd = opt + 112
        thunk = 8
        ord_flag = 1 << 63
    elif magic == 0x10B:
        dd = opt + 96
        thunk = 4
        ord_flag = 1 << 31
    else:
        raise ValueError("unknown optional header magic 0x%x" % magic)
    sections = []
    sec0 = opt + opt_size
    for i in range(nsec):
        off = sec0 + 40 * i
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, off + 8)
        sections.append((vaddr, max(vsize, rawsize), rawptr))

    def r2o(rva):
        for vaddr, size, rawptr in sections:
            if vaddr <= rva < vaddr + size:
                return rawptr + (rva - vaddr)
        raise ValueError("RVA 0x%x outside all sections" % rva)

    def cstr(off):
        end = data.index(b"\x00", off)
        return data[off:end].decode("ascii", "replace")

    def read_thunks(int_rva, dll, target):
        toff = r2o(int_rva)
        fmt = "<Q" if thunk == 8 else "<I"
        while True:
            (val,) = struct.unpack_from(fmt, data, toff)
            if val == 0:
                break
            if val & ord_flag:
                target.setdefault(dll, []).append("ordinal#%d" % (val & 0xFFFF))
            else:
                target.setdefault(dll, []).append(cstr(r2o(val & (ord_flag - 1)) + 2))
            toff += thunk

    normal, delay = {}, {}
    irva, _isz = struct.unpack_from("<II", data, dd + 8 * 1)
    if irva:
        doff = r2o(irva)
        while True:
            oft, _ts, _fc, name_rva, ft = struct.unpack_from("<IIIII", data, doff)
            if oft == 0 and name_rva == 0 and ft == 0:
                break
            dll = cstr(r2o(name_rva))
            normal.setdefault(dll, [])
            read_thunks(oft or ft, dll, normal)
            doff += 20
    drva, _dsz = struct.unpack_from("<II", data, dd + 8 * 13)
    if drva:
        doff = r2o(drva)
        while True:
            attrs, name_rva, _mod, _iat, int_rva = struct.unpack_from("<IIIII", data, doff)
            if attrs == 0 and name_rva == 0 and int_rva == 0:
                break
            dll = cstr(r2o(name_rva))
            delay.setdefault(dll, [])
            read_thunks(int_rva, dll, delay)
            doff += 32
    return normal, delay


def part1_imports():
    section("1. PE import table of lib_lightgbm.dll")
    with open(DLL, "rb") as fh:
        data = fh.read()
    print("path:", DLL)
    print("bytes:", len(data), "sha256:", sha256_bytes(data))
    normal, delay = parse_pe_imports(data)
    print("normal import DLLs (%d):" % len(normal))
    for dll in sorted(normal):
        syms = normal[dll]
        print("  %-28s %4d symbols  e.g. %s" % (dll, len(syms), ", ".join(syms[:8])))
    print("delay-load import DLLs (%d):" % len(delay))
    for dll in sorted(delay):
        syms = delay[dll]
        print("  %-28s %4d symbols  e.g. %s" % (dll, len(syms), ", ".join(syms[:8])))
    flags = ["opencl", "cuda", "cublas", "cudnn", "nvcuda", "boost", "vcomp", "libomp", "gomp", "msvcp", "vcruntime"]
    hits = sorted({d for d in list(normal) + list(delay) if any(f in d.lower() for f in flags)})
    print("runtime-like DLL names linked:", hits)
    ocl = [s for s in normal.get("OpenCL.dll", []) + delay.get("OpenCL.dll", [])]
    print("OpenCL.dll symbols imported:", len(ocl), ocl[:12])
    cuda_like = sorted({s for d in list(normal) + list(delay) for s in (normal.get(d, []) + delay.get(d, []))
                        if s.lower().startswith("cu")})
    print("CUDA-style symbols imported (cu*):", len(cuda_like), cuda_like[:12])
    return data


def part2_strings(data):
    section("2. Build-message and runtime strings in lib_lightgbm.dll")
    probes = [
        "CUDA Tree Learner was not enabled in this build",
        "GPU Tree Learner was not enabled in this build",
        "Please recompile with CMake option -DUSE_CUDA=1",
        "Please recompile with CMake option -DUSE_GPU=1",
        "Using sparse features with CUDA is currently not supported",
        "boost_compute",
        "OpenCL",
        "clGetPlatformIDs",
        "nvcuda.dll",
        "cudart64",
    ]
    for text in probes:
        ascii_hits = data.count(text.encode("ascii"))
        utf16_hits = data.count(text.encode("utf-16-le"))
        print("  ascii=%d utf16le=%d  %s" % (ascii_hits, utf16_hits, text))


def part3_registry():
    section("3. Khronos OpenCL vendor registry (read-only)")

    def read_vendors(view_flag, label):
        vals = []
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Khronos\OpenCL\Vendors",
                                 0, winreg.KEY_READ | view_flag)
        except OSError as exc:
            print("  %s: key open failed: %s" % (label, exc))
            return
        i = 0
        try:
            while True:
                name, value, _typ = winreg.EnumValue(key, i)
                vals.append((name, value))
                i += 1
        except OSError:
            pass
        finally:
            winreg.CloseKey(key)
        print("  %s: %d value(s) %s" % (label, len(vals), vals))

    read_vendors(winreg.KEY_WOW64_64KEY, "HKLM SOFTWARE\\Khronos\\OpenCL\\Vendors (64-bit view)")
    read_vendors(winreg.KEY_WOW64_32KEY, "HKLM SOFTWARE\\Khronos\\OpenCL\\Vendors (32-bit view)")
    nv = glob.glob(r"C:\Windows\System32\DriverStore\FileRepository\*\nvopencl64.dll")
    print("  nvopencl64.dll in DriverStore:", nv)
    print("  nvopencl64.dll in System32:", os.path.exists(os.path.join(SYS32, "nvopencl64.dll")))
    print("  nvcuda.dll in System32:", os.path.exists(os.path.join(SYS32, "nvcuda.dll")))
    print("  OpenCL.dll (ICD loader) in System32:", os.path.exists(os.path.join(SYS32, "OpenCL.dll")))


def part4_icd():
    section("4. OpenCL ICD loader queries (read-only)")
    CL_PLATFORM_NAME = 0x0902
    CL_PLATFORM_VENDOR = 0x0903
    CL_DEVICE_NAME = 0x102B
    CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
    try:
        ocl = ctypes.WinDLL(os.path.join(SYS32, "OpenCL.dll"))
    except OSError as exc:
        print("  OpenCL.dll load failed:", exc)
        return
    num = ctypes.c_uint(0)
    rc = ocl.clGetPlatformIDs(ctypes.c_uint(0), None, ctypes.byref(num))
    print("  clGetPlatformIDs(0, NULL) rc=%d num_platforms=%d" % (rc, num.value))
    if rc != 0 or num.value == 0:
        return
    plats = (ctypes.c_void_p * num.value)()
    rc2 = ocl.clGetPlatformIDs(num, plats, None)
    print("  clGetPlatformIDs(n) rc=%d" % rc2)

    def info(handle, param):
        size = ctypes.c_size_t(0)
        ocl.clGetPlatformInfo(handle, ctypes.c_uint(param), ctypes.c_size_t(0), None, ctypes.byref(size))
        buf = ctypes.create_string_buffer(max(size.value, 1))
        ocl.clGetPlatformInfo(handle, ctypes.c_uint(param), size, buf, None)
        return buf.value.decode("utf-8", "replace")

    for idx, p in enumerate(plats):
        handle = ctypes.c_void_p(p)
        name = info(handle, CL_PLATFORM_NAME)
        vendor = info(handle, CL_PLATFORM_VENDOR)
        ndev = ctypes.c_uint(0)
        rc3 = ocl.clGetDeviceIDs(handle, ctypes.c_uint64(CL_DEVICE_TYPE_ALL), ctypes.c_uint(0), None,
                                 ctypes.byref(ndev))
        print("  platform[%d] name=%r vendor=%r clGetDeviceIDs rc=%d num_devices=%d"
              % (idx, name, vendor, rc3, ndev.value))


def part5_logs():
    section("5. Audit of E2 logs cited by the claim")
    markers = {
        "boost_compute": "boost_compute",
        "LGB_FATAL": "[LightGBM] [Fatal]",
        "CUDA_MSG": "CUDA Tree Learner was not enabled",
        "GPU_MSG": "GPU Tree Learner was not enabled",
        "Traceback": "Traceback",
        "UnicodeDecodeError": "UnicodeDecodeError",
        "CASE_START": "_CASE_START",
        "CASE_END": "_CASE_END",
        "BENCH_JSON": "E2_BENCH_JSON",
        "ENV_JSON": "E2_ENV_JSON",
        "ABORT_banner": "Application terminated with error",
    }
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(LOGDIR, "E2_*")))
    rows = [n for n in names if ("lgb" in n.lower() or "driver" in n.lower() or "env_check" in n.lower()
                                 or "smoke" in n.lower() or "retry" in n.lower())]
    print("E2 logs selected:", len(rows))
    for n in rows:
        path = os.path.join(LOGDIR, n)
        with open(path, "rb") as fh:
            raw = fh.read()
        text = raw.decode("utf-8", "replace")
        counts = {k: text.count(v) for k, v in markers.items()}
        nonzero = {k: c for k, c in counts.items() if c}
        print("  %-36s bytes=%-6d mtime=%s sha256=%s lines=%d %s"
              % (n, len(raw), fmt_mtime(path), sha256_bytes(raw)[:12], text.count("\n"), nonzero))
    print("")
    print("driver EXIT lines:")
    for n in names:
        if "driver" not in n.lower():
            continue
        path = os.path.join(LOGDIR, n)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "EXIT=" in line:
                    print("  %-28s %s" % (n, line.strip()))
    print("")
    a = os.path.join(LOGDIR, "E2_smoke_lgb_gpu.log")
    b = os.path.join(LOGDIR, "E2_retry_lgb_gpu.log")
    if os.path.exists(a) and os.path.exists(b):
        print("smoke_lgb_gpu vs retry_lgb_gpu identical bytes:", sha256_file(a) == sha256_file(b))
    bench_gpu = os.path.join(LOGDIR, "E2_bench_lgb_gpu.log")
    if os.path.exists(bench_gpu):
        print("E2_bench_lgb_gpu.log bytes=%d mtime=%s (the 16:59:43 OpenCL bench writes here if the driver used this name)"
              % (os.path.getsize(bench_gpu), fmt_mtime(bench_gpu)))


def main():
    print("VERIFY_E2_1 independent static check")
    import sys
    print("python:", sys.version.split()[0], sys.executable)
    data = part1_imports()
    part2_strings(data)
    part3_registry()
    part4_icd()
    part5_logs()


if __name__ == "__main__":
    main()
