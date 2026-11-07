import sys, struct, ast
data = open(sys.argv[1], "rb").read()
i = 0; seen = []
while True:
    j = data.find(b"PK\x03\x04", i)
    if j < 0: break
    if j + 30 > len(data): break
    (_, _, flag, comp, _, _, _, csize, usize, nlen, xlen) = struct.unpack("<IHHHHHIIIHH", data[j:j+30])
    name = data[j+30:j+30+nlen].decode("utf-8", "replace")
    start = j + 30 + nlen + xlen
    if comp == 0 and name.endswith(".npy") and start + 10 < len(data):
        # parse npy header for dtype and shape only (no array values)
        magic_ok = data[start:start+6] == b"\x93NUMPY"
        if magic_ok:
            ver_major = data[start+6]
            if ver_major == 1:
                hlen = struct.unpack("<H", data[start+8:start+10])[0]; hstart = start+10
            else:
                hlen = struct.unpack("<I", data[start+8:start+12])[0]; hstart = start+12
            hdr = ast.literal_eval(data[hstart:hstart+hlen].decode("latin1"))
            seen.append((name, hdr["descr"], hdr["shape"]))
        else:
            seen.append((name, "no-npy-magic", None))
    else:
        seen.append((name, "compressed-or-truncated" if comp else "", usize))
    i = j + 4
for s in seen: print(s)
print("members found:", len(seen))
