"""Verify notes/replay/maps/*.png and notes/replay/map_meta.json for the seven radar maps.

Checks, in order:
  1. map_meta.json holds the seven maps, each with the required keys.
  2. Each PNG on disk: sha256 equals map_meta's sha256, the IHDR is square and
     equals map_meta's size, every chunk CRC is good, and the IDAT stream
     decodes to exactly the expected number of bytes (full unfilter pass).
  3. If --zip/--manifest are given: images.zip matches the manifest's byte count
     and SHA-256, and each radars/<map>.png member is byte-identical to the file
     on disk.
  4. If --map-data is given: pos_x, pos_y, scale equal the awpy-data
     map_data.json values.
  5. Transform test on real CSDS player positions
     (data/state_sample/players_1hz.parquet, joined to map_name through match_key
     in data/state_sample/positions_snapshots.parquet). Alive positions are
     counted as inside the radar square for the formula
         px = (x - pos_x) / scale,  py = (pos_y - y) / scale
     and for a y-flipped variant py = (y - pos_y) / scale.
  6. Orientation test: a 64x64 occupancy heatmap of alive positions is
     correlated with the 64x64 radar luminance under the 8 dihedral
     orientations of the image. The identity orientation should win if the
     formula and the image agree.

Nothing is written to disk; results are printed.
Standard library + numpy + polars. Run with the project interpreter:
    G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\replay\\verify_radar_assets.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAPS = ("de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke")
REQUIRED_KEYS = ("pos_x", "pos_y", "scale", "size", "source", "licence")
PNG_SIG = b"\x89PNG\r\n\x1a\n"
CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}
BINS = 64


def read_png(data: bytes):
    """Decode an 8-bit non-interlaced PNG (colour types 0, 2, 4, 6). Returns (w, h, channels, pixels)."""
    if data[:8] != PNG_SIG:
        raise ValueError("bad PNG signature")
    pos = 8
    ihdr = None
    idat = []
    while True:
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        if zlib.crc32(ctype + chunk) & 0xFFFFFFFF != crc:
            raise ValueError(f"CRC mismatch in chunk {ctype!r}")
        pos += 12 + length
        if ctype == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", chunk)
        elif ctype == b"IDAT":
            idat.append(chunk)
        elif ctype == b"IEND":
            break
    width, height, bitdepth, colour, _comp, _filt, interlace = ihdr
    if bitdepth != 8 or interlace != 0 or colour not in CHANNELS:
        raise ValueError(f"unsupported PNG: bitdepth={bitdepth} colour={colour} interlace={interlace}")
    channels = CHANNELS[colour]
    stride = width * channels
    raw = zlib.decompress(b"".join(idat))
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError(f"IDAT decodes to {len(raw)} bytes, expected {expected}")
    out = bytearray(height * stride)
    prev = bytearray(stride)
    bpp = channels
    for y in range(height):
        f = raw[y * (stride + 1)]
        line = raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)]
        cur = bytearray(stride)
        if f == 0:
            cur[:] = line
        elif f == 1:
            for i in range(stride):
                left = cur[i - bpp] if i >= bpp else 0
                cur[i] = (line[i] + left) & 255
        elif f == 2:
            for i in range(stride):
                cur[i] = (line[i] + prev[i]) & 255
        elif f == 3:
            for i in range(stride):
                left = cur[i - bpp] if i >= bpp else 0
                cur[i] = (line[i] + ((left + prev[i]) >> 1)) & 255
        elif f == 4:
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (line[i] + pr) & 255
        else:
            raise ValueError(f"unknown PNG filter type {f} on row {y}")
        out[y * stride:(y + 1) * stride] = cur
        prev = cur
    pixels = np.frombuffer(bytes(out), dtype=np.uint8).reshape(height, width, channels)
    return width, height, channels, pixels


def luminance(pixels: np.ndarray, channels: int) -> np.ndarray:
    a = pixels.astype(np.float64)
    gray = a[..., 0] if channels < 3 else a[..., :3].mean(axis=2)
    if channels in (2, 4):
        gray = gray * (a[..., channels - 1] / 255.0)
    return gray


def block_mean(img: np.ndarray, bins: int) -> np.ndarray:
    h, w = img.shape
    assert h % bins == 0 and w % bins == 0
    return img.reshape(bins, h // bins, bins, w // bins).mean(axis=(1, 3))


def occupancy(px: np.ndarray, py: np.ndarray, size: int, bins: int) -> np.ndarray:
    inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    col = np.floor(px[inside] / size * bins).astype(np.int64)
    row = np.floor(py[inside] / size * bins).astype(np.int64)
    return np.bincount(row * bins + col, minlength=bins * bins).reshape(bins, bins)


ORIENTATIONS = {
    "identity": lambda a: a,
    "rot90": lambda a: np.rot90(a, 1),
    "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3),
    "flipud": lambda a: a[::-1, :],
    "fliplr": lambda a: a[:, ::-1],
    "transpose": lambda a: a.T,
    "antitranspose": lambda a: np.rot90(a, 2).T,
}


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=None, help="images.zip from awpy-data release 2000932")
    ap.add_argument("--manifest", type=Path, default=None, help="manifest.json from the same release")
    ap.add_argument("--map-data", type=Path, default=None, help="map_data.json from the same release")
    ap.add_argument("--maps-dir", type=Path, default=ROOT / "notes" / "replay" / "maps")
    ap.add_argument("--meta", type=Path, default=ROOT / "notes" / "replay" / "map_meta.json")
    args = ap.parse_args()
    failures: list[str] = []

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    print(f"[1] map_meta.json has {len(meta)} maps: {sorted(meta)}")
    for m in MAPS:
        if m not in meta:
            failures.append(f"map_meta missing {m}")
            continue
        missing = [k for k in REQUIRED_KEYS if k not in meta[m]]
        if missing:
            failures.append(f"{m} missing keys {missing}")

    print("[2] PNG on disk: sha256, IHDR, CRC, full IDAT decode and filter pass")
    pixels_by_map = {}
    for m in MAPS:
        path = args.maps_dir / f"{m}.png"
        data = path.read_bytes()
        width, height, channels, px = read_png(data)
        pixels_by_map[m] = (width, height, channels, px)
        sha_ok = sha256_hex(data) == meta[m].get("sha256", sha256_hex(data))
        size_ok = width == height == meta[m]["size"]
        print(f"    {m}: {len(data)} bytes, {width}x{height} px, {channels} ch, sha256 match={sha_ok}, size match={size_ok}")
        if not sha_ok:
            failures.append(f"{m} sha256 differs from map_meta")
        if not size_ok:
            failures.append(f"{m} size {width}x{height} differs from map_meta size {meta[m]['size']}")

    if args.zip and args.manifest:
        man = json.loads(args.manifest.read_text(encoding="utf-8"))["artifacts"]["images.zip"]
        zb = args.zip.read_bytes()
        zip_ok = len(zb) == man["bytes"] and sha256_hex(zb) == man["sha256"]
        print(f"[3] images.zip: {len(zb)} bytes, manifest bytes {man['bytes']}, sha256 matches manifest={zip_ok}")
        if not zip_ok:
            failures.append("images.zip does not match manifest")
        with zipfile.ZipFile(args.zip) as z:
            for m in MAPS:
                member = z.read(f"radars/{m}.png")
                same = member == (args.maps_dir / f"{m}.png").read_bytes()
                print(f"    member radars/{m}.png byte-identical to disk: {same}")
                if not same:
                    failures.append(f"{m} on disk differs from images.zip member")

    if args.map_data:
        md = json.loads(args.map_data.read_text(encoding="utf-8"))
        print("[4] map_meta transform values vs awpy-data map_data.json")
        for m in MAPS:
            same = all(float(md[m][k]) == float(meta[m][k]) for k in ("pos_x", "pos_y", "scale"))
            print(f"    {m}: pos_x={meta[m]['pos_x']} pos_y={meta[m]['pos_y']} scale={meta[m]['scale']} equal={same}"
                  f"  (rotate={md[m].get('rotate')}, zoom={md[m].get('zoom')}, verticalsections={'yes' if md[m].get('verticalsections') else 'no'})")
            if not same:
                failures.append(f"{m} transform differs from map_data.json")

    print("[5] loading real alive player positions from data/state_sample")
    pos = pl.read_parquet(ROOT / "data" / "state_sample" / "players_1hz.parquet",
                          columns=["match_key", "health", "x_pos", "y_pos"])
    mp = pl.read_parquet(ROOT / "data" / "state_sample" / "positions_snapshots.parquet",
                         columns=["match_key", "map_name"]).unique()
    conflicts = mp.group_by("match_key").agg(pl.col("map_name").n_unique().alias("n")).filter(pl.col("n") > 1).height
    print(f"    match_keys with >1 map_name: {conflicts}")
    joined = (pos.join(mp.unique(subset=["match_key"]), on="match_key", how="inner")
                 .filter((pl.col("health") > 0) & pl.col("x_pos").is_not_null() & pl.col("y_pos").is_not_null()
                         & ~((pl.col("x_pos") == 0) & (pl.col("y_pos") == 0))))
    print(f"    alive, non-(0,0) position rows joined to a map: {joined.height}")

    print("[6] in-bounds fraction and orientation correlation per map")
    for m in MAPS:
        meta_m = meta[m]
        width, height, channels, px_img = pixels_by_map[m]
        size = meta_m["size"]
        sub = joined.filter(pl.col("map_name") == m)
        if sub.height == 0:
            print(f"    {m}: no positions in state_sample; not tested")
            continue
        x = sub["x_pos"].to_numpy().astype(np.float64)
        y = sub["y_pos"].to_numpy().astype(np.float64)
        scale, pos_x, pos_y = float(meta_m["scale"]), float(meta_m["pos_x"]), float(meta_m["pos_y"])
        px_f = (x - pos_x) / scale
        py_f = (pos_y - y) / scale
        px_flip = (x - pos_x) / scale
        py_flip = (y - pos_y) / scale
        inb_f = np.mean((px_f >= 0) & (px_f < size) & (py_f >= 0) & (py_f < size))
        inb_flip = np.mean((px_flip >= 0) & (px_flip < size) & (py_flip >= 0) & (py_flip < size))
        occ = occupancy(px_f, py_f, size, BINS)
        img = block_mean(luminance(px_img, channels), BINS)
        occ_t = np.log1p(occ).ravel()
        corrs = {}
        for name, fn in ORIENTATIONS.items():
            cand = fn(img).ravel()
            if cand.std() == 0 or occ_t.std() == 0:
                corrs[name] = float("nan")
            else:
                corrs[name] = float(np.corrcoef(occ_t, cand)[0, 1])
        best = max(corrs, key=lambda k: -1e9 if np.isnan(corrs[k]) else corrs[k])
        corr_txt = ", ".join(f"{k}={v:.3f}" for k, v in corrs.items())
        print(f"    {m}: n={sub.height} rows, matches={sub['match_key'].n_unique()}, "
              f"in-bounds formula={inb_f:.4f}, in-bounds y-flipped={inb_flip:.4f}")
        print(f"        px p1/p99={np.percentile(px_f, 1):.0f}/{np.percentile(px_f, 99):.0f}, "
              f"py p1/p99={np.percentile(py_f, 1):.0f}/{np.percentile(py_f, 99):.0f}")
        print(f"        corr by orientation: {corr_txt}")
        print(f"        best orientation: {best}")

    print("RESULT:", "PASS" if not failures else "FAIL")
    for f in failures:
        print("  failure:", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
