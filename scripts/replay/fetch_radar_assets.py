"""Fetch the radar PNGs for seven CS2 maps and write notes/replay/map_meta.json.

Source: awpy-data release 2000932 (pnxenopoulos/awpy-data, MIT packaging).
Inputs, all from that release:
  images.zip     radar PNGs (radars/<map>.png). Checked against manifest.json.
  manifest.json  byte count and SHA-256 of images.zip.
  map_data.json  pos_x, pos_y and scale per map (the world-to-radar transform).

Only the seven maps listed in MAPS are read out of images.zip. The other
members of the archive are never written to disk.

Transform (matches awpy's own plot code, crates/awpy-python/python/awpy/plot.py,
world_to_pixel):
  px = (x - pos_x) / scale
  py = (pos_y - y) / scale
  The radar PNG is size x size pixels with its origin at the top-left corner.

Run with the project interpreter:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\replay\\fetch_radar_assets.py ^
      --zip <images.zip> --manifest <manifest.json> --map-data <map_data.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path

RELEASE = "2000932"
RELEASE_BASE = f"https://github.com/pnxenopoulos/awpy-data/releases/download/{RELEASE}"
MAPS = (
    "de_ancient",
    "de_anubis",
    "de_cache",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
)
LICENCE = (
    "MIT (pnxenopoulos/awpy-data LICENSE, Copyright (c) 2025 Peter Xenopoulos) covers the "
    "packaging and the coordinate table. The radar images are derived from Valve CS2 game "
    "assets, which are not MIT-licensed; that licence is not verified here."
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_SIZE = 1024


def png_header(data: bytes) -> tuple[int, int, int, int]:
    """Return (width, height, bit_depth, colour_type) from a PNG's IHDR chunk."""
    if data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise ValueError("not a PNG file with an IHDR chunk")
    width, height = struct.unpack(">II", data[16:24])
    return width, height, data[24], data[25]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Fetch seven CS2 radar PNGs and their transform table.")
    parser.add_argument("--zip", required=True, type=Path, help="path to images.zip from the release")
    parser.add_argument("--manifest", required=True, type=Path, help="path to manifest.json from the release")
    parser.add_argument("--map-data", required=True, type=Path, help="path to map_data.json from the release")
    parser.add_argument("--out-dir", type=Path, default=root / "notes" / "replay" / "maps")
    parser.add_argument("--meta", type=Path, default=root / "notes" / "replay" / "map_meta.json")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = manifest["artifacts"]["images.zip"]
    zip_bytes = args.zip.read_bytes()
    if len(zip_bytes) != expected["bytes"]:
        raise SystemExit(f"images.zip is {len(zip_bytes)} bytes, manifest says {expected['bytes']}")
    digest = hashlib.sha256(zip_bytes).hexdigest()
    if digest != expected["sha256"]:
        raise SystemExit(f"images.zip SHA-256 {digest} does not match manifest {expected['sha256']}")
    print(f"images.zip verified: {len(zip_bytes)} bytes, sha256 {digest}")

    map_data = json.loads(args.map_data.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict] = {}
    with zipfile.ZipFile(args.zip) as archive:
        for name in MAPS:
            member = f"radars/{name}.png"
            info = archive.getinfo(member)
            print(f"before saving {name}: {member}, {info.file_size} bytes uncompressed, from {RELEASE_BASE}/images.zip")
            png = archive.read(member)
            width, height, bit_depth, colour_type = png_header(png)
            if width != height:
                raise SystemExit(f"{name}: radar is {width}x{height}, expected a square image")
            if width > MAX_SIZE:
                print(f"WARNING {name}: {width} px is above {MAX_SIZE}; no imaging library, so the original is kept")

            out_path = args.out_dir / f"{name}.png"
            out_path.write_bytes(png)
            saved = out_path.read_bytes()
            entry = map_data[name]
            meta[name] = {
                "pos_x": float(entry["pos_x"]),
                "pos_y": float(entry["pos_y"]),
                "scale": float(entry["scale"]),
                "size": width,
                "file": f"maps/{name}.png",
                "sha256": hashlib.sha256(saved).hexdigest(),
                "source": f"{RELEASE_BASE}/images.zip (member {member}); transform from {RELEASE_BASE}/map_data.json",
                "licence": LICENCE,
            }
            print(
                f"after saving {name}: {out_path} {len(saved)} bytes, {width}x{height} px, "
                f"bit depth {bit_depth}, colour type {colour_type}"
            )

    args.meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.meta} with {len(meta)} maps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
