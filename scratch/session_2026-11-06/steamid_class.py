"""Count-only: does steam_id hold map names, team codes or numbers? Prints counts and length histogram only."""
import glob, re, sys
from collections import Counter
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
files = {
    "2024-02-g": r"G:/pureskill/csds-cs2/evac_2026-10-13/tomes/2024-02-g/tome/csds/basic_player_personal.2023-12-28,2024-02-25/dataframe_00000",
    "2026-10-01_to_08": r"G:/pureskill/csds-cs2/evac_2026-10-13/tomes/2026-10-01_to_08/tome/csds/basic_player_personal.2026-09-18,2026-10-08/dataframe_00000",
}
MAP_RE = re.compile(r"^(de_)?(ancient|anubis|cache|dust2|dust_2|inferno|mirage|nuke|train|overpass|vertigo|office|italy|agency)$", re.I)
TEAM_RE = re.compile(r"^(ct|t|terrorist|terrorists|counter-?terrorist|counterterrorist|team_?[0-9a-z]|side_?[0-9]|[0-9]|1[0-9]|2[0-9])$", re.I)
for label, path in files.items():
    vals = pq.read_table(path, columns=["steam_id"]).column("steam_id").to_pylist()
    nonempty = [str(v) for v in vals if v not in (None, "")]
    c = Counter(nonempty)
    print(f"{label}: nonempty={len(nonempty)} distinct={len(c)}")
    print("   map-name pattern rows:", sum(bool(MAP_RE.match(v)) for v in nonempty))
    print("   team/slot-code pattern rows:", sum(bool(TEAM_RE.match(v)) for v in nonempty))
    print("   distinct values matching map-name pattern:", sum(1 for v in c if MAP_RE.match(v)), "| team/slot-code pattern:", sum(1 for v in c if TEAM_RE.match(v)))
    print("   length histogram (length: distinct values):", sorted(Counter(len(v) for v in c).items()))
    print("   contains letters share (distinct):", round(sum(1 for v in c if re.search(r'[A-Za-z]', v)) / len(c), 3))
