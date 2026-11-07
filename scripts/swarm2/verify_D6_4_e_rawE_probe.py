"""D6-4 independent verifier, part E: read-only spot check of the raw E: source for a few affected matches.
Reads are only from E: (never written). Prints the tome round_end / bomb / freeze-end values for rounds 12-14 and,
where a raw match folder exists, the raw round_end and bomb channel rows for the same rounds.
Chooses affected build-9961 24-round matches (round 13 bomb outcome != round_end) spread over days.
"""
import sys, glob, collections, os
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
B24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
RAW_ROOTS = [Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02"),
             Path(r"E:\csds_final\csds\2024\02")]
T, CT = 2, 3


def load(base, ch, cols):
    if ch == "header":
        fd = base / "header"
    else:
        fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f).select(cols) for f in files], how="vertical_relaxed").collect()


H = load(B24, "header", ["key", "build_num", "match_date"])
bm = {k: (b, d) for k, b, d in H.iter_rows()}
RE = load(B24, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
BS = load(B24, "bomb_state", ["match_key", "round", "event_type", "tick"])
outs = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("btick")))
outs = outs.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(CT).otherwise(T).alias("bomb_side"))
J = RE.filter((pl.col("round") == 13) & (pl.col("player_count") == 20)).join(outs, on=["match_key", "round"], how="inner")
J = J.filter(pl.col("bomb_side") != pl.col("winner_team_code"))
J = J.filter(pl.col("match_key").map_elements(lambda m: bm.get(m, (0, ""))[0] == 9961, return_dtype=pl.Boolean))
J = J.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")).sort("day")
print(f"affected build-9961 round-13 bomb rows: {J.height}; days: {sorted(set(J['day'].to_list()))}")
picks = []
for d in ["12", "15", "20", "25"]:
    sub = J.filter(pl.col("day") == d)
    if sub.height:
        picks.append(sub.row(0, named=True))
print(f"picked {len(picks)} matches (one per day among 12, 15, 20, 25)")

for p in picks:
    m = p["match_key"]
    print(f"\n== match {m}  (build {bm[m][0]}, date {bm[m][1]}) ==")
    print(f"tome round 13: round_end winner={p['winner_team_code']} tick={p['tick']} bomb_side={p['bomb_side']} bomb_tick={p['btick']}")
    mid = m.split("/")[-1] if not m.endswith("csds") else m.split("/")[-2]
    print(f"match id component for raw lookup: {mid}")
    found = None
    for root in RAW_ROOTS:
        cands = list(root.glob(f"*/{mid}*")) if root.exists() else []
        if cands:
            found = cands[0]
            break
    if found is None:
        print("  raw match folder: not found under E: candidate roots (read-only check, nothing written)")
        continue
    print(f"  raw match folder: {found}")
    chans = sorted(os.listdir(found))
    print(f"  channels present (first 40): {chans[:40]}")
    for ch in ["round_end", "bomb_state"]:
        cand = found / ch
        if not cand.exists():
            cand = next((found / c for c in chans if c.startswith(ch)), None)
        if cand is None or not cand.exists():
            print(f"  raw channel {ch}: not present")
            continue
        files = sorted(glob.glob(str(cand / "dataframe_*")))
        if not files:
            print(f"  raw channel {ch}: no dataframe_* parquet at {cand}")
            continue
        df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
        cols = [c for c in ["round", "tick", "winner_team_code", "win_reason_code", "event_type", "player_count"] if c in df.columns]
        sel = df.filter(pl.col("round").is_in([12, 13, 14])).select(cols) if "round" in df.columns else df.select(cols)
        print(f"  raw {ch} rows for rounds 12-14 ({sel.height}):")
        print(str(sel.sort([c for c in ["round", "tick"] if c in sel.columns])))
