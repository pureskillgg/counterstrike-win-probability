"""verify D2-6: replicate D2_lead.py rules exactly (pc filter, contiguity on filtered rounds, break at first null sx)
and compare D2's sx with the raw-derived sx round by round. Reuses build() from verify_D2_6_03_main.py.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_6_05_replicate.py
"""
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
MK = "match_key"

spec = importlib.util.spec_from_file_location("v03", ROOT / "scripts" / "swarm2" / "verify_D2_6_03_main.py")
v03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v03)

PC = {"2024": 20, "2026": 10}


def d2_lead(M, label, sx_key):
    """D2_lead.py semantics: rows filtered by pc; contiguous 1..max on filtered rounds; chain breaks at first null."""
    n_contig = checked = bad_r = 0
    bad_m = set()
    first = Counter()
    for mk, byr in M.items():
        rs_f = sorted(rd for rd, r in byr.items() if r["pc"] == PC[label])
        if not rs_f or rs_f[0] != 1 or rs_f[-1] != len(rs_f):
            continue
        n_contig += 1
        rmax = rs_f[-1]
        scoreX = scoreY = 0
        first_bad = None
        for rd in range(1, rmax + 1):
            r = byr[rd]
            sx = r[sx_key]
            R = r["R"]
            if sx is None or R is None:
                break
            if r["t0"] is not None and r["ct0"] is not None:
                rs_lead = r["ct0"] - r["t0"]
                tracked = (scoreX - scoreY) if sx == 3 else (scoreY - scoreX)
                checked += 1
                if tracked != rs_lead:
                    bad_r += 1
                    bad_m.add(mk)
                    if first_bad is None:
                        first_bad = rd
            if R == sx:
                scoreX += 1
            else:
                scoreY += 1
        if first_bad is not None:
            first[first_bad] += 1
    return {"contig": n_contig, "checked": checked, "bad_rounds": bad_r, "bad_matches": len(bad_m),
            "first_hist": sorted(first.items())[:10]}


def main():
    label = "2024"
    rt, panel, hd, info = v03.build(label)
    rows = rt.sort([MK, "round"]).to_dicts()
    M = defaultdict(dict)
    for r in rows:
        M[r[MK]][r["round"]] = r

    d2 = pl.read_parquet(ROOT / "logs" / "swarm2" / f"D2_rounds_{label}.parquet").select(
        [MK, "round", "sx", "winner_team_code", "player_count"]).rename({"sx": "sx_D2", "winner_team_code": "R_D2"})
    print(f"D2_rounds rows {d2.height}; pc values {d2['player_count'].unique().to_list()}")
    j = rt.select([MK, "round", "sx", "R", "pc"]).join(d2, on=[MK, "round"], how="inner")
    print(f"joined rounds (raw table x D2_rounds): {j.height}")
    print("R equal to D2 winner_team_code:", j.filter(pl.col("R") == pl.col("R_D2")).height,
          " differ:", j.filter(pl.col("R") != pl.col("R_D2")).height)
    both = j.filter(pl.col("sx").is_not_null() & pl.col("sx_D2").is_not_null())
    print(f"sx both non-null: {both.height}; equal {both.filter(pl.col('sx') == pl.col('sx_D2')).height}; "
          f"differ {both.filter(pl.col('sx') != pl.col('sx_D2')).height}")
    print("sx null in raw only:", j.filter(pl.col("sx").is_null() & pl.col("sx_D2").is_not_null()).height,
          " null in D2 only:", j.filter(pl.col("sx").is_not_null() & pl.col("sx_D2").is_null()).height,
          " null in both:", j.filter(pl.col("sx").is_null() & pl.col("sx_D2").is_null()).height)
    print("rounds in D2 table missing from raw table:",
          d2.join(rt.select([MK, "round"]), on=[MK, "round"], how="anti").height)
    print("pc==20 rounds in raw table:", rt.filter(pl.col("pc") == 20).height)

    print("\nD2_lead replication, raw-derived sx:", d2_lead(M, label, "sx"))

    # D2 sx in the same structure
    M2 = defaultdict(dict)
    dd = j.select([MK, "round", "sx_D2", "R", "t0", "ct0", "pc"]) if False else None
    d2map = {(r[MK], r["round"]): r["sx_D2"] for r in d2.to_dicts()}
    for mk, byr in M.items():
        for rd, r in byr.items():
            r["sx_d2"] = d2map.get((mk, rd), None) if (mk, rd) in d2map else None
    print("D2_lead replication, D2 sx (break at null):", d2_lead(M, label, "sx_d2"))


if __name__ == "__main__":
    main()
