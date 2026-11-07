"""verify D2-6: validate team identity X and its side sx against the death record (player_team_code, side-attached),
and re-run the D2-rule and unfiltered chains with sx restricted to codes {2, 3} (no invalid team_code 0 rosters).

  V1. For each death of a player who is on team X (membership from the first valid 5v5 round), the victim's
      player_team_code should equal sx(round). Counts by year, and by whether D2's sx is null on that round.
  V2. Composition of player_info rounds where D2's sx is null but raw sx is set (n code 2, n code 3).
  V3. Chains (break at first null sx, D2 pc filter and contiguity) using sx restricted to {2,3}: D2 sx and raw sx.
  V4. Same chain with raw sx but requiring exactly five X players and five Y players present in the round.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_6_07_sx_validation.py
"""
import glob
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
MK = "match_key"
PC = {"2024": 20, "2026": 10}

spec = importlib.util.spec_from_file_location("v03", ROOT / "scripts" / "swarm2" / "verify_D2_6_03_main.py")
v03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v03)


def scan(label, ch):
    d = Path(sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0])
    return pl.scan_parquet(sorted(glob.glob(str(d / "dataframe_*"))))


def chain(byr, rs_sel, sx_key, need5=False):
    if not rs_sel or rs_sel[0] != 1 or rs_sel[-1] != len(rs_sel):
        return None
    rmax = rs_sel[-1]
    scoreX = scoreY = 0
    checked = bad = 0
    first_bad = None
    for rd in range(1, rmax + 1):
        r = byr[rd]
        sx, R = r[sx_key], r["R"]
        if sx is None or R is None or sx not in (2, 3):
            break
        if need5 and not (r["nX"] == 5 and r["nY"] == 5):
            break
        if r["t0"] is not None and r["ct0"] is not None:
            rs_lead = r["ct0"] - r["t0"]
            tracked = (scoreX - scoreY) if sx == 3 else (scoreY - scoreX)
            checked += 1
            if tracked != rs_lead:
                bad += 1
                if first_bad is None:
                    first_bad = rd
        scoreX += 1 if R == sx else 0
        scoreY += 0 if R == sx else 1
    return {"checked": checked, "bad": bad, "first_bad": first_bad}


def main():
    for label in ["2024", "2026"]:
        rt, panel, hd, info = v03.build(label)
        # raw roster per round for Y count, and X membership from first valid round
        pi = scan(label, "player_info").select([MK, "round", "player_id", "team_code"]).collect()
        pi = pi.unique([MK, "round", "player_id"], keep="first", maintain_order=True)
        cnt = pi.group_by([MK, "round"]).agg((pl.col("team_code") == 2).sum().alias("n2"),
                                             (pl.col("team_code") == 3).sum().alias("n3"))
        valid = (cnt.filter((pl.col("n2") == 5) & (pl.col("n3") == 5)).sort([MK, "round"])
                 .unique([MK], keep="first", maintain_order=True).select([MK, pl.col("round").alias("F")]))
        xm = (pi.join(valid, on=MK).filter(pl.col("round") == pl.col("F")).filter(pl.col("team_code") == 2)
              .select([MK, "player_id"]).unique().with_columns(pl.lit(1).alias("isX")))
        pix = pi.join(xm, on=[MK, "player_id"], how="left")
        # per-round X and Y presence counts (raw sx uses X codes only)
        xy = pix.group_by([MK, "round"]).agg(
            (pl.col("isX") == 1).sum().alias("nX_all"),
            ((pl.col("isX").is_null()) & (pl.col("team_code") == 2)).sum().alias("n2_Y"),
            ((pl.col("isX").is_null()) & (pl.col("team_code") == 3)).sum().alias("n3_Y"),
            ((pl.col("isX") == 1) & (pl.col("team_code") == 2)).sum().alias("nX2"),
            ((pl.col("isX") == 1) & (pl.col("team_code") == 3)).sum().alias("nX3"),
        )
        rt = rt.join(xy, on=[MK, "round"], how="left")
        rows = rt.sort([MK, "round"]).to_dicts()
        M = defaultdict(dict)
        for r in rows:
            M[r[MK]][r["round"]] = r
        # sx restricted to {2,3} from X codes, and Y count for need5
        for mk, byr in M.items():
            for rd, r in byr.items():
                nX = (r.get("nX_all") or 0)
                r["nX"] = nX
                r["nY"] = (r.get("n2_Y") or 0) + (r.get("n3_Y") or 0)
                if r["sx"] not in (2, 3):
                    r["sx"] = None
        # D2 sx (for the D2-rule chain)
        d2 = pl.read_parquet(ROOT / "logs" / "swarm2" / f"D2_rounds_{label}.parquet").select(
            [MK, "round", "sx"]).rename({"sx": "sx_d2"})
        d2map = {(r[MK], r["round"]): r["sx_d2"] for r in d2.to_dicts()}
        for mk, byr in M.items():
            for rd, r in byr.items():
                r["sx_d2"] = d2map.get((mk, rd))
        print(f"\n################ {label} ################")
        # V1: X victims vs sx
        xdeath = (scan(label, "player_death").select([MK, "round", "player_id", "player_team_code"]).collect()
                  .join(xm, on=[MK, "player_id"], how="inner").filter(pl.col("player_team_code").is_in([2, 3])))
        v1 = Counter()
        for row in xdeath.to_dicts():
            byr = M.get(row[MK])
            if byr is None or row["round"] not in byr:
                v1["no round row"] += 1
                continue
            r = byr[row["round"]]
            if r["sx"] is None:
                v1["sx null (raw)"] += 1
                continue
            tag = "D2 sx null" if r["sx_d2"] is None else "D2 sx set"
            v1[(tag, "agree" if row["player_team_code"] == r["sx"] else "DISAGREE")] += 1
        print("  V1 X-victim death side vs sx:", dict(v1))

        # V2: composition where D2 sx is null but raw sx set
        comp = Counter()
        for mk, byr in M.items():
            for rd, r in byr.items():
                if r["sx_d2"] is None and r["sx"] is not None:
                    comp[(r["nX2"] if r["sx"] == 2 else r["nX3"], "n_X_on_sx", r["nY"])] += 1
        print("  V2 D2-null rounds with raw sx: (n X on sx, n Y present) counts:", sorted(comp.items(), key=lambda kv: -kv[1])[:8])

        # V3/V4 chains
        for sx_key, label2 in [("sx_d2", "D2 sx"), ("sx", "raw sx in {2,3}")]:
            d2rule = dict(n=0, checked=0, bad=0, matches=0, first=Counter())
            allr = dict(n=0, checked=0, bad=0, matches=0, first=Counter())
            n5 = dict(n=0, checked=0, bad=0, matches=0, first=Counter())
            for mk, byr in M.items():
                rs_f = sorted(rd for rd, r in byr.items() if r["pc"] == PC[label])
                res = chain(byr, rs_f, sx_key)
                if res is not None:
                    d2rule["n"] += 1
                    d2rule["checked"] += res["checked"]
                    d2rule["bad"] += res["bad"]
                    if res["bad"]:
                        d2rule["matches"] += 1
                        d2rule["first"][res["first_bad"]] += 1
                rs_a = sorted(byr)
                res = chain(byr, rs_a, sx_key)
                if res is not None:
                    allr["n"] += 1
                    allr["checked"] += res["checked"]
                    allr["bad"] += res["bad"]
                    if res["bad"]:
                        allr["matches"] += 1
                        allr["first"][res["first_bad"]] += 1
                res = chain(byr, rs_a, sx_key, need5=True)
                if res is not None:
                    n5["n"] += 1
                    n5["checked"] += res["checked"]
                    n5["bad"] += res["bad"]
                    if res["bad"]:
                        n5["matches"] += 1
                        n5["first"][res["first_bad"]] += 1
            print(f"  V3 [{label2}] D2 pc filter: contig={d2rule['n']} checked={d2rule['checked']} bad_rounds={d2rule['bad']} "
                  f"bad_matches={d2rule['matches']} first={sorted(d2rule['first'].items())[:6]}")
            print(f"  V3 [{label2}] all rounds (no pc): contig={allr['n']} checked={allr['checked']} bad_rounds={allr['bad']} "
                  f"bad_matches={allr['matches']} first={sorted(allr['first'].items())[:6]}")
            print(f"  V4 [{label2}] all rounds, need 5 X and 5 Y present: contig={n5['n']} checked={n5['checked']} "
                  f"bad_rounds={n5['bad']} bad_matches={n5['matches']} first={sorted(n5['first'].items())[:6]}")


if __name__ == "__main__":
    main()
