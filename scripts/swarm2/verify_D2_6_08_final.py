"""verify D2-6: final cross-tab on the full population (raw sx restricted to {2,3}, all contiguous matches).

  Truth T = score-chain winner where known, else bomb/elimination winner (T_src recorded).
  Flagged match = any round with R != T. Divergent match = R-tracked lead != round_state lead (break at first
  round without a valid sx; no pc filter). Classifies flagged-but-clean matches and checks the rounds 1-12 flips.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_6_08_final.py
"""
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
MK = "match_key"
PC = {"2024": 20, "2026": 10}

spec = importlib.util.spec_from_file_location("v07", ROOT / "scripts" / "swarm2" / "verify_D2_6_07_sx_validation.py")
v07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v07)
v03 = v07.v03


def run(label):
    import polars as pl
    rt, panel, hd, info = v03.build(label)
    pi = v07.scan(label, "player_info").select([MK, "round", "player_id", "team_code"]).collect()
    pi = pi.unique([MK, "round", "player_id"], keep="first", maintain_order=True)
    cnt = pi.group_by([MK, "round"]).agg((pl.col("team_code") == 2).sum().alias("n2"),
                                         (pl.col("team_code") == 3).sum().alias("n3"))
    valid = (cnt.filter((pl.col("n2") == 5) & (pl.col("n3") == 5)).sort([MK, "round"])
             .unique([MK], keep="first", maintain_order=True).select([MK, pl.col("round").alias("F")]))
    xm = (pi.join(valid, on=MK).filter(pl.col("round") == pl.col("F")).filter(pl.col("team_code") == 2)
          .select([MK, "player_id"]).unique().with_columns(pl.lit(1).alias("isX")))
    pix = pi.join(xm, on=[MK, "player_id"], how="left")
    xy = pix.group_by([MK, "round"]).agg((pl.col("isX") == 1).sum().alias("nX_all"),
                                         ((pl.col("isX").is_null()) & pl.col("team_code").is_in([2, 3])).sum().alias("nY"))
    rt = rt.join(xy, on=[MK, "round"], how="left")
    M = defaultdict(dict)
    for r in rt.sort([MK, "round"]).to_dicts():
        M[r[MK]][r["round"]] = r
    for mk, byr in M.items():
        for rd, r in byr.items():
            r["sx"] = r["sx"] if r["sx"] in (2, 3) else None
            r["nX"] = r.get("nX_all") or 0
            prev = byr.get(rd - 1)
            r["swap_in"] = bool(prev and r["sx"] and prev["sx"] and r["sx"] != prev["sx"])
            r["W_sc"] = None
            sx, sxn = r["sx"], r["sx_next"]
            if sx and sxn in (2, 3) and None not in (r["t0"], r["ct0"], r["t1"], r["ct1"]):
                x0 = r["t0"] if sx == 2 else r["ct0"]
                y0 = r["ct0"] if sx == 2 else r["t0"]
                x1 = r["t1"] if sxn == 2 else r["ct1"]
                y1 = r["ct1"] if sxn == 2 else r["t1"]
                dX, dY = x1 - x0, y1 - y0
                if dX == 1 and dY == 0:
                    r["W_sc"] = sx
                elif dY == 1 and dX == 0:
                    r["W_sc"] = 5 - sx
            r["W_ind"] = r["bomb_side"] if r["bomb_side"] is not None else r["elim_side"]
            if r["W_sc"] is not None:
                r["T"], r["T_src"] = r["W_sc"], "score"
            elif r["W_ind"] is not None:
                r["T"], r["T_src"] = r["W_ind"], ("bomb" if r["bomb_side"] is not None else "elim")
            else:
                r["T"], r["T_src"] = None, None

    print(f"\n################ {label} (full population, raw sx in {{2,3}}) ################")
    cross = Counter()
    flagged_13 = 0
    div_matches = set()
    flagged = set()
    diff_hist = Counter()
    cnt_m = 0
    stop_reasons = Counter()
    flip_1_12_rows = []
    for mk, byr in M.items():
        rs_all = sorted(byr)
        if not rs_all or rs_all[0] != 1 or rs_all[-1] != len(rs_all):
            continue
        cnt_m += 1
        rmax = rs_all[-1]
        # R-chain (break at first round without a valid sx)
        scoreX = scoreY = 0
        first_bad = None
        stop = None
        checked = 0
        for rd in range(1, rmax + 1):
            r = byr[rd]
            sx, R = r["sx"], r["R"]
            if sx is None:
                stop = rd
                break
            if r["t0"] is not None and r["ct0"] is not None:
                rs_lead = r["ct0"] - r["t0"]
                tracked = (scoreX - scoreY) if sx == 3 else (scoreY - scoreX)
                checked += 1
                if tracked != rs_lead:
                    diff_hist[tracked - rs_lead] += 1
                    if first_bad is None:
                        first_bad = rd
            scoreX += 1 if R == sx else 0
            scoreY += 0 if R == sx else 1
        flips = [rd for rd in rs_all if byr[rd]["T"] is not None and byr[rd]["R"] != byr[rd]["T"]]
        for rd in flips:
            if rd <= 12:
                flip_1_12_rows.append((mk, rd, byr[rd]["T_src"], rmax, first_bad, stop))
        if flips:
            flagged.add(mk)
        if flips and 13 in flips:
            flagged_13 += 1
        if first_bad is not None:
            div_matches.add(mk)
        if flips and first_bad is not None:
            cross["flagged & divergent"] += 1
            cross["divergent: first_bad==firstflip+1" if first_bad == flips[0] + 1 else "divergent: first_bad!=firstflip+1"] += 1
        elif flips and first_bad is None:
            if stop is not None and stop <= (flips[0] + 1) and flips[0] + 1 <= rmax:
                cross["flagged, clean: chain stops at/before first flip+1 (raw sx null)"] += 1
            elif flips[0] == rmax:
                cross["flagged, clean: flip only in last round"] += 1
            elif flips[0] + 1 > rmax:
                cross["flagged, clean: flip only in last round"] += 1
            else:
                cross["flagged, clean: other"] += 1
                if cross["flagged, clean: other"] <= 5:
                    print("   other:", mk, flips[:5], "rmax", rmax, "stop", stop, "first_bad", first_bad)
        elif not flips and first_bad is not None:
            cross["divergent, no flip under T"] += 1
        else:
            cross["clean, no flip"] += 1
    print(f"  contiguous matches: {cnt_m}")
    for k, v in sorted(cross.items()):
        print(f"    {k}: {v}")
    print(f"  flagged matches (any R!=T): {len(flagged)}; flagged at round 13: {flagged_13}")
    print(f"  divergent matches (R-chain): {len(div_matches)}; both: {len(div_matches & flagged)}")
    print(f"  tracked - round_state lead on divergent rounds (value: count): {sorted(diff_hist.items())}")
    print(f"  flips in rounds 1-12 (match, round, T_src, rmax, first_bad, stop): {len(flip_1_12_rows)} rows")
    for x in flip_1_12_rows:
        print("     ", x)


def main():
    for label in ["2024", "2026"]:
        run(label)


if __name__ == "__main__":
    main()
