"""verify D5-9, step 7: the same independent checks on the 2026 tome (claim: 2026 not checked for MR15 / regime A).

Builds per-(match, round) freeze-end fields, round_end winner, bomb outcome and last-victim team from the
2026 tome channels (suffix .2026-09-18,2026-10-08), then reports:
  * regime marker (round_announce_match_start present) counts
  * 4-player (2v2) and 10-player matches (distinct players in player_info)
  * last-round-half announcement labels (7 = 2v2 halftime after 8; 11 = 5v5 MR12 halftime after 12)
  * freeze-end boundary classes and label-level round_end vs score-step truth (regime A and B)
  * bomb and last-victim anchors at the first post-halftime label for both formats
Reads G: only. Projected columns. Scratchpad output only.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_07_2026.py
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUF = ".2026-09-18,2026-10-08"


def scan(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    if not fs:
        raise FileNotFoundError(ch)
    return pl.concat([pl.scan_parquet(f).select(cols) for f in fs], how="vertical_relaxed").collect()


hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))[0]
H = pl.read_parquet(hf, columns=["key", "map_name", "match_type"]).rename({"key": "match_key"})
print("2026 header rows:", H.height, " match_key sample:", H["match_key"][0])
print("2026 match_type:", H.group_by("match_type").len().to_dicts())

RE = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
RE = (RE.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("winner_team_code").first().alias("end_winner"),
    pl.col("player_count").first().alias("end_player_count"),
    pl.len().alias("end_nrows")))
print("2026 round_end (match,round) pairs:", RE.height, " rows:", RE.select(pl.col("end_nrows").sum()).item())

RS = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
print("2026 round_state rows:", RS.height)
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("fe_t"),
                                            pl.col("ct_score").first().alias("fe_ct")))
HALF = (RS.filter(pl.col("phase") == "halftime").group_by(["match_key", "round"]).agg(pl.len().alias("n_ht")))
ANN = (RS.filter(pl.col("event_type").is_in(["round_announce_match_start", "round_announce_last_round_half"]))
       .group_by(["match_key", "event_type"]).agg(pl.col("round").sort().alias("labels"), pl.len().alias("n")))
ann_start = ANN.filter(pl.col("event_type") == "round_announce_match_start").select(
    ["match_key", pl.col("n").alias("n_match_start")])
ann_half = ANN.filter(pl.col("event_type") == "round_announce_last_round_half").select(
    ["match_key", pl.col("labels").alias("half_labels")])

BO = scan("bomb_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
BOMB = BO.group_by(["match_key", "round"]).agg(
    pl.col("event_type").filter(pl.col("event_type") == "bomb_defused").len().alias("n_defused"),
    pl.col("event_type").filter(pl.col("event_type") == "bomb_exploded").len().alias("n_exploded"))

DE = scan("player_death", ["match_key", "round", "tick", "player_team_code"]).sort(["match_key", "round", "tick"])
DEATH = DE.group_by(["match_key", "round"]).agg(pl.col("player_team_code").last().alias("last_victim_team"))

PI = scan("player_info", ["match_key", "player_id"])
NPL = PI.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n_players"))

R = (RE.join(FE, on=["match_key", "round"], how="full", coalesce=True)
     .join(BOMB, on=["match_key", "round"], how="left")
     .join(DEATH, on=["match_key", "round"], how="left")
     .join(HALF, on=["match_key", "round"], how="left")
     .with_columns(pl.col("n_defused").fill_null(0), pl.col("n_exploded").fill_null(0)))
M = (H.join(NPL, on="match_key", how="left").join(ann_start, on="match_key", how="left")
     .join(ann_half, on="match_key", how="left")
     .with_columns(regime=pl.when(pl.col("n_match_start").fill_null(0) > 0).then(pl.lit("A")).otherwise(pl.lit("B"))))
print("2026 matches:", M.height, " regime A:", int((M["regime"] == "A").sum()),
      " regime B:", int((M["regime"] == "B").sum()))
print("2026 n_players distribution:", M.group_by("n_players").len().sort("n_players").to_dicts())
print("2026 last-round-half label distribution (top):")
print(M.with_columns(hl=pl.col("half_labels").cast(pl.List(pl.String)).list.join(",")).group_by(["hl"]).len().sort("len", descending=True).head(12))
print("2026 round_end player_count distribution (top):")
print(RE.group_by("end_player_count").len().sort("len", descending=True).head(8))

fe = defaultdict(dict)
rend = defaultdict(dict)
for r in R.iter_rows(named=True):
    if r["fe_t"] is not None:
        fe[r["match_key"]][r["round"]] = (r["fe_t"], r["fe_ct"])
    if r["end_winner"] is not None:
        rend[r["match_key"]][r["round"]] = r


def bound(mk):
    out = {}
    F = fe[mk]
    for r in sorted(F):
        if r + 1 not in F:
            continue
        t0, c0 = F[r]
        t1, c1 = F[r + 1]
        dt, dc = t1 - t0, c1 - c0
        s = (dt + dc) == 1
        ns = s and dt in (0, 1) and dc in (0, 1)
        sw = s and (t1 - c0) in (0, 1) and (c1 - t0) in (0, 1)
        if ns and sw:
            out[r] = ("ambig", None)
        elif ns:
            out[r] = ("noswap", 2 if dt == 1 else 3)
        elif sw:
            out[r] = ("swap", 2 if (c1 - t0) == 1 else 3)
        else:
            out[r] = ("bad", None)
    return out


mmeta = {r["match_key"]: r for r in M.to_dicts()}
BND = {mk: bound(mk) for mk in mmeta}
print("\n2026 boundary class counts:", dict(Counter(c for mk in BND for (c, _) in BND[mk].values())))
bad = sum(1 for mk in BND if any(c == "bad" for (c, _) in BND[mk].values()))
print("2026 matches with >=1 bad boundary (sum or fit failure):", bad, "of", len(BND))
swap_any = {mk for mk in BND if any(c == "swap" for (c, _) in BND[mk].values())}
print("2026 matches with any freeze-end swap:", len(swap_any))
sw_at = Counter()
for mk in swap_any:
    for r, (c, _) in BND[mk].items():
        if c == "swap":
            sw_at[(mmeta[mk]["regime"], mmeta[mk]["n_players"], r)] += 1
print("2026 swap boundaries (regime, n_players, boundary label): ", dict(sorted(sw_at.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2]))))

print("\n2026 label-level round_end vs score-step truth (usable), by regime")
tab = defaultdict(lambda: [0, 0])
for mk in BND:
    for r, (c, truth) in BND[mk].items():
        if c not in ("noswap", "swap") or r not in rend[mk]:
            continue
        key = (mmeta[mk]["regime"], r)
        tab[key][0] += 1
        if rend[mk][r]["end_winner"] != truth:
            tab[key][1] += 1
for k in sorted(tab):
    if k[1] <= 30:
        print("  ", k, "usable", tab[k][0], "inverted", tab[k][1])

print("\n2026 bomb anchor: round_end vs bomb outcome (all rounds with a bomb outcome), by regime")
bc = Counter()
for mk in mmeta:
    for r, row in []:
        pass
for row in R.filter((pl.col("n_defused") + pl.col("n_exploded")) > 0).iter_rows(named=True):
    mk = row["match_key"]
    if row["end_winner"] is None:
        continue
    truth = 3 if (row["n_defused"] > 0 and row["n_exploded"] == 0) else (2 if row["n_exploded"] > 0 and row["n_defused"] == 0 else None)
    if truth is None:
        continue
    rg = mmeta[mk]["regime"]
    bc[(rg, "all_rounds", "rows")] += 1
    if row["end_winner"] != truth:
        bc[(rg, "all_rounds", "round_end_vs_bomb_mismatch")] += 1
    if row["round"] in (9, 13):
        bc[(rg, f"label{row['round']}", "rows")] += 1
        if row["end_winner"] != truth:
            bc[(rg, f"label{row['round']}", "mismatch")] += 1
for k in sorted(bc):
    print("  ", k, bc[k])

print("\n2026 first post-halftime anchors (5v5 MR12: label 13; 2v2: label 9): round_end vs truth", flush=True)
for rg in ["A", "B"]:
    for label, cond in [("5v5 label13", lambda m: m["n_players"] == 10 and m["half_labels"] is not None and 11 in (m["half_labels"] or [])),
                        ("2v2 label9", lambda m: m["n_players"] == 4 and m["half_labels"] is not None and 7 in (m["half_labels"] or []))]:
        lab = 13 if label.startswith("5v5") else 9
        mks = [mk for mk in mmeta if mmeta[mk]["regime"] == rg and cond(mmeta[mk])]
        usable = inv = bombs = bombmis = 0
        for mk in mks:
            b = BND[mk].get(lab)
            if b and b[0] in ("noswap", "swap") and lab in rend[mk]:
                usable += 1
                if rend[mk][lab]["end_winner"] != b[1]:
                    inv += 1
            row = rend[mk].get(lab)
            if row is not None:
                rr = R.filter((pl.col("match_key") == mk) & (pl.col("round") == lab)).row(0, named=True)
                if rr["n_defused"] + rr["n_exploded"] > 0:
                    bombs += 1
                    truth = 3 if rr["n_defused"] > 0 and rr["n_exploded"] == 0 else 2
                    if rr["end_winner"] != truth:
                        bombmis += 1
        print(f"  regime {rg} {label}: matches {len(mks)} usable {usable} inverted {inv} | bomb rows {bombs} round_end!=bomb {bombmis}")

print("\nDONE 2026 check")
