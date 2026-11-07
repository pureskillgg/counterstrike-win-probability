"""verify D6-6, step 06: are the 7 build-9961 round-13 bomb rows without mismatch (and the 434 mismatching ones) first-post-swap
rounds by the D6 player_info definition? Also: is a per-match timestamp cutoff (2024-02-07T01:01Z) equivalent to build_num?
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_06_exceptions.py
"""
import collections
import glob
import re
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
B24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T, CT = 2, 3


def fl(base, ch):
    if ch == "header":
        return sorted(glob.glob(str(base / "header" / "dataframe_*")))
    return sorted(glob.glob(str(sorted(base.glob(f"basic_{ch}.*"))[0] / "dataframe_*")))


def rd(base, ch, cols):
    files = fl(base, ch)
    have = set(pl.read_parquet_schema(files[0]).keys())
    return pl.concat([pl.read_parquet(f, columns=[c for c in cols if c in have]) for f in files], how="vertical_relaxed")


def label(t0, c0, t1, c1):
    no = (t1 - t0, c1 - c0) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    if no and sw:
        return "both"
    if no:
        return "no"
    if sw:
        return "swap"
    return "none"


def winner(t0, c0, t1, c1, lab):
    if lab == "no":
        return T if (t1 - t0, c1 - c0) == (1, 0) else CT
    if lab == "swap":
        return T if (t1 - c0, c1 - t0) == (0, 1) else CT
    return None


def day_of(m):
    g = re.search(r"csds/20\d\d/\d\d/(\d\d)/", m)
    return g.group(1) if g else "??"


def pi_bounds(base, keys):
    pi = rd(base, "player_info", ["match_key", "round", "player_id", "team_code"])
    pi = pi.unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
    cnt = pi.group_by(["match_key", "round"]).len().rename({"len": "n"})
    minr = cnt.group_by("match_key").agg(pl.col("round").min().alias("minr"))
    refr = cnt.filter(pl.col("n") >= 8).group_by("match_key").agg(pl.col("round").min().alias("refr"))
    ref = (minr.join(refr, on="match_key", how="left").with_columns(pl.coalesce(["refr", "minr"]).alias("ref"))
           .select(["match_key", "ref"]))
    p0 = (pi.join(ref, on="match_key").filter(pl.col("round") == pl.col("ref"))
          .select(["match_key", "player_id", pl.col("team_code").alias("t0")]))
    j = pi.join(p0, on=["match_key", "player_id"], how="inner")
    fr = (j.group_by(["match_key", "round"])
          .agg(pl.len().alias("common"), (pl.col("team_code") != pl.col("t0")).mean().alias("frac"))
          .filter(pl.col("common") >= 4)
          .with_columns(pl.when(pl.col("frac") > 0.5).then(1).otherwise(0).alias("orient"))
          .sort(["match_key", "round"]))
    seqs = collections.defaultdict(list)
    for m, r, o in fr.select(["match_key", "round", "orient"]).iter_rows():
        seqs[m].append((r, o))
    out = {m: [] for m in keys}
    for m, s in seqs.items():
        out[m] = [a[0] for a, b in zip(s, s[1:]) if a[1] != b[1] and b[0] == a[0] + 1]
    return out


H = rd(B24, "header", ["key", "match_date", "build_num", "map_name"]).unique(["key"], keep="first", maintain_order=True)
hb = {r["key"]: (r["build_num"], r["match_date"], r["map_name"]) for r in H.iter_rows(named=True)}
RS = rd(B24, "round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
RS = RS.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first", maintain_order=True)
scm_all = collections.defaultdict(dict)
for m, r, t, c in RS.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    scm_all[m][r] = (t, c)
RE = rd(B24, "round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    ["match_key", "round"], keep="first", maintain_order=True)
re_w = {(m, r): (w, pc) for m, r, w, pc in RE.select(["match_key", "round", "winner_team_code", "player_count"]).iter_rows()}
PIB = pi_bounds(B24, list(scm_all.keys()))

BS = rd(B24, "bomb_state", ["match_key", "round", "event_type"])
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
rows = []
for m, r, evs in outc.select(["match_key", "round", "evs"]).iter_rows():
    if (m, r) not in re_w or re_w[(m, r)][1] != 20 or r != 13:
        continue
    b_side = CT if "bomb_defused" in evs else T
    s1 = PIB.get(m, [None])[0] if PIB.get(m) else None
    post = (s1 + 1) if s1 is not None else None
    sc_r = None
    if 13 in scm_all.get(m, {}) and 14 in scm_all[m]:
        t0, c0 = scm_all[m][13]
        t1, c1 = scm_all[m][14]
        sc_r = winner(t0, c0, t1, c1, label(t0, c0, t1, c1))
    build, mdate, mp = hb.get(m, (None, None, None))
    rows.append((m, day_of(m), build, mdate, mp, b_side, re_w[(m, r)][0], sc_r, s1, post,
                 post == 13, b_side != re_w[(m, r)][0]))
D = pl.DataFrame(rows, schema=["match", "keyday", "build", "match_date", "map", "bomb_side", "round_end_w", "score_w",
                               "pi_s1", "pi_post_round", "r13_is_pi_post", "mismatch"], orient="row")
print(f"round-13 bomb rows (pc==20): {D.height}")
print(D.group_by(["build", "mismatch", "r13_is_pi_post"]).len().sort(["build", "mismatch", "r13_is_pi_post"]))

print("\n== F1. Build-9961 round-13 bomb rows WITHOUT mismatch ==")
ex = D.filter((pl.col("build") == 9961) & (~pl.col("mismatch")))
print(ex.select(["keyday", "match_date", "map", "bomb_side", "round_end_w", "score_w", "pi_s1", "pi_post_round", "r13_is_pi_post"]))

print("\n== F2. Build-9961 round-13 bomb rows WITH mismatch: score winner availability and agreement with bomb ==")
mm = D.filter((pl.col("build") == 9961) & pl.col("mismatch"))
print("score available:", mm.filter(pl.col("score_w").is_not_null()).height, "of", mm.height,
      "| score == bomb:", mm.filter(pl.col("score_w") == pl.col("bomb_side")).height,
      "| score == round_end:", mm.filter(pl.col("score_w") == pl.col("round_end_w")).height)

print("\n== F3. Calendar cutoff equivalence: matches on/after 2024-02-07T01:01Z by build (all 2024 matches) ==")
cut = pl.DataFrame([(k, v[0], v[1]) for k, v in hb.items()], schema=["key", "build", "mdate"], orient="row")
print(cut.with_columns((pl.col("mdate") >= "2024-02-07T01:01:18").alias("after_cutoff"))
      .group_by(["build", "after_cutoff"]).len().sort(["build", "after_cutoff"]))
print("9842 rows on/after cutoff:", cut.filter((pl.col("build") == 9842) & (pl.col("mdate") >= "2024-02-07T01:01:18")).rows())
