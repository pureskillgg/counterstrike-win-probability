"""verify D6-6, step 05: calendar match_date vs folder day, per-map check inside build 9961, FACEIT rows,
and the seven build-9961 round-13 bomb rows that do not mismatch. Reuses the D6 definitions (PI first post-swap, W1 score winner).
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_05_calendar.py
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


def day_of(m):
    g = re.search(r"csds/20\d\d/\d\d/(\d\d)/", m)
    return g.group(1) if g else "??"


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


H = rd(B24, "header", ["key", "match_date", "build_num", "server_name", "map_name", "match_id"])
H = H.unique(["key"], keep="first", maintain_order=True)
hmap = {r["key"]: r for r in H.iter_rows(named=True)}

RS = rd(B24, "round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
RS = RS.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first", maintain_order=True)
scm_all = collections.defaultdict(dict)
for m, r, t, c in RS.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    scm_all[m][r] = (t, c)

RE = rd(B24, "round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "player_count"])
RE = RE.unique(["match_key", "round"], keep="first", maintain_order=True)
re_w = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
PIB = pi_bounds(B24, list(scm_all.keys()))

pp = []
for m, scm in scm_all.items():
    labs = {}
    for r in scm:
        if r + 1 in scm:
            t0, c0 = scm[r]
            t1, c1 = scm[r + 1]
            lab = label(t0, c0, t1, c1)
            labs[r] = winner(t0, c0, t1, c1, lab)

    def st(r):
        if r not in labs or labs[r] is None:
            return "unscorable"
        w_re = re_w.get((m, r))
        if w_re is None:
            return "unscorable"
        return "agree" if labs[r] == w_re else "disagree"

    b = PIB.get(m, [])
    if b:
        rp = b[0] + 1
        h = hmap.get(m, {})
        pp.append((m, day_of(m), h.get("match_date", ""), h.get("build_num"), h.get("map_name"), h.get("server_name"),
                   rp, st(rp)))
P = pl.DataFrame(pp, schema=["match", "keyday", "match_date", "build", "map", "server", "post_round", "status"], orient="row")
P = P.with_columns(pl.col("match_date").str.slice(0, 10).alias("mdate"))
print(f"PI-post matches: {P.height}; statuses: {P.group_by('status').len().sort('status').rows()}")

print("\n== E1. PI-post statuses by CALENDAR match_date (UTC date from header) and build ==")
cal = (P.group_by(["mdate", "build", "status"]).len().sort(["mdate", "build", "status"]))
print(cal.filter(pl.col("mdate") >= "2024-02-05"))

print("\n== E2. PI-post by calendar match date: first date with a disagreement, and days with any agree in 9961 ==")
dis_dates = P.filter(pl.col("status") == "disagree")["mdate"].unique().sort().to_list()
print("calendar dates with >=1 disagreement:", dis_dates[:3], "...", dis_dates[-3:], f"(n dates={len(dis_dates)})")
print("disagreements with calendar date < 2024-02-12:",
      P.filter((pl.col("status") == "disagree") & (pl.col("mdate") < "2024-02-12")).height,
      "| by date:", P.filter((pl.col("status") == "disagree") & (pl.col("mdate") < "2024-02-12")).group_by("mdate").len().sort("mdate").rows())
print("matches dated 2024-02-07 by build and status:",
      P.filter(pl.col("mdate") == "2024-02-07").group_by(["build", "status"]).len().sort(["build", "status"]).rows())

print("\n== E3. PI-post in build 9961 by map (per-map consistency) ==")
print(P.filter(pl.col("build") == 9961).group_by(["map", "status"]).len().sort(["map", "status"]))
print("== E3b. PI-post in build 9842 by map ==")
print(P.filter(pl.col("build") == 9842).group_by(["map", "status"]).len().sort(["map", "status"]))

print("\n== E4. FACEIT server rows and nanosecond-format match_date rows: PI-post status ==")
print(P.filter(pl.col("server").str.contains("FACEIT")).select(["match", "keyday", "match_date", "build", "map", "post_round", "status"]))
print(P.filter(~pl.col("match_date").str.contains(r"\.000Z$")).select(["match", "keyday", "match_date", "build", "map", "post_round", "status"]))

print("\n== E5. Round-13 bomb rows (player_count==20): share of build 9961 and mismatch, by folder day (e2 definition) ==")
BS = rd(B24, "bomb_state", ["match_key", "round", "event_type", "player_team_code"])
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
bomb = (outc.join(RE.filter(pl.col("player_count") == 20), on=["match_key", "round"], how="inner")
        .join(H.select(["key", "build_num", "match_date"]).rename({"key": "match_key", "build_num": "build"}), on="match_key", how="left")
        .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bomb_side"))
        .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
                      pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("keyday")))
r13 = bomb.filter(pl.col("round") == 13)
print(r13.group_by("keyday").agg(pl.len().alias("n"), (pl.col("build") == 9961).sum().alias("n_9961"),
                                 pl.col("mismatch").sum().alias("mm"),
                                 (pl.col("mismatch") & (pl.col("build") == 9961)).sum().alias("mm_9961"))
      .sort("keyday"))
print("r13 bomb rows: total", r13.height, "| build 9961:", r13.filter(pl.col("build") == 9961).height,
      "| mismatches in 9961:", r13.filter((pl.col("build") == 9961) & pl.col("mismatch")).height,
      "| mismatches in 9842:", r13.filter((pl.col("build") == 9842) & pl.col("mismatch")).height)
ex = r13.filter((pl.col("build") == 9961) & (~pl.col("mismatch")))
print("build-9961 round-13 bomb rows WITHOUT mismatch (exceptions):")
for row in ex.iter_rows(named=True):
    m = row["match_key"]
    scw = None
    if m in scm_all and 13 in scm_all[m] and 14 in scm_all[m]:
        t0, c0 = scm_all[m][13]
        t1, c1 = scm_all[m][14]
        scw = winner(t0, c0, t1, c1, label(t0, c0, t1, c1))
    print(f"  keyday={row['keyday']} match_date={row['match_date']} evs={row['evs']} bomb_side={row['bomb_side']} "
          f"round_end_winner={row['winner_team_code']} score_winner={scw} map={hmap.get(m, {}).get('map_name')}")
