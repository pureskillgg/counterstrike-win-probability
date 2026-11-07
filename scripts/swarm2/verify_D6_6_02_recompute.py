"""verify D6-6, step 02: independent recompute of first-post-swap score-vs-round_end disagreement
by match day AND by header build_num (2024 tome). Also 2026 control and bomb-based mismatch by build.

Definitions (ASSUMPTIONS marked):
  W1    score winner of round r = change in (t_score, ct_score) between round_freeze_end(r) and round_freeze_end(r+1).
        'no' label: sides kept; 'swap' label: sides exchanged; exactly one side must gain one point (D6 section 11 rule).
  PI    first post-swap round from player_info orientation (D6 method, reimplemented in polars). r_post = s1 + 1.
  SC    first post-swap round from score labels: first r whose label (r -> r+1) is 'swap'; r_post = r + 1.
  BUILD header.build_num per match (2024 header is one parquet; 2026 header read the same way).
  ASSUMPTION  duplicate freeze_end and round_end keys keep the first row (as in D6); duplicate counts are printed.
  ASSUMPTION  bomb side = 3 (CT) if bomb_defused, else 2 (T) (as in scripts/swarm/e2_side_check_r13.py); player_count == 20.
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_02_recompute.py
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
B26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
T, CT = 2, 3


def fl(base, ch):
    if ch == "header":
        return sorted(glob.glob(str(base / "header" / "dataframe_*")))
    fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


def rd(base, ch, cols):
    files = fl(base, ch)
    have = set(pl.read_parquet_schema(files[0]).keys())
    use = [c for c in cols if c in have]
    return pl.concat([pl.read_parquet(f, columns=use) for f in files], how="vertical_relaxed")


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


def load_header(base, extra):
    h = rd(base, "header", ["key", "match_date", "build_num", "server_name", "map_name"] + extra)
    h = h.unique(["key"], keep="first", maintain_order=True)
    return h


def load_scores(base):
    rs = rd(base, "round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
    rs = rs.filter(pl.col("event_type") == "round_freeze_end")
    n_all = rs.height
    rs = rs.unique(["match_key", "round"], keep="first", maintain_order=True)
    return rs, n_all - rs.height


def load_round_end(base):
    re_ = rd(base, "round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "player_count", "tick"])
    dup = re_.height - re_.unique(["match_key", "round"]).height
    re_ = re_.unique(["match_key", "round"], keep="first", maintain_order=True)
    return re_, dup


def pi_bounds(base, match_keys):
    """D6 orientation method in polars. Returns {match: [completed rounds s with orientation change s -> s+1]}."""
    pi = rd(base, "player_info", ["match_key", "round", "player_id", "team_code"])
    pi = pi.unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
    cnt = pi.group_by(["match_key", "round"]).len().rename({"len": "n"})
    minr = cnt.group_by("match_key").agg(pl.col("round").min().alias("minr"))
    refr = cnt.filter(pl.col("n") >= 8).group_by("match_key").agg(pl.col("round").min().alias("refr"))
    ref = (minr.join(refr, on="match_key", how="left")
           .with_columns(pl.coalesce(["refr", "minr"]).alias("ref"))
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
    bounds = {m: [] for m in match_keys}
    for m, s in seqs.items():
        bounds[m] = [a[0] for a, b in zip(s, s[1:]) if a[1] != b[1] and b[0] == a[0] + 1]
    return bounds


def analyse(tag, base, extra_hdr):
    H = load_header(base, extra_hdr)
    hmap = {r["key"]: r for r in H.iter_rows(named=True)}
    RS, dupfe = load_scores(base)
    RE, dupre = load_round_end(base)
    re_w = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
    scm_all = collections.defaultdict(dict)
    for m, r, t, c in RS.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
        scm_all[m][r] = (t, c)
    PIB = pi_bounds(base, list(scm_all.keys()))
    print(f"[{tag}] headers={H.height} matches_with_freeze_end={len(scm_all)} "
          f"dup_freeze_end_dropped={dupfe} dup_round_end_dropped={dupre} matches_with_pi_bounds={len(PIB)}")

    per_round, per_post = [], []
    for m, scm in scm_all.items():
        day = day_of(m)
        build = hmap[m]["build_num"] if m in hmap else None
        labs = {}
        for r in scm:
            if (r + 1) in scm:
                t0, c0 = scm[r]
                t1, c1 = scm[r + 1]
                lab = label(t0, c0, t1, c1)
                labs[r] = (lab, winner(t0, c0, t1, c1, lab))

        def st(r):
            if r not in labs or labs[r][1] is None:
                return "unscorable"
            w_re = re_w.get((m, r))
            if w_re is None:
                return "unscorable"
            return "agree" if labs[r][1] == w_re else "disagree"

        for r in labs:
            per_round.append((tag, m, day, build, r, st(r)))
        b = PIB.get(m, [])
        s1_pi = b[0] if b else None
        s1_sc = next((r for r in sorted(labs) if labs[r][0] == "swap"), None)
        if s1_pi is not None:
            per_post.append((tag, m, day, build, "pi", s1_pi + 1, st(s1_pi + 1)))
        if s1_sc is not None:
            per_post.append((tag, m, day, build, "sc", s1_sc + 1, st(s1_sc + 1)))
    PR = pl.DataFrame(per_round, schema=["tag", "match", "day", "build", "round", "status"], orient="row")
    PP = pl.DataFrame(per_post, schema=["tag", "match", "day", "build", "kind", "post_round", "status"], orient="row")
    return H, RE, PR, PP


def tally(df, keys):
    return (df.group_by(keys)
            .agg(pl.len().alias("n_rows"),
                 (pl.col("status") == "agree").sum().alias("agree"),
                 (pl.col("status") == "disagree").sum().alias("disagree"),
                 (pl.col("status") == "unscorable").sum().alias("unscorable"))
            .with_columns((pl.col("disagree") / (pl.col("agree") + pl.col("disagree"))).round(3).alias("rate_disagree"))
            .sort(keys))


# ====================================================================== 2024
H24, RE24, PR24, PP24 = analyse("2024", B24, [])
H26, RE26, PR26, PP26 = analyse("2026", B26, ["max_rounds"])

pp24_pi = PP24.filter(pl.col("kind") == "pi")
pp24_sc = PP24.filter(pl.col("kind") == "sc")
print(f"\n2024 PI-post matches: {pp24_pi.height}; SC-post matches: {pp24_sc.height}")

print("\n== T1. D6 parity: PI first-post-swap, by day (2024). Compare to D6 section 17 (day 12: 437/389/151, 13: 76/96/31 ...) ==")
t1 = tally(pp24_pi, ["day"])
print(t1)
print("days 01-07 combined (PI-post):", pp24_pi.filter(pl.col("day") <= "07").group_by("status").len().sort("status").rows())
print("days 12-25 combined (PI-post):", pp24_pi.filter(pl.col("day") >= "12").group_by("status").len().sort("status").rows())

print("\n== T2. PI-post by day AND build_num (2024) ==")
t2 = tally(pp24_pi, ["day", "build"])
print(t2)

print("\n== T3. PI-post by build_num only (2024, all days) ==")
print(tally(pp24_pi, ["build"]))

print("\n== T4. PI-post by build_num, days 12-25 only ==")
print(tally(pp24_pi.filter(pl.col("day") >= "12"), ["build"]))

print("\n== T5. SC-post (score-label first swap, independent of player_info) by build (2024) and by day ==")
print(tally(pp24_sc, ["build"]))
print(tally(pp24_sc, ["day"]))

print("\n== T6. Every score check (all rounds, 2024): disagreements by day-group, build and round (count of disagree > 0 only) ==")
PR24 = PR24.with_columns(pl.when(pl.col("day") <= "07").then(pl.lit("d01-07")).when(pl.col("day") >= "12").then(pl.lit("d12-25")).otherwise(pl.lit("other")).alias("dgrp"))
dis = PR24.filter(pl.col("status") == "disagree")
print("disagree rows by dgrp x build:")
print(dis.group_by(["dgrp", "build"]).len().sort(["dgrp", "build"]))
print("disagree rows by round (top 20):")
print(dis.group_by("round").len().sort("len", descending=True).head(20))
print("rows by dgrp x build (all statuses):")
print(PR24.group_by(["dgrp", "build", "status"]).len().sort(["dgrp", "build", "status"]))

print("\n== T7. Bomb-based check (e2 definition, player_count==20), 2024: mismatch by build and round group ==")
BS = rd(B24, "bomb_state", ["match_key", "round", "event_type"])
bo = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
hk = H24.select(["key", "build_num", "match_date"]).rename({"key": "match_key", "build_num": "build"})
bom = (bo.join(RE24.filter(pl.col("player_count") == 20), on=["match_key", "round"], how="inner")
       .join(hk, on="match_key", how="left")
       .with_columns(
           pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bomb_side"))
       .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"))
       .with_columns(pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day"))
       .with_columns(pl.when(pl.col("day") <= "07").then(pl.lit("d01-07")).when(pl.col("day") >= "12").then(pl.lit("d12-25")).otherwise(pl.lit("other")).alias("dgrp")))
print("bomb rows (pc==20) total:", bom.height, "mismatch:", int(bom["mismatch"].sum()))
print(bom.filter(pl.col("round") == 13).group_by(["dgrp", "build"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort(["dgrp", "build"]))
print(bom.group_by(["dgrp", "build"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort(["dgrp", "build"]))
print("bomb mismatch by round, all builds (rounds with any mismatch):")
print(bom.group_by("round").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).filter(pl.col("mm") > 0).sort("round"))

print("\n== T8. 2026 control: all-round score vs round_end, and PI-post ==")
PR26 = PR26.with_columns(pl.lit("2026").alias("dgrp"))
print("2026 all-round statuses:", PR26.group_by("status").len().sort("status").rows())
print("2026 disagreements by round:", PR26.filter(pl.col("status") == "disagree").group_by("round").len().sort("round").rows())
print("2026 PI-post statuses:", PP26.filter(pl.col("kind") == "pi").group_by("status").len().sort("status").rows())
print("2026 SC-post statuses:", PP26.filter(pl.col("kind") == "sc").group_by("status").len().sort("status").rows())

print("\n== T9. Build vs server and map (2024, matches) ==")
print(H24.group_by(["build_num", "server_name"]).len().sort(["build_num", "len"], descending=[False, True]).head(12))
print(H24.with_columns(pl.col("key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day")).filter(pl.col("day") >= "12")
      .group_by(["build_num", "map_name"]).len().sort(["build_num", "len"], descending=[False, True]))

print("\n== T10. Build ordering in time within days 12-25 (match_date sorted) ==")
ord_ = (H24.with_columns(pl.col("key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day"))
        .filter(pl.col("day") >= "12").sort("match_date"))
blds = ord_["build_num"].to_list()
first_9961 = next(i for i, b in enumerate(blds) if b == 9961)
print(f"matches in days 12-25: {len(blds)}; first build 9961 at position {first_9961} (date {ord_['match_date'][first_9961]});"
      f" build 9842 matches after that position: {sum(1 for b in blds[first_9961:] if b == 9842)}")
print("first 10 and last 10 builds in time order:", blds[:10], blds[-10:])
print("first/last match_date for each build (days 12-25 folder subset):",
      ord_.group_by("build_num").agg(pl.col("match_date").min().alias("min_md"),
                                     pl.col("match_date").max().alias("max_md")).sort("build_num").rows())
