"""D6-6 verifier step 02b: independent recompute of the D6 section-17 day table, the e2 round-13 table,
and the 2026 control, under two day definitions:
  (a) folder day  = DD in match_key csds/2024/02/DD/<uuid>  (the definition used by e2 and D6 section 17)
  (b) match day  = UTC calendar day of header match_date     (the wording 'dated 2024-02-12..25' in the claim)

Definitions copied from D6 (re-implemented here, not imported):
  - swap boundaries from basic_player_info team_code (detect_swaps rule, D6 section 1/2)
  - score winner from basic_round_state round_freeze_end t_score/ct_score (label 'no' or 'swap' (W1 rule))
  - round_end winner_team_code, first row per (match, round)
  - bomb side 3 if bomb_defused else 2, from basic_bomb_state (e2 rule)
  - first post-swap round r = s1 + 1, s1 = first detected swap boundary (D6 section 17)
ASSUMPTION A1: match day = first 10 characters of header match_date (UTC), i.e. no time-zone shift.
ASSUMPTION A2: unique(keep='first') with maintain_order=True; duplicate keys counted and reported.
Writes the long per-round table to the scratchpad (not to the repo).
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_02b_day_recompute.py
"""
import collections
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASES = {
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
}
T, CT = 2, 3


def ch_files(base, ch):
    if ch == "header":
        return sorted(glob.glob(str(base / "header" / "dataframe_*")))
    fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read(base, ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in ch_files(base, ch)], how="vertical_relaxed")


def detect_swaps(rounds_map):
    """D6 detect_swaps rule. Returns (ref, bounds, gap_bounds)."""
    rounds = sorted(rounds_map)
    ref = next((r for r in rounds if len(rounds_map[r]) >= 8), rounds[0])
    t0 = rounds_map[ref]
    seq = []
    for r in rounds:
        cur = rounds_map[r]
        common = [p for p in cur if p in t0]
        if len(common) < 4:
            continue
        frac = sum(1 for p in common if cur[p] != t0[p]) / len(common)
        seq.append((r, 1 if frac > 0.5 else 0))
    bounds, gap = [], []
    for a, b in zip(seq, seq[1:]):
        if a[1] != b[1]:
            (bounds if b[0] == a[0] + 1 else gap).append(a[0])
    return ref, bounds, gap


def score_winner(scm, r):
    """Round r winner from freeze_end scores at r and r+1 (W1). None if not scorable or ambiguous."""
    if r not in scm or (r + 1) not in scm:
        return None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    no = (t1 - t0, c1 - c0) in ((1, 0), (0, 1))
    sw = (t1 - c0, c1 - t0) in ((1, 0), (0, 1))
    if no and sw:
        return None
    if no:
        return T if (t1 - t0, c1 - c0) == (1, 0) else CT
    if sw:
        return T if (t1 - c0, c1 - t0) == (0, 1) else CT
    return None


def build(era):
    base = BASES[era]
    print(f"\n################ ERA {era} ################")
    hcols = ["key", "match_date", "map_name"] + (["max_rounds"] if era == "2026" else [])
    H = read(base, "header", hcols).unique(["key"], keep="first", maintain_order=True)
    hdr = {r["key"]: r for r in H.iter_rows(named=True)}

    pi = read(base, "player_info", ["match_key", "round", "player_id", "team_code"])
    n_pi_dup = pi.height - pi.unique(["match_key", "round", "player_id"], keep="first").height
    pi = pi.unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
    pmap = collections.defaultdict(lambda: collections.defaultdict(dict))
    for m, r, p, t in pi.iter_rows():
        pmap[m][r][p] = t
    det = {m: detect_swaps(rm) for m, rm in pmap.items()}
    print(f"player_info rows={pi.height} dup_keys_dropped={n_pi_dup} matches={len(pmap)} "
          f"with_bounds={sum(1 for d in det.values() if d[1])}")

    RE_raw = read(base, "round_end", ["match_key", "round", "winner_team_code", "player_count"])
    dup_re = RE_raw.height - RE_raw.unique(["match_key", "round"], keep="first").height
    RE = RE_raw.unique(["match_key", "round"], keep="first", maintain_order=True)
    print(f"round_end rows={RE_raw.height} dup_keys_dropped={dup_re}")

    RS_raw = read(base, "round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
    RS_raw = RS_raw.filter(pl.col("event_type") == "round_freeze_end")
    dup_rs = RS_raw.height - RS_raw.unique(["match_key", "round"], keep="first").height
    RS = RS_raw.unique(["match_key", "round"], keep="first", maintain_order=True)
    print(f"freeze_end rows={RS_raw.height} dup_keys_dropped={dup_rs}")

    BS = read(base, "bomb_state", ["match_key", "round", "event_type"])
    outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
            .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    bomb = {}
    for m, r, evs in outc.iter_rows():
        bomb[(m, r)] = CT if "bomb_defused" in evs else T

    scm_all = collections.defaultdict(dict)
    for m, r, t, c in RS.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
        scm_all[m][r] = (t, c)
    re_w = {}
    re_rounds = collections.defaultdict(set)
    for m, r, w, pc in RE.select(["match_key", "round", "winner_team_code", "player_count"]).iter_rows():
        re_w[(m, r)] = (w, pc)
        re_rounds[m].add(r)
    bomb_rounds = collections.defaultdict(set)
    for (m, r) in bomb:
        bomb_rounds[m].add(r)

    rows = []
    for m, h in hdr.items():
        key_day = m.split("/")[3]
        md = str(h["match_date"])[:10]
        mday = md[8:10] if md[:7] == "2024-02" else ("pre" if md < "2024-02-01" else "other")
        mr = int(h["max_rounds"]) if era == "2026" else None
        d = det.get(m)
        s1 = d[1][0] if (d and d[1]) else None
        scm = scm_all.get(m, {})
        rset = set(scm) | re_rounds.get(m, set()) | bomb_rounds.get(m, set())
        for r in sorted(rset):
            w_sc = score_winner(scm, r)
            ent = re_w.get((m, r))
            w_re = ent[0] if ent else None
            pc = ent[1] if ent else None
            bs = bomb.get((m, r))
            rows.append({
                "era": era, "match": m, "key_day": key_day, "md": md, "mday": mday, "max_rounds": mr,
                "s1": s1, "round": r, "w_sc": w_sc, "w_re": w_re, "pc": pc, "bomb": bs,
            })
    df = pl.DataFrame(rows, infer_schema_length=None)
    df = df.with_columns(
        (pl.col("s1").is_not_null() & (pl.col("round") == pl.col("s1") + 1)).alias("is_post"),
        (pl.col("w_sc").is_not_null() & pl.col("w_re").is_not_null()).alias("scorable"),
        (pl.col("w_sc").is_not_null() & pl.col("w_re").is_not_null() & (pl.col("w_sc") != pl.col("w_re"))).alias("sc_dis"),
        (pl.col("bomb").is_not_null() & pl.col("w_re").is_not_null() & (pl.col("bomb") != pl.col("w_re"))).alias("bomb_mm"),
    )
    return df


def show(title, t):
    print(f"\n-- {title}")
    print(t)


# ----------------------------------------------------------------------------------- 2024
df24 = build("2024")
df24 = df24.with_columns(
    pl.when(pl.col("key_day").cast(pl.Int32) <= 7).then(pl.lit("A 01-07")).otherwise(pl.lit("B 12-25")).alias("fgrp"),
)

# (1) e2 round-13 table, folder day and match day
r13 = df24.filter((pl.col("round") == 13) & (pl.col("pc") == 20) & pl.col("bomb").is_not_null())
print(f"\nround-13, pc==20, bomb outcome present: rows={r13.height} mismatches={int(r13['bomb_mm'].sum())}")
show("(1a) e2 round-13 bomb mismatch by FOLDER day (claim/e2 definition)",
     r13.group_by("key_day").agg(pl.len().alias("n"), pl.col("bomb_mm").sum().alias("mm"),
                                 (pl.col("bomb_mm").mean()).round(3).alias("rate")).sort("key_day"))
show("(1b) same, by MATCH day (header match_date)",
     r13.group_by("mday").agg(pl.len().alias("n"), pl.col("bomb_mm").sum().alias("mm"),
                              (pl.col("bomb_mm").mean()).round(3).alias("rate")).sort("mday"))

# (2) D6 section 17 recompute: first post-swap round, scorable, by folder day
post = df24.filter(pl.col("is_post") & pl.col("w_re").is_not_null())
post_sc = post.filter(pl.col("scorable"))
print(f"\nfirst-post-swap rounds with round_end row: {post.height}; scorable: {post_sc.height}; "
      f"disagree: {int(post_sc['sc_dis'].sum())}")
show("(2a) D6-17 recompute by FOLDER day (claim: 01-07 0/776; 12 389/826 ...)",
     post.group_by("key_day").agg(
         pl.col("scorable").sum().alias("scorable"),
         pl.col("sc_dis").sum().alias("disagree"),
         ((pl.col("sc_dis").sum()) / pl.col("scorable").sum()).round(3).alias("rate"),
         (pl.len() - pl.col("scorable").sum()).alias("unscorable")).sort("key_day"))
show("(2b) same first-post-swap rows by MATCH day",
     post.group_by("mday").agg(
         pl.col("scorable").sum().alias("scorable"),
         pl.col("sc_dis").sum().alias("disagree"),
         ((pl.col("sc_dis").sum()) / pl.col("scorable").sum()).round(3).alias("rate")).sort("mday"))
show("(2c) first-post-swap by FORMAT (s1=8 -> round 9 ; s1=12 -> round 13) and folder group",
     post.group_by(["s1", "fgrp"]).agg(
         pl.col("scorable").sum().alias("scorable"),
         pl.col("sc_dis").sum().alias("disagree"),
         ((pl.col("sc_dis").sum()) / pl.col("scorable").sum()).round(3).alias("rate")).sort(["s1", "fgrp"]))

# (3) round-13 score disagreement (score independent of bomb), by folder day and match day
r13s = df24.filter((pl.col("round") == 13) & pl.col("scorable"))
show("(3a) round-13 score-vs-round_end disagreement, all pc, by FOLDER day",
     r13s.group_by("key_day").agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis"),
                                   (pl.col("sc_dis").mean()).round(3).alias("rate")).sort("key_day"))
show("(3b) same by MATCH day",
     r13s.group_by("mday").agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis"),
                                (pl.col("sc_dis").mean()).round(3).alias("rate")).sort("mday"))

# (4) folder 12 split by match date
f12 = df24.filter((pl.col("key_day") == "12") & (pl.col("round") == 13) & pl.col("scorable"))
show("(4) folder 12, round 13 score disagreement, by match date (UTC)",
     f12.group_by("md").agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort("md"))
f12_pre = f12.filter(pl.col("md") < "2024-02-07")
f12_post = f12.filter(pl.col("md") >= "2024-02-07")
print(f"folder 12 with match_date < 2024-02-07: n={f12_pre.height} dis={int(f12_pre['sc_dis'].sum())} | "
      f"match_date >= 2024-02-07: n={f12_post.height} dis={int(f12_post['sc_dis'].sum())}")

# (5) cutoff search on match date vs folder day for round-13 score disagreement (scorable rows)
r13m = df24.filter((pl.col("round") == 13) & pl.col("scorable")).select(["md", "sc_dis", "key_day"])
best = []
for cut in sorted(set(r13m["md"].to_list())):
    lo = r13m.filter(pl.col("md") < cut)
    hi = r13m.filter(pl.col("md") >= cut)
    err = int(lo["sc_dis"].sum()) + int(hi.height - hi["sc_dis"].sum())
    best.append((err, cut, lo.height, int(lo["sc_dis"].sum()), hi.height, int(hi["sc_dis"].sum())))
best.sort()
print("\n(5) best single match-date cutoff for round-13 score disagreement (errors = dis below + agree above):")
for row in best[:4]:
    print("   err=%d cut=%s below n=%d dis=%d | at/above n=%d dis=%d" % row)
best_f = []
for cut in sorted(set(r13m["key_day"].to_list())):
    lo = r13m.filter(pl.col("key_day") < cut)
    hi = r13m.filter(pl.col("key_day") >= cut)
    err = int(lo["sc_dis"].sum()) + int(hi.height - hi["sc_dis"].sum())
    best_f.append((err, cut, lo.height, int(lo["sc_dis"].sum()), hi.height, int(hi["sc_dis"].sum())))
best_f.sort()
print("    best single FOLDER-day cutoff:")
for row in best_f[:3]:
    print("   err=%d cut=%s below n=%d dis=%d | at/above n=%d dis=%d" % row)

# (6) all-round score disagreements (2024) by round
sc_all = df24.filter(pl.col("scorable"))
show("(6) 2024 all-round score-vs-round_end disagreements by round, all days (dis>0 only)",
     sc_all.group_by("round").agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis"),
                                   (pl.col("sc_dis").mean()).round(3).alias("rate"))
     .filter(pl.col("dis") > 0).sort("dis", descending=True).head(14))

# ----------------------------------------------------------------------------------- 2026 control
df26 = build("2026")
post26 = df26.filter(pl.col("is_post") & pl.col("w_re").is_not_null())
post26_sc = post26.filter(pl.col("scorable"))
print(f"\n2026 first-post-swap rounds with round_end: {post26.height}; scorable: {post26_sc.height}; "
      f"disagree: {int(post26_sc['sc_dis'].sum())}")
show("(7a) 2026 first-post-swap by format (s1) and round",
     post26_sc.group_by(["s1", "round"]).agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort(["s1", "round"]))
allsc26 = df26.filter(pl.col("scorable"))
print(f"\n2026 ALL-round scorable (score winner and round_end both present): {allsc26.height}; "
      f"agree={allsc26.height - int(allsc26['sc_dis'].sum())}; disagree={int(allsc26['sc_dis'].sum())}")
print(f"2026 rows with score winner (any round_end): {df26.filter(pl.col('w_sc').is_not_null()).height}; "
      f"with round_end (any score): {df26.filter(pl.col('w_re').is_not_null()).height}")
r13_26 = df26.filter((pl.col("round") == 13) & (pl.col("pc") == 20) & pl.col("bomb").is_not_null())
print(f"2026 round-13 bomb rows (pc==20): {r13_26.height}; bomb-vs-round_end mismatches: {int(r13_26['bomb_mm'].sum())}")

scratch = Path(os.environ.get("VERIFY_SCRATCH", r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"))
scratch.mkdir(parents=True, exist_ok=True)
df24.write_parquet(scratch / "verify_D6_6_long_2024.parquet")
df26.write_parquet(scratch / "verify_D6_6_long_2026.parquet")
print(f"\nlong tables written to scratchpad: {scratch}")
