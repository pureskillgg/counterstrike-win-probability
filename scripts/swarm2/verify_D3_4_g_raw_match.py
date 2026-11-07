"""verify_D3_4_g_raw_match.py: spot-check tome rows against the raw tail-copy Parquet for three matches (read-only).

Matches: one (12,) match on day 14, one (8,) match on day 21, one (12,) match on day 03 (days 01..07 window).
For each: compares tome round_end (winner_team_code, win_reason_code) and round_state freeze scores with the
raw files E:\\csds_final\\csds\\csds_tail_recovered_20261011\\csds\\2024\\02\\<DD>\\<match_id>\\{round_end,round_state}.
ASSUMPTION A22: the raw files are the source the tome was built from (the claim names this tail copy).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_4_g_raw_match.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
RAW = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")


def scan(ch, cols):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.scan_parquet(files).select(cols).collect()


RS = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"], maintain_order=True)
      .agg(pl.col("t_score").first(), pl.col("ct_score").first()))
live = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
RE = scan("round_end", ["match_key", "round", "winner_team_code", "win_reason_code"]).unique(
    ["match_key", "round"], keep="first", maintain_order=True)
win = {(m, r): (w, rc) for m, r, w, rc in RE.select(["match_key", "round", "winner_team_code", "win_reason_code"]).iter_rows()}
by_m = collections.defaultdict(list)
for (m, r) in live:
    by_m[m].append(r)


def clean(dt, dc):
    return (dc == 1 and dt == 0) or (dt == 1 and dc == 0)


def swaps_of(m):
    out = []
    for r in sorted(by_m[m]):
        if (m, r + 1) not in live:
            continue
        t1, c1 = live[(m, r)]
        t2, c2 = live[(m, r + 1)]
        dt, dc = t2 - t1, c2 - c1
        if clean(dt, dc) or (dt, dc) == (0, 0):
            continue
        if dt * dc <= 0:
            out.append(r)
    return tuple(out)


def pick(day, sig):
    for m in sorted(by_m):
        if int(m.split("/")[3]) == day and swaps_of(m) == sig:
            return m
    return None


targets = [("day14 (12,)", pick(14, (12,))), ("day21 (8,)", pick(21, (8,))), ("day03 (12,)", pick(3, (12,)))]
for label, m in targets:
    if m is None:
        print(f"{label}: no match found")
        continue
    mid = m.split("/")[4]
    day = m.split("/")[3]
    rdir = RAW / day / mid
    print(f"\n== {label}: {m} ==")
    if not rdir.exists():
        print(f"  raw folder missing: {rdir}")
        continue
    raw_re = pl.read_parquet(rdir / "round_end", columns=["round", "winner_team_code", "win_reason_code"])
    raw_rs = pl.read_parquet(rdir / "round_state", columns=["round", "tick", "event_type", "t_score", "ct_score"])
    raw_fe = (raw_rs.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
              .group_by("round", maintain_order=True).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
    rawlive = {r: (t, c) for r, t, c in raw_fe.select(["round", "t_score", "ct_score"]).iter_rows()}
    rawwin = {r: (w, rc) for r, w, rc in raw_re.unique(["round"], keep="first", maintain_order=True)
              .select(["round", "winner_team_code", "win_reason_code"]).iter_rows()}
    same_w = same_f = n = 0
    for r in sorted(set(rawwin) & {rr for (mm, rr) in win if mm == m}):
        n += 1
        same_w += int(rawwin[r] == win[(m, r)])
    for r in sorted(rawlive):
        if (m, r) in live:
            same_f += int(rawlive[r] == live[(m, r)])
    print(f"  rounds compared (round_end): {n}; identical winner+reason to tome: {same_w}")
    print(f"  freeze score pairs compared: {sum(1 for r in rawlive if (m, r) in live)}; identical to tome: {same_f}")
    for r in range(11, 15):
        print(f"    r={r}: tome W/reason={win.get((m, r))} freeze={live.get((m, r))} | raw W/reason={rawwin.get(r)} freeze={rawlive.get(r)}")
