"""Probe for the 1 Hz state extraction (one match). Not the extractor.

Checks on one kept 2026 match, reading its player_status and player_vector raw files from E
(read-only, column-projected) and the tome channels from G:
  - tick coverage: are rows present at every tick, and at freeze_end + 64k?
  - player_id key: does raw player_status.player_id match tome player_info / player_death ids?
  - team_code nulls in raw player_vector; duplicate (tick, round, player_id) rows
  - alive count per side at snapshot ticks vs deaths so far (strict and inclusive tick rules)

Run from the project root:
  pixi run python -I -X utf8 scripts\\eda\\state_1hz_probe.py
"""

import collections
import importlib.util
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
_spec = importlib.util.spec_from_file_location("bakeoff", ROOT / "scripts" / "model" / "bakeoff.py")
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)
fv = bk.fv
T, CT, START = fv.T, fv.CT, fv.START
MK = "csds/2026/10/07/m_3d2a6755b8cc51e27d727b89/csds"
E_BASE = r"E:\csds_final\csds\2026\10\07\m_3d2a6755b8cc51e27d727b89"


def raw(ch, cols):
    return pl.read_parquet(os.path.join(E_BASE, ch), columns=cols)


def main():
    print("match", MK)
    RE = fv.read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).filter(
        pl.col("match_key") == MK)
    RS = fv.read("round_state", ["match_key", "round", "tick", "event_type"]).filter(
        (pl.col("match_key") == MK) & (pl.col("event_type") == "round_freeze_end"))
    DE = fv.read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"]).filter(
        pl.col("match_key") == MK)
    PI = fv.read("player_info", ["match_key", "round", "player_id", "team_code"]).filter(
        pl.col("match_key") == MK)
    print("rounds", RE.height, "freeze rows", RS.height, "deaths", DE.height, "player_info rows", PI.height)

    ps = raw("player_status", ["tick", "round", "player_id", "health", "armor"])
    pv = raw("player_vector", ["tick", "round", "player_id", "team_code"])
    print("player_status rows", ps.height, "player_vector rows", pv.height)
    print("player_vector team_code null rows:", pv.filter(pl.col("team_code").is_null()).height)
    dup_s = ps.height - ps.unique(["tick", "round", "player_id"]).height
    dup_v = pv.height - pv.unique(["tick", "round", "player_id"]).height
    print("duplicate (tick, round, player_id): status", dup_s, "vector", dup_v)

    rr = RE.select(["round", "tick"]).rename({"tick": "end"})
    fz = RS.select(["round", "tick"]).rename({"tick": "freeze"})
    rounds = rr.join(fz, on="round", how="inner").sort("round")
    cover = []
    for row in rounds.iter_rows(named=True):
        r, F, E = row["round"], row["freeze"], row["end"]
        s_ticks = set(ps.filter(pl.col("round") == r)["tick"].unique().to_list())
        v_ticks = set(pv.filter(pl.col("round") == r)["tick"].unique().to_list())
        snaps = [F + 64 * k for k in range(0, 100000) if F + 64 * k < E]
        in_s = sum(1 for t in snaps if t in s_ticks)
        in_v = sum(1 for t in snaps if t in v_ticks)
        span_s = [t for t in s_ticks if F <= t < E]
        cover.append((r, F, E, len(snaps), in_s, in_v, len(span_s),
                      (min(span_s) - F) if span_s else None))
    tot_snaps = sum(c[3] for c in cover)
    print("snapshots (all rounds, exact freeze+64k < end):", tot_snaps,
          "with exact status row:", sum(c[4] for c in cover), "with exact vector row:", sum(c[5] for c in cover))
    for c in cover[:6]:
        print("  round", c[0], "freeze", c[1], "end", c[2], "snaps", c[3], "exact_status", c[4],
              "exact_vector", c[5], "status ticks in live", c[6], "first status tick - freeze", c[7])

    # player_id key: status ids vs tome player_info ids per round
    pid_info = collections.defaultdict(set)
    for x in PI.iter_rows(named=True):
        pid_info[int(x["round"])].add(int(x["player_id"]))
    agree = 0
    tot = 0
    for r in sorted(set(int(v) for v in ps["round"].unique().to_list())):
        sids = set(ps.filter(pl.col("round") == r)["player_id"].unique().to_list())
        tot += 1
        agree += int(sids <= pid_info.get(r, set()))
    print("rounds where all status player_ids are in player_info ids (same round):", agree, "of", tot)

    # alive count check, first 6 rounds with a freeze row
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        deaths[int(x["round"])].append((int(x["tick"]), int(x["player_id"]),
                                        x["player_team_code"]))
    team_pi = {(int(x["round"]), int(x["player_id"])): x["team_code"] for x in PI.iter_rows(named=True)}
    res = collections.Counter()
    for row in rounds.iter_rows(named=True):
        r, F, E = row["round"], row["freeze"], row["end"]
        seen = set()
        dlist = []
        for tk, pid, side in deaths.get(r, []):
            if tk < F or tk > E:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            s = side if side is not None else team_pi.get((r, pid))
            dlist.append((tk, s))
        sub = ps.filter(pl.col("round") == r)
        vsub = pv.filter(pl.col("round") == r)
        joined = sub.join(vsub, on=["tick", "round", "player_id"], how="left")
        for k in range(0, 100000):
            t = F + 64 * k
            if t >= E:
                break
            alive_rows = joined.filter(pl.col("tick") == t)
            n_t = int((alive_rows["team_code"] == T).sum())
            n_ct = int((alive_rows["team_code"] == CT).sum())
            d_strict_t = sum(1 for tk, s in dlist if tk < t and s == T)
            d_strict_ct = sum(1 for tk, s in dlist if tk < t and s == CT)
            d_incl_t = sum(1 for tk, s in dlist if tk <= t and s == T)
            d_incl_ct = sum(1 for tk, s in dlist if tk <= t and s == CT)
            ok_strict = (n_t == START - d_strict_t) and (n_ct == START - d_strict_ct)
            ok_incl = (n_t == START - d_incl_t) and (n_ct == START - d_incl_ct)
            res["snapshots"] += 1
            res["ok_strict_rule"] += int(ok_strict)
            res["ok_inclusive_rule"] += int(ok_incl)
            if not ok_strict and len(res) < 40:
                res[f"ex_round{r}_k{k}_T{n_t}_CT{n_ct}_dT{d_strict_t}_dCT{d_strict_ct}"] += 1
    print("alive check:", {k: v for k, v in res.items() if not k.startswith("ex_")})
    print("examples (strict):", [k for k in res if k.startswith("ex_")][:10])


if __name__ == "__main__":
    main()
