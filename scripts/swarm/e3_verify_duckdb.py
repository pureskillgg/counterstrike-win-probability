"""Independent check of scripts/swarm/e3_trades.py (E3). Re-derives kills, opening side,
first-10-second kill difference, trade gaps and the point AUCs with DuckDB SQL over the G:
tome pages, a different code path from the Python loops in e3_trades.py.

Compares round by round with logs/swarm/e3_rounds_<era>.parquet (written by e3_trades.py)
and with the point estimates in logs/swarm/e3_trades.json. Reads G: tome pages only.
Run after e3_trades.py has finished:
  pixi run python -I -X utf8 scripts\\swarm\\e3_verify_duckdb.py [2026|2024]
"""
import collections
import glob
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
LOG = ROOT / "logs" / "swarm"
T, CT, TICK, SENT = 2, 3, 64, 65535
ERAS = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}


def files(base, suffix, ch):
    fs = sorted(glob.glob(str(base / f"basic_{ch}{suffix}" / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no pages for {ch}")
    return [Path(f).as_posix() for f in fs]


def src(base, suffix, ch):
    lst = "[" + ",".join("'" + f + "'" for f in files(base, suffix, ch)) + "]"
    return f"read_parquet({lst}, union_by_name=true)"


def auc(y, x):
    y = np.asarray(y, dtype=np.int64)
    x = np.asarray(x, dtype=float)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, x))


def verify(era):
    base, suffix = ERAS[era]
    con = duckdb.connect()
    con.execute(f"CREATE TABLE pd AS SELECT match_key, round, tick, player_id, attacker_id, "
                f"player_team_code, attacker_team_code FROM {src(base, suffix, 'player_death')}")
    con.execute(f"CREATE TABLE rs AS SELECT match_key, round, tick FROM {src(base, suffix, 'round_state')} "
                f"WHERE event_type = 'round_freeze_end'")
    con.execute(f"CREATE TABLE re AS SELECT match_key, round, tick, winner_team_code FROM "
                f"{src(base, suffix, 'round_end')}")

    dump = pl.read_parquet(LOG / f"e3_rounds_{era}.parquet").rename({"match": "match_key"})
    con.register("dump", dump.to_arrow())
    out = {"era": era, "dump_rounds": dump.height}

    con.execute("CREATE TABLE live AS SELECT match_key, round, min(tick) AS live_tick, "
                "count(*) AS n_live, count(DISTINCT tick) AS n_live_ticks FROM rs GROUP BY 1, 2")
    con.execute("CREATE TABLE ends AS SELECT match_key, round, min(tick) AS end_tick, count(*) AS n_end, "
                "min(winner_team_code) AS winner_min, max(winner_team_code) AS winner_max FROM re GROUP BY 1, 2")
    out["live_multi_rows"] = con.execute("SELECT count(*) FROM live WHERE n_live > 1").fetchone()[0]
    out["end_multi_rows"] = con.execute("SELECT count(*) FROM ends WHERE n_end > 1").fetchone()[0]

    # restrict to the dump's rounds, and take each victim's first death in the live window
    con.execute("CREATE TABLE dr AS SELECT d.match_key, d.round, d.ct_win, d.live_tick AS dump_live, "
                "d.end_rel AS dump_end_rel, d.kills_json, l.live_tick, e.end_tick, e.winner_min, e.winner_max "
                "FROM dump d JOIN live l USING (match_key, round) JOIN ends e USING (match_key, round)")
    out["dump_joined"] = con.execute("SELECT count(*) FROM dr").fetchone()[0]
    out["dump_live_mismatch"] = con.execute("SELECT count(*) FROM dr WHERE live_tick <> dump_live").fetchone()[0]
    out["dump_endrel_mismatch"] = con.execute(
        "SELECT count(*) FROM dr WHERE end_tick - live_tick <> dump_end_rel").fetchone()[0]
    out["dump_winner_ctwin_mismatch"] = con.execute(
        "SELECT count(*) FROM dr WHERE (CASE WHEN winner_min = 3 THEN 1 ELSE 0 END) <> ct_win "
        "OR winner_min <> winner_max").fetchone()[0]

    con.execute("CREATE TABLE win AS SELECT p.*, r.live_tick, r.end_tick FROM pd p JOIN dr r USING (match_key, round) "
                "WHERE p.tick >= r.live_tick AND p.tick <= r.end_tick")
    out["in_window_deaths"] = con.execute("SELECT count(*) FROM win").fetchone()[0]
    out["in_window_null_victim_team"] = con.execute(
        "SELECT count(*) FROM win WHERE player_team_code IS NULL OR player_team_code NOT IN (2, 3)").fetchone()[0]
    con.execute("CREATE TABLE w1 AS SELECT *, row_number() OVER (PARTITION BY match_key, round, player_id "
                "ORDER BY tick) AS rn FROM win")
    out["victim_repeat_deaths_in_window"] = con.execute(
        "SELECT count(*) FROM w1 WHERE rn > 1").fetchone()[0]
    out["first_deaths_null_attacker_team_valid_attacker"] = con.execute(
        "SELECT count(*) FROM w1 WHERE rn = 1 AND attacker_id IS NOT NULL AND attacker_id <> 65535 "
        "AND attacker_id <> player_id AND attacker_team_code IS NULL").fetchone()[0]
    con.execute("CREATE TABLE kills AS SELECT match_key, round, tick - live_tick AS rel, "
                "attacker_team_code AS aside FROM w1 WHERE rn = 1 AND attacker_id IS NOT NULL "
                "AND attacker_id <> 65535 AND attacker_id <> player_id AND player_team_code IN (2, 3) "
                "AND attacker_team_code IN (2, 3) AND attacker_team_code <> player_team_code")

    kills = collections.defaultdict(list)
    for mk, rd_, rel, aside in con.execute("SELECT match_key, round, rel, aside FROM kills").fetchall():
        kills[(mk, int(rd_))].append((int(rel), int(aside)))

    rows = con.execute("SELECT match_key, round, ct_win, dump_end_rel, kills_json FROM dr").fetchall()
    mism, n_kill_mine, n_kill_dump = 0, 0, 0
    rounds = []
    for mk, rd_, ct_win, end_rel, kj in rows:
        mine = sorted(kills.get((mk, int(rd_)), []))
        theirs = sorted((int(a), int(b)) for a, b in json.loads(kj))
        n_kill_mine += len(mine)
        n_kill_dump += len(theirs)
        if mine != theirs:
            mism += 1
        rounds.append({"match": mk, "ct_win": int(ct_win), "end_rel": int(end_rel), "kills": mine})
    out["kill_list_mismatch_rounds"] = mism
    out["kills_total_mine"] = n_kill_mine
    out["kills_total_dump"] = n_kill_dump
    out["rounds_compared"] = len(rounds)
    out["matches"] = len({r["match"] for r in rounds})
    out["ct_win_rate"] = float(np.mean([r["ct_win"] for r in rounds]))

    # analysis at state times, recomputed from my kill lists
    main_json = json.loads((LOG / "e3_trades.json").read_text(encoding="utf-8"))[era]
    ref = {a["S_s"]: a for a in main_json["analysis"] if a["label"] == f"{era} all"}
    out["analysis"] = {}
    for S in (10, 20, 30):
        cut = S * TICK
        R = [r for r in rounds if r["end_rel"] >= cut]
        y_all, x_d10, y_o, x_o = [], [], [], []
        for r in R:
            pre = [(rel, s) for rel, s in r["kills"] if rel < cut]
            d10 = sum(1 for rel, s in pre if rel < 10 * TICK and s == CT) - \
                sum(1 for rel, s in pre if rel < 10 * TICK and s == T)
            y_all.append(r["ct_win"])
            x_d10.append(d10)
            if pre:
                first = min(rel for rel, _ in pre)
                sides = {s for rel, s in pre if rel == first}
                if len(sides) == 1:
                    y_o.append(r["ct_win"])
                    x_o.append(int(sides.pop() == CT))
        # trade gaps and first-kill share (whole round, descriptive)
        gaps = []
        for r in rounds:
            k = r["kills"]
            for i, (ti, si) in enumerate(k):
                for tj, sj in k[i + 1:]:
                    if sj != si:
                        gaps.append((tj - ti) / TICK)
                        break
        rec = {
            "n_rounds": len(R),
            "rounds_with_opening": len(y_o),
            "AUC_opening_CT_point": auc(y_o, x_o),
            "AUC_diff10_point": auc(y_all, x_d10),
            "mean_kills_first10s": float(np.mean([sum(1 for rel, _ in r["kills"] if rel < 10 * TICK)
                                                  for r in R])) if R else None,
        }
        a = ref.get(S)
        if a is not None:
            rec["main_n_rounds"] = a["n_rounds"]
            rec["main_rounds_with_opening"] = a["rounds_with_opening"]
            rec["main_AUC_opening_CT_point"] = a.get("AUC_opening_CT", {}).get("auc")
            rec["main_AUC_diff10_point"] = a.get("AUC_diff10", {}).get("auc")
        out["analysis"][str(S)] = rec
    if ref:
        tg = main_json["trade_gap"]
        out["trade_gap_mine"] = {"kills_with_later_other_side_kill": len(gaps),
                                 "median_s": float(np.median(gaps)),
                                 "share_le_5s": float(np.mean(np.asarray(gaps) <= 5))}
        out["trade_gap_main"] = {"kills_with_later_other_side_kill": tg["kills_with_later_other_side_kill"],
                                 "median_s": tg["median_s"], "share_le_5s": tg["share_le_5s"]}
    first_s = [r["kills"][0][0] / TICK for r in rounds if r["kills"]]
    out["share_first_kill_before_10s_mine"] = float(np.mean([t < 10 for t in first_s]))
    out["share_first_kill_before_10s_main"] = main_json["share_rounds_first_kill_before_s"]["10"]
    return out


def main():
    eras = sys.argv[1:] or ["2026", "2024"]
    res = {}
    for era in eras:
        res[era] = verify(era)
        print(json.dumps(res[era], indent=2, default=str))
    (LOG / "e3_verify.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"wrote {LOG / 'e3_verify.json'}")


if __name__ == "__main__":
    main()
