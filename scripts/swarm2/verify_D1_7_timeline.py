"""verify_D1_7 timeline (swarm2 verifier).

For a handful of overtime matches (2024 and 2026), print round-by-round:
winner_team_code, game win message, round_state score first/last values, and the
event types / ticks at which the score fields change. Used to understand where the
score increment lives relative to the round window, before any recomputation.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")


def files(root, ch):
    return sorted(glob.glob(str(root / f"basic_{ch}*" / "dataframe_*")))


def scan(root, ch, cols):
    return pl.concat([pl.scan_parquet(f).select(cols) for f in files(root, ch)], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)


def show_matches(root, label, n_good, n_null):
    RE = scan(root, "round_end", ["match_key", "round", "winner_team_code", "win_reason_message", "player_count", "tick"]).collect().unique(["match_key", "round"])
    RS = scan(root, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect()
    # matches that reached round 27
    m27 = RE.filter(pl.col("round") == 27)["match_key"].unique().sort().to_list()
    hdr(f"{label}: matches reaching round 27 = {len(m27)}")
    # pick examples: those with round 27 present in RS and the 27 score derivation
    chosen = m27[:n_good + n_null]
    for mk in chosen:
        rs = RS.filter(pl.col("match_key") == mk).sort(["round", "tick"])
        re_ = RE.filter(pl.col("match_key") == mk).sort("round")
        print("\nmatch", mk, "player_count r27:", re_.filter(pl.col("round") == 27)["player_count"].to_list())
        rr = list(range(23, 31))
        for r in rr:
            x = rs.filter(pl.col("round") == r)
            if x.height == 0:
                print(f"  r{r}: no round_state rows")
                continue
            first = x.row(0, named=True)
            last = x.row(-1, named=True)
            w = re_.filter(pl.col("round") == r)
            wt = w["winner_team_code"].to_list()
            wm = w["win_reason_message"].to_list()
            # tick positions of score changes within round
            chg = []
            prev = None
            for row in x.iter_rows(named=True):
                cur = (row["t_score"], row["ct_score"])
                if prev is not None and cur != prev:
                    chg.append((row["tick"], row["event_type"], prev, cur))
                prev = cur
            print(f"  r{r}: winner={wt} msg={[s.replace('#SFUI_Notice_','') for s in wm]} "
                  f"first(t,ct)=({first['t_score']},{first['ct_score']}) last(t,ct)=({last['t_score']},{last['ct_score']}) "
                  f"rows={x.height}")
            for c in chg[:6]:
                print(f"      change tick={c[0]} ev={c[1]} {c[2]} -> {c[3]}")


show_matches(T24, "2024", 3, 2)
show_matches(T26, "2026", 3, 0)
