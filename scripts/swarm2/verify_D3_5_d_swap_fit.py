"""D3-5 verifier, step D: fit a single side swap to each contiguous 2024 match.

Model (ASSUMPTION: one halftime swap at round s, no OT; checked separately for OT matches):
  P = team that starts on T, Q = team that starts on CT. Winner codes are side codes (2 = T side, 3 = CT side).
  Round r is played with P on T iff r <= s. Freeze fields (t_score, ct_score) are side-attached:
  if P is on T at freeze k then (t, ct) = (P_k, Q_k), else (t, ct) = (Q_k, P_k).
  Predicted freeze fields are rolled forward from freeze 1 = (0, 0) using the round_end winners.
For each match, every candidate s (1..N, and 'none' = no swap) is scored by the number of freeze rows matching.
Reports the best s distribution, perfect-fit counts, and which rounds mismatch under the best s.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"])
RE = (RE.with_columns((pl.col("player_count") == 20).cast(pl.Int8).alias("p20"))
      .sort(["match_key", "round", "p20"], descending=[False, False, True])
      .unique(["match_key", "round"], keep="first"))
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))

rounds_re = collections.defaultdict(dict)
for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
    rounds_re[m][r] = w
live = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    live[m][r] = (t, c)

contig = [m for m, d in rounds_re.items() if sorted(d) == list(range(1, len(d) + 1))]
day = lambda m: int(m.split("/")[3])
print("contiguous-from-1 matches:", len(contig), "| day distribution:",
      collections.Counter("12-25" if day(m) >= 12 else "<12" for m in contig))


def predict_fit(winners, fields, s):
    """Return (matches, rows) for freeze rows 2..N+1 compared with the prediction under swap after round s.
    s = 0 means P starts on CT for all rounds (same as no swap with the team mapping flipped, equivalent fit)."""
    N = len(winners)
    P = Q = 0
    ok = bad = 0
    bad_rounds = []
    # freeze 1 is (0, 0) with P on T
    for r in range(1, N + 1):
        w = winners[r]
        pT = r <= s  # P on T during round r (s = N means no swap in the match)
        winP = (w == T and pT) or (w == CT and not pT)
        if winP:
            P += 1
        else:
            Q += 1
        k = r + 1
        pT_next = k <= s
        t_pred, c_pred = (P, Q) if pT_next else (Q, P)
        if k in fields:
            if fields[k] == (t_pred, c_pred):
                ok += 1
            else:
                bad += 1
                bad_rounds.append(r)
    return ok, bad, bad_rounds


best_counter = collections.Counter()
perfect = collections.Counter()
mm_by_round = collections.Counter()
mm_matches_by_round = collections.Counter()
summary = []
for m in contig:
    winners = rounds_re[m]
    N = len(winners)
    fields = live.get(m, {})
    if 1 not in fields:
        best_counter["no_freeze_1"] += 1
    best = None
    for s in range(0, N + 1):
        ok, bad, br = predict_fit(winners, fields, s)
        key = (bad, -ok)
        if best is None or key < best[0]:
            best = (key, s, br)
    (bad, nok), s_best, br = best
    grp = "12-25" if day(m) >= 12 else "<12"
    best_counter[(grp, s_best)] += 1
    if bad == 0:
        perfect[grp] += 1
    else:
        summary.append((m, s_best, bad, br[:12]))
        for r in br:
            mm_by_round[(grp, r)] += 1
        for r in set(br):
            mm_matches_by_round[(grp, r)] += 1
print("\nbest swap round s (N meaning no swap) counts, by day group:")
for k, v in sorted(best_counter.items(), key=lambda kv: str(kv[0])):
    if isinstance(k, tuple) and v >= 5:
        print("  ", k, v)
print("zero-mismatch matches under their best s:", dict(perfect))
print("matches with >=1 mismatch under best s:", sum(1 for x in summary))
print("mismatch rows by winner round (grp, round): rows", sorted(
    [(k, v) for k, v in mm_by_round.items() if v >= 1], key=lambda kv: (kv[0][0], kv[0][1]))[:60])
print("\nsample mismatched matches (match, best_s, n_bad, first bad rounds):")
for x in summary[:8]:
    print("  ", x)
