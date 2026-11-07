"""verify_D7_5_30_detail: drill-down for the D7-5 verification.

Items: (A) header total test on the full round sequence (not filtered by player_count, which is a per-round
count and can drop rounds from a match); (B) decomposition of code vs within disagreements by round number;
(C) bomb-side disagreements of the orientation-free and DP labels; (D) round-27 disagreements in the pre period;
(E) round-attribution (tick window) test; (F) build-vs-time check for the 27 post-time build-9842 rounds.
ASSUMPTION: header tF/cF = t_starters_score_final/ct_starters_score_final as final round totals of the team that
started on T (team A) and the other team (team B). Used as a total/parity test only.
"""
import sys
import pickle
from pathlib import Path
from collections import Counter
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(230)
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet"))
with open(str(SCRATCH / "verify_D7_5_state.pkl"), "rb") as fh:
    ST = pickle.load(fh)
hdr = ST["hdr"]
print("round rows (all player counts):", R.height, "matches:", R["match_key"].n_unique())

# ---------- (A) header total test on full round sequence ----------
print("\n== (A) header total test, full round sequence (all player counts) ==")
def seq_ok(rounds):
    return rounds == list(range(1, len(rounds) + 1))

Rs = R.sort(["match_key", "round"])
per_match = {}
for (mk,), g in Rs.group_by(["match_key"], maintain_order=True):
    rounds = g["round"].to_list()
    per_match[mk] = {
        "period": g["period"][0], "n": len(rounds), "contig": seq_ok(rounds),
        "orient": g["orient"].to_list(), "code": g["W_code"].to_list(),
        "within": g["W_within"].to_list(), "dp": g["W_dp"].to_list(), "bomb": g["W_bomb"].to_list(),
        "rounds": rounds,
    }
print("matches:", len(per_match), "| contiguous 1..N:", sum(1 for v in per_match.values() if v["contig"]),
      "| non-contiguous:", sum(1 for v in per_match.values() if not v["contig"]))
print("non-contiguous by period:", Counter(v["period"] for v in per_match.values() if not v["contig"]))
ex = [(k, v["rounds"][:6], v["n"]) for k, v in per_match.items() if not v["contig"]][:4]
print("examples non-contiguous:", ex)

def verdicts(src, lenient):
    """A = team on T at round 1 (orientation 0). full: rounds 1..N; lenient: rounds 1..N-1 only."""
    out = {}
    for mk, v in per_match.items():
        if not v["contig"]:
            out[mk] = "noncontig"; continue
        h = hdr[mk]
        tF, cF = h["t_starters_score_final"], h["ct_starters_score_final"]
        N = v["n"]
        o = v["orient"]
        if any(x is None for x in o):
            out[mk] = "no_orient"; continue
        if src == "code":
            sides = list(v["code"])
        elif src == "within":
            sides = list(v["within"])
        else:
            sides = list(v["dp"])
            sides[-1] = v["within"][-1]
        if lenient:
            # use rounds 1..N-1 only (the last round is not inferred from a next freeze_end)
            idx = list(range(N - 1))
            if any(sides[i] is None for i in idx):
                out[mk] = "missing"; continue
            A = sum(1 for i in idx if (sides[i] == 2 and o[i] == 0) or (sides[i] == 3 and o[i] == 1))
            Bt = (N - 1) - A
            okA = tF - A in (0, 1)
            okB = cF - Bt in (0, 1)
            out[mk] = "pass" if (okA and okB and (tF - A) + (cF - Bt) == 1) else "fail"
        else:
            if any(s is None for s in sides):
                out[mk] = "missing"; continue
            A = sum(1 for s, oo in zip(sides, o) if (s == 2 and oo == 0) or (s == 3 and oo == 1))
            Bt = N - A
            if A == tF and Bt == cF:
                out[mk] = "pass"
            elif A == cF and Bt == tF:
                out[mk] = "pass_swapped"
            else:
                out[mk] = "fail"
    return out

for src in ["code", "within", "dp"]:
    for lenient in [False, True]:
        vd = verdicts(src, lenient)
        cnt = Counter((per_match[mk]["period"], vd[mk]) for mk in vd)
        tag = "lenient(rounds 1..N-1)" if lenient else "full(rounds 1..N)"
        print(f"source={src:7s} {tag:22s}:", sorted(cnt.items()))

# ---------- (B) code vs within disagreement by round number ----------
print("\n== (B) code != within by round number (all player counts, both labels present) ==")
both = R.filter(pl.col("W_code").is_not_null() & pl.col("W_within").is_not_null())
both = both.with_columns((pl.col("W_code") != pl.col("W_within")).alias("dis"))
for p in ["pre", "post"]:
    s = both.filter(pl.col("period") == p)
    g = s.group_by("round").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).filter(pl.col("dis") > 0).sort("round")
    print(p, "rounds with disagreements (round, n, dis):", g.rows())
print("post disagreements by rbin (r13 / r25 / other):",
      both.filter(pl.col("period") == "post").group_by("rbin").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort("rbin").rows())

# ---------- (C) bomb-side disagreements of within and dp ----------
print("\n== (C) bomb-side disagreements (player_count==20 and all) ==")
Bm = R.filter(pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0))
for label, sub in [("all player counts", Bm), ("player_count==20", Bm.filter(pl.col("player_count") == 20))]:
    w = sub.filter(pl.col("W_within").is_not_null() & (pl.col("W_within") != pl.col("W_bomb")))
    d = sub.filter(pl.col("W_dp").is_not_null() & (pl.col("W_dp") != pl.col("W_bomb")))
    print(label, "within!=bomb:", w.height, "of", sub.filter(pl.col("W_within").is_not_null()).height,
          "| dp!=bomb:", d.height, "of", sub.filter(pl.col("W_dp").is_not_null()).height)
    print("  within!=bomb by period/rbin:", w.group_by(["period", "rbin"]).len().sort(["period", "rbin"]).rows())
    print("  dp!=bomb by period/rbin:", d.group_by(["period", "rbin"]).len().sort(["period", "rbin"]).rows())
wd = Bm.filter(pl.col("W_within").is_not_null() & (pl.col("W_within") != pl.col("W_bomb")))
print("\nwithin!=bomb rows (all periods):")
print(wd.select(["match_key", "round", "period", "build", "player_count", "mdt", "map", "W_bomb", "W_code", "W_within", "W_dp",
                 "dT", "dC", "fe_t", "fe_c", "reason_msg", "bomb_out_tick", "fe_tick", "fe_next_tick", "roe_tick", "re_tick", "planter", "maxr"]).sort(["period", "mdt"]))

# ---------- (D) round-27 disagreements in pre period (all player counts) ----------
print("\n== (D) pre-period code != within, by round number 27 vs others, with bomb availability ==")
pre_dis = both.filter((pl.col("period") == "pre") & pl.col("dis"))
print("pre code!=within rows:", pre_dis.height, "| round values:", Counter(pre_dis["round"].to_list()))
print("pre code!=within rows with bomb outcome:", pre_dis.filter(pl.col("W_bomb").is_not_null()).height)
print(pre_dis.select(["match_key", "round", "mdt", "map", "player_count", "W_code", "W_within", "W_dp", "W_bomb", "dT", "dC",
                      "fe_t", "fe_c", "reason_msg", "maxr"]).head(12))
print("pre round-27 rows overall:", R.filter((pl.col("period") == "pre") & (pl.col("round") == 27)).height,
      "| code!=within:", R.filter((pl.col("period") == "pre") & (pl.col("round") == 27) & pl.col("W_code").is_not_null() & pl.col("W_within").is_not_null() & (pl.col("W_code") != pl.col("W_within"))).height)

# ---------- (E) round attribution: tick windows ----------
print("\n== (E) tick-window test for bomb rows: is bomb outcome tick inside the round window [fe_tick, fe_next_tick)? ==")
Wd = R.filter(pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0) & pl.col("fe_tick").is_not_null() & pl.col("fe_next_tick").is_not_null())
Wd = Wd.with_columns(
    ((pl.col("bomb_out_tick") >= pl.col("fe_tick")) & (pl.col("bomb_out_tick") < pl.col("fe_next_tick"))).alias("bomb_in_win"),
    ((pl.col("re_tick") >= pl.col("fe_tick")) & (pl.col("re_tick") < pl.col("fe_next_tick"))).alias("re_in_win"),
    (pl.col("W_code") != pl.col("W_bomb")).alias("code_mm"),
    (pl.col("W_within") != pl.col("W_bomb")).alias("within_mm"),
)
print(Wd.group_by(["period", "code_mm", "bomb_in_win", "re_in_win"]).len().sort(["period", "code_mm"]).rows())
print("(columns: period, code_mismatch_vs_bomb, bomb_tick_inside_window, round_end_tick_inside_window, n)")
print("within mismatch vs bomb by bomb_in_win:", Wd.group_by(["period", "within_mm", "bomb_in_win"]).len().sort(["period", "within_mm"]).rows())
lag = Wd.with_columns((pl.col("bomb_out_tick") - pl.col("re_tick")).alias("lag"))
print("post, lag (bomb_out - round_end tick) for code-mismatch rows (top):", lag.filter((pl.col("period") == "post") & pl.col("code_mm")).group_by("lag").len().sort("len", descending=True).head(5).rows())
print("post, lag for code-clean rows (top):", lag.filter((pl.col("period") == "post") & ~pl.col("code_mm")).group_by("lag").len().sort("len", descending=True).head(5).rows())

# ---------- (F) build vs time: post-time build-9842 rounds ----------
print("\n== (F) build 9842 rounds dated after 2024-02-07 01:01:18 UTC (period_time=post, build=9842) ==")
F = R.filter((pl.col("period_time") == "post") & (pl.col("build") == 9842))
print("rows:", F.height, "| matches:", F["match_key"].n_unique())
print(F.select(["match_key", "round", "mdt", "map", "W_code", "W_within", "W_dp", "W_bomb"]).sort("mdt").head(30))
print("code!=within among these:", F.filter(pl.col("W_code").is_not_null() & pl.col("W_within").is_not_null() & (pl.col("W_code") != pl.col("W_within"))).height,
      "| with bomb:", F.filter(pl.col("W_bomb").is_not_null()).height)

# ---------- (G) post-period code != within for non-first-after-swap, by rbin and map ----------
print("\n== (G) post code!=within by map (player_count==20) ==")
Pm = both.filter((pl.col("period") == "post") & (pl.col("player_count") == 20))
print(Pm.group_by("map").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort("map").rows())
