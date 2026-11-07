"""verify_D1_5 final checks.
X1  bomb-event attribution: are the round-13 bomb outcome ticks inside the round-13 window
    (freeze_end(13) .. round_end tick(13))? Compare bomb-mismatch rows with clean rows.
X2  record-level: in build 9961 round-13 disagreements, does win_reason_message agree with the
    (wrong) round_end winner or with the bomb/score side?
X3  map question: within build 9961, d12-25, round-13 disagreement by map_name.
X4  build 9842 control: any disagreement in any round? (all rounds, all days)
ASSUMPTIONS: A1 side codes, A5 swap rule (MR24, A8), fe-derived side as in verify_D1_5_fe.py.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(45)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
MR = 24


def swap_expected(r, mr=MR):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def fl(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}{SUF24}" / "dataframe_*")))


RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                            "win_reason_message", "player_count"]) for f in fl("round_end")],
               how="vertical_relaxed").unique(["match_key", "round"], keep="first")
RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
                for f in fl("round_state")], how="vertical_relaxed")
BS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type"]) for f in fl("bomb_state")],
               how="vertical_relaxed")

fe_df = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
         .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t"), pl.col("ct_score").first().alias("c"),
                                              pl.col("tick").first().alias("fe_tick")))
fe = {(m, int(r)): (int(t), int(c), int(ft)) for m, r, t, c, ft in
      fe_df.select(["match_key", "round", "t", "c", "fe_tick"]).iter_rows()}


def fe_winner(m, r):
    a = fe.get((m, r))
    b = fe.get((m, r + 1))
    if a is None or b is None:
        return None
    t_r, c_r, _ = a
    t_n, c_n, _ = b
    if swap_expected(r):
        dT, dC = c_n - t_r, t_n - c_r
    else:
        dT, dC = t_n - t_r, c_n - c_r
    if dT == 1 and dC == 0:
        return 2
    if dC == 1 and dT == 0:
        return 3
    return None


outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("bomb_tick")))
HD0 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(T24 / "header" / "dataframe_*")))], how="vertical_relaxed")
HDm = HD0.select([pl.col("key").alias("match_key"), "build_num", "map_name"]).unique("match_key")

# all rounds with fe-derived winner
rows = []
for m, r, w, tick, wrm, pc in RE.select(["match_key", "round", "winner_team_code", "tick", "win_reason_message", "player_count"]).iter_rows():
    fw = fe_winner(m, int(r))
    if fw is None:
        continue
    rows.append((m, int(r), int(w), fw, int(tick), wrm))
D = pl.DataFrame(rows, schema=["match_key", "round", "w", "fw", "end_tick", "wrm"], orient="row").with_columns(
    (pl.col("w") != pl.col("fw")).alias("dis"),
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
D = D.join(HDm, on="match_key", how="left")

# X4 build 9842 control
hdr = lambda t: print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)  # noqa: E731
hdr("X4 build 9842 control: disagreements in any round")
print(D.group_by("build_num").agg(pl.len().alias("n_rounds"), pl.col("dis").sum().alias("disagree")).sort("build_num"))

# X1 attribution: bomb tick inside round-13 window
hdr("X1 round-13 bomb outcome tick vs round-13 window (freeze_end tick .. round_end tick)")
R13 = D.filter(pl.col("round") == 13).join(outc.filter(pl.col("round") == 13), on="match_key", how="inner")
R13 = R13.with_columns(pl.col("match_key").map_elements(lambda m: fe.get((m, 13), (None, None, None))[2], return_dtype=pl.Int64).alias("fe13_tick"))
R13 = R13.with_columns(((pl.col("bomb_tick") >= pl.col("fe13_tick")) & (pl.col("bomb_tick") <= pl.col("end_tick"))).alias("in_window"),
                       (pl.col("end_tick") - pl.col("bomb_tick")).alias("end_minus_bomb_ticks"))
print(R13.group_by(["dis", "in_window"]).agg(pl.len().alias("n"),
                                              pl.col("end_minus_bomb_ticks").median().alias("median_end_minus_bomb"),
                                              pl.col("end_minus_bomb_ticks").min().alias("min_end_minus_bomb")).sort(["dis", "in_window"]))
print("tick rate: 64 per second (assumption A10); median end-minus-bomb ticks above is the gap between the bomb event and the recorded round_end tick")

# X2 record-level: win_reason in build 9961 round-13 disagreements
hdr("X2 build 9961 round 13: win_reason_message vs round_end winner vs fe-derived side (d12-25)")
X2 = D.filter((pl.col("round") == 13) & (pl.col("build_num") == 9961) & pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)]))
X2 = X2.with_columns(pl.col("wrm").str.contains("Terrorists|Target_Bombed").alias("wr_says_T"))
print(X2.group_by(["dis", "wrm"]).agg(pl.len().alias("n"), pl.col("w").first().alias("w_example"), pl.col("fw").first().alias("fw_example")).sort(["dis", "n"], descending=[False, True]))
print("win_reason consistent with the round_end winner (T msg with 2, CT msg with 3):")
print(X2.with_columns(((pl.col("wr_says_T") & (pl.col("w") == 2)) | (~pl.col("wr_says_T") & (pl.col("w") == 3))).alias("wr_eq_w"))
      .group_by(["dis", "wr_eq_w"]).len().sort(["dis", "wr_eq_w"]))

# X3 map question within build 9961, d12-25
hdr("X3 round 13, build 9961, d12-25: disagreement by map_name (is the map the discriminator?)")
X3 = D.filter((pl.col("round") == 13) & (pl.col("build_num") == 9961) & pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)]))
print(X3.group_by("map_name").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"),
                                 pl.col("dis").mean().round(3).alias("rate")).sort("n", descending=True))
print("overall build 9961 round 13 d12-25 rate:", round(float(X3["dis"].mean()), 4), "n", X3.height)
