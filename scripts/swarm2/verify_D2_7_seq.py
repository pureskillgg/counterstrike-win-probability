"""D2-7 verifier, exploration: per-match round sequences around round 13 in mismatched matches."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
TMP = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
base = pl.read_parquet(TMP / "d27_rounds_2024.parquet")


def read(folder, cols):
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


RS = read(BASE / f"basic_round_state{SUF}", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
ev = RS.filter(pl.col("event_type").is_in(["round_announce_last_round_half", "round_start", "round_freeze_end",
                                            "bomb_planted", "bomb_defused", "bomb_exploded", "round_end",
                                            "cs_win_panel_round", "round_announce_match_point"]))
BS = read(BASE / f"basic_bomb_state{SUF}", ["match_key", "round", "tick", "event_type"])

# day >= 12 mismatch matches with round 13 bomb outcome
r13 = base.filter((pl.col("round") == 13) & (pl.col("bomb_bs") != "none") & (pl.col("day").cast(pl.Int64) >= 12))
mm = r13.filter(((pl.col("bomb_bs") == "exploded") & (pl.col("winner") != 2)) |
                ((pl.col("bomb_bs") == "defused") & (pl.col("winner") != 3)))
print("day>=12 round-13 rows with bomb outcome:", r13.height, "mismatches:", mm.height)
sample_keys = mm.sort("match_key").select("match_key").head(3)["match_key"].to_list()
print("sample mismatch matches:", sample_keys)

RE = read(BASE / f"basic_round_end{SUF}", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
for mk in sample_keys:
    print("\n==== match", mk)
    seq = base.filter((pl.col("match_key") == mk) & (pl.col("round") >= 10) & (pl.col("round") <= 15)) \
        .select(["round", "winner", "win_reason_code", "bomb_bs", "t_score", "ct_score", "t_next", "ct_next",
                 "score_winner_noswap", "score_winner_swap", "freeze_tick", "tick"]).sort("round")
    print(seq)
    e = ev.filter((pl.col("match_key") == mk) & (pl.col("round") >= 10) & (pl.col("round") <= 15)) \
        .select(["round", "tick", "event_type", "t_score", "ct_score"]).sort(["round", "tick"])
    print(e)
    b = BS.filter((pl.col("match_key") == mk) & (pl.col("round") >= 11) & (pl.col("round") <= 14)) \
        .sort(["round", "tick"])
    print("bomb_state rows:")
    print(b)
