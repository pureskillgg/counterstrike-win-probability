import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(160)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {"2026": (ROOT/"tomes"/"2026-10-01_to_08"/"tome"/"csds", ".2026-09-18,2026-10-08"),
         "2024": (ROOT/"tomes"/"2024-02-g"/"tome"/"csds", ".2023-12-28,2024-02-25")}
def scan(base, suf, ch):
    return pl.scan_parquet(str(base / f"basic_{ch}{suf}" / "dataframe_*"))
for label, (base, suf) in TOMES.items():
    hdr = pl.scan_parquet(str(base/"header"/"dataframe_*")).select(["key","match_date","t_starters_score_final","ct_starters_score_final"] + (["max_rounds"] if label=="2026" else [])).collect()
    print("==", label, "headers", hdr.height, "unique key", hdr["key"].n_unique())
    re_ = scan(base, suf, "round_end").select(["match_key","round","winner_team_code","player_count"]).collect()
    fe = scan(base, suf, "round_state").filter(pl.col("event_type")=="round_freeze_end").select(["match_key","round","t_score","ct_score","tick"]).collect()
    rs = scan(base, suf, "round_start").select(["match_key","round"]).collect() if False else None
    print("round_end rows", re_.height, "dup (match,round)", re_.height - re_.select(["match_key","round"]).unique().height,
          "matches", re_["match_key"].n_unique())
    print("freeze rows", fe.height, "dup", fe.height - fe.select(["match_key","round"]).unique().height, "matches", fe["match_key"].n_unique())
    print("winner codes", re_.group_by("winner_team_code").len().sort("winner_team_code"))
    print("player_count", re_.group_by("player_count").len().sort("player_count"))
    a = re_.select(["match_key","round"]).unique()
    b = fe.select(["match_key","round"]).unique()
    print("freeze keys with round_end", b.join(a, on=["match_key","round"], how="inner").height, "freeze w/o round_end", b.join(a, on=["match_key","round"], how="anti").height, "round_end w/o freeze", a.join(b, on=["match_key","round"], how="anti").height)
    per_m_fe = b.group_by("match_key").agg(pl.len().alias("nf"), pl.col("round").max().alias("maxr"))
    per_m_re = a.group_by("match_key").agg(pl.len().alias("nre"), pl.col("round").max().alias("maxre"))
    j = per_m_fe.join(per_m_re, on="match_key", how="full", coalesce=True).with_columns(pl.col("nf").fill_null(0), pl.col("nre").fill_null(0))
    print("matches in freeze-set", per_m_fe.height, "in round_end-set", per_m_re.height)
    print("rounds per match (freeze) describe"); print(per_m_fe["nf"].describe())
    print("rounds per match (round_end) describe"); print(per_m_re["nre"].describe())
    print("match with freeze but no round_end", j.filter(pl.col("nre")==0).height, "with round_end but no freeze", j.filter(pl.col("nf")==0).height)
    print("header sum vs round_end count (matches) head:")
    hs = hdr.with_columns((pl.col("t_starters_score_final")+pl.col("ct_starters_score_final")).alias("hsum")).select(["key","hsum"]).rename({"key":"match_key"})
    jj = per_m_re.join(hs, on="match_key", how="left")
    print("header sum == round_end count:", jj.filter(pl.col("nre")==pl.col("hsum")).height, "of", jj.height, "; with header", jj.filter(pl.col("hsum").is_not_null()).height)
    print("round_end count distribution top:", jj.group_by("nre").len().sort("len", descending=True).head(8))
