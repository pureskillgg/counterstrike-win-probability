# verify_A2_1 final controls: same-folder-day build split, date histogram, full-length exceptions,
# and reason-code specificity for bomb rounds (normal vs switch rounds). Uses the saved round table.
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(40)

SP = r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
R = pl.read_parquet(os.path.join(SP, "verify_A2_1_round_table.parquet"))
hdr_path = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds\header\dataframe_00000"
import glob  # noqa: E402
import polars as _pl  # noqa: E402
hdr = _pl.scan_parquet(hdr_path).select(["key", "match_date", "build_num"]).collect().rename({"key": "match_key"})
hdr = hdr.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("fday"),
                       pl.col("match_date").str.slice(0, 10).alias("mdate"))

# re-derive fe-based score winners and per-match format/last round (same rule as the claim)
BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"
fs = sorted(glob.glob(os.path.join(BASE, f"basic_round_state{SUF}", "dataframe_*")))
fes = (pl.scan_parquet(fs[0]).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
       .filter(pl.col("event_type") == "round_freeze_end").collect()
       .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first"))
nx = (fes.sort(["match_key", "round"])
      .with_columns(pl.col("round").shift(-1).over("match_key").alias("r_next"),
                    pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
                    pl.col("ct_score").shift(-1).over("match_key").alias("ct_next"))
      .filter(pl.col("r_next") == pl.col("round") + 1)
      .with_columns((pl.col("t_next") - pl.col("t_score")).alias("dT"),
                    (pl.col("ct_next") - pl.col("ct_score")).alias("dCT"))
      .with_columns(pl.when(pl.col("dT") == 1).then(2).when(pl.col("dCT") == 1).then(3).otherwise(0).alias("w_score"))
      .select(["match_key", "round", "w_score"]))
sec = lambda t: print("\n=== " + t, flush=True)  # noqa: E731

fs2 = (fes.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("first_second")))
rend = pl.scan_parquet(sorted(glob.glob(os.path.join(BASE, f"basic_round_end{SUF}", "dataframe_*")))[0]).select(
    ["match_key", "round"]).collect().group_by("match_key").agg(pl.col("round").max().alias("last_round"))
M = hdr.join(fs2, on="match_key", how="left").join(rend, on="match_key", how="left").with_columns(
    pl.when(pl.col("first_second") == 9).then(pl.lit("MR8"))
    .when(pl.col("first_second") == 13).then(pl.lit("MR12"))
    .otherwise(pl.lit("none")).alias("fmt"))

sec("(F1) same folder-day control: k=13 (MR12) and k=9 (MR8) score-vs-end bad/n by folder day and build")
P = (nx.join(R.select(["match_key", "round", "winner_team_code", "bomb_side"]), on=["match_key", "round"], how="left")
     .join(M.select(["match_key", "build_num", "fday", "fmt", "last_round", "mdate"]), on="match_key", how="left")
     .with_columns((pl.col("w_score") != pl.col("winner_team_code")).alias("bad")))
F = P.filter(((pl.col("fmt") == "MR12") & (pl.col("round") == 13)) | ((pl.col("fmt") == "MR8") & (pl.col("round") == 9)))
F = F.filter(pl.col("w_score") != 0)
g = F.group_by(["fday", "build_num"]).agg(pl.col("bad").sum().alias("bad"), pl.len().alias("n")).sort(["fday", "build_num"])
for r in g.rows():
    print("  folder", r[0], "build", r[1], "bad/n", f"{r[2]}/{r[3]}", flush=True)

sec("(F2) date histogram by build, Feb 2024 (match_date day), counts")
print(M.filter(pl.col("mdate").str.slice(0, 7) == "2024-02").group_by(["build_num", "mdate"]).len()
      .sort(["mdate", "build_num"]).rows(), flush=True)
print("9842 matches dated >= 2024-02-07:", M.filter((pl.col("build_num") == 9842) & (pl.col("mdate") >= "2024-02-07")).height,
      "| 9961 matches dated < 2024-02-07:", M.filter((pl.col("build_num") == 9961) & (pl.col("mdate") < "2024-02-07")).height, flush=True)

sec("(F3) full-length exceptions: 9961 matches with defect-eligible round present (MR12 last>=14, MR8 last>=10) and zero disagreement")
MM = (P.filter(pl.col("w_score") != 0).group_by(["match_key", "build_num", "fmt"])
      .agg(pl.col("bad").sum().alias("nbad"), pl.len().alias("np")).join(M.select(["match_key", "last_round"]), on="match_key"))
ex = MM.filter((pl.col("build_num") == 9961) & (((pl.col("fmt") == "MR12") & (pl.col("last_round") >= 14)) |
                                                ((pl.col("fmt") == "MR8") & (pl.col("last_round") >= 10))) & (pl.col("nbad") == 0))
print("9961 full-length matches with zero disagreement:", ex.height, flush=True)
ex2 = MM.filter((pl.col("build_num") == 9961) & (((pl.col("fmt") == "MR12") & (pl.col("last_round") >= 14)) |
                                                 ((pl.col("fmt") == "MR8") & (pl.col("last_round") >= 10))))
print("9961 full-length matches total:", ex2.height, "| with >=1 disagreement:", ex2.filter(pl.col("nbad") > 0).height, flush=True)
print("9842 full-length matches total:", MM.filter((pl.col("build_num") == 9842) & (((pl.col("fmt") == "MR12") & (pl.col("last_round") >= 14)) | ((pl.col("fmt") == "MR8") & (pl.col("last_round") >= 10)))).height,
      "| with >=1 disagreement:", MM.filter((pl.col("build_num") == 9842) & (((pl.col("fmt") == "MR12") & (pl.col("last_round") >= 14)) | ((pl.col("fmt") == "MR8") & (pl.col("last_round") >= 10))) & (pl.col("nbad") > 0)).height, flush=True)

sec("(F4) reason code for bomb rounds: normal rounds vs switch rounds (9961 first-after-switch vs others) and 9842")
RB = R.filter(pl.col("bomb_side").is_not_null()).join(M.select(["match_key", "fmt"]), on="match_key", how="left")
RB = RB.with_columns(
    pl.when(((pl.col("fmt") == "MR12") & (pl.col("round") == 13)) | ((pl.col("fmt") == "MR8") & (pl.col("round") == 9))
            | ((pl.col("build_num") == 9961) & pl.col("round").is_in([25, 28]))).then(pl.lit("switch"))
    .otherwise(pl.lit("other")).alias("kind"))
print("reason code x bomb side, by build and kind (build, kind, bomb_side, reason_code, n):")
print(RB.group_by(["build_num", "kind", "bomb_side", "win_reason_code"]).len().sort(["build_num", "kind", "bomb_side", "win_reason_code"]).rows(), flush=True)

sec("(F5) bomb-to-winner agreement for normal (non-switch) bomb rounds, by build")
print(RB.group_by(["build_num", "kind"]).agg(pl.len().alias("n"),
                                              (pl.col("bomb_side") != pl.col("winner_team_code")).sum().alias("bad"))
      .sort(["build_num", "kind"]).rows(), flush=True)
print("DONE final", flush=True)
