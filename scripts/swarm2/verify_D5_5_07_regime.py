r"""Verifier for D5-5, stage 6: regime signature and cross-period check.

  R1. Round-1 capture: freeze_end(1) and round_end(1) presence by r1 (round_end label 1 present).
  R2. Score-sum consistency: freeze_end(k) scores sum to k-1 in both regimes (no label offset).
  R3. Reason coding by r1 across all labels: is round_end reason a function of winner in r1-true matches?
  R4. Denominators: label-13 bomb-outcome rows with player_count==20, folder days 12..25 and 01..07.
  P1. 2026 tome cross-check (round_end + bomb_state only): label-13 bomb mismatch by r1 regime,
      and reason code presence.

Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_5_07_regime.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_5")
W = pl.read_parquet(SCR / "wide.parquet")
RS = pl.read_parquet(SCR / "round_state_sub.parquet")
RE = pl.read_parquet(SCR / "round_end.parquet")
R1 = W.filter(pl.col("k") == 1).select(["match_key", pl.col("rw").is_not_null().alias("r1")])
matches = pl.read_parquet(SCR / "header.parquet").select(["key"]).rename({"key": "match_key"})
R1 = matches.join(R1, on="match_key", how="left").with_columns(pl.col("r1").fill_null(False))
W = W.join(R1, on="match_key", how="left")

# ---------- R1. round-1 capture ----------
fe1 = (RS.filter((pl.col("event_type") == "round_freeze_end") & (pl.col("round") == 1))
       .select("match_key").unique().with_columns(fe1=pl.lit(True)))
print("== R1. round 1 capture by r1 ==")
X = R1.join(fe1, on="match_key", how="left").with_columns(pl.col("fe1").fill_null(False))
print(X.group_by(["r1", "fe1"]).agg(pl.len().alias("matches")).sort(["r1", "fe1"]))

# ---------- R2. freeze_end score sums vs label ----------
fes = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
       .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t"), pl.col("ct_score").first().alias("ct")))
fes = fes.with_columns(sumsc=pl.col("t") + pl.col("ct")).with_columns(resid=pl.col("sumsc") - (pl.col("round") - 1))
fes = fes.join(R1, on="match_key", how="left")
print("\n== R2. freeze_end(k): (t+ct) - (k-1) by r1 (0 = no label offset) ==")
print(fes.group_by(["r1", "resid"]).agg(pl.len().alias("n")).sort(["r1", "resid"]).head(20))

# ---------- R3. reason coding ----------
print("\n== R3. round_end reason x winner by r1, all labels (counts) ==")
RR = W.filter(pl.col("rw").is_not_null()).select(["r1", "rw", "reason"])
print(RR.group_by(["r1", "rw", "reason"]).agg(pl.len().alias("n")).sort(["r1", "rw", "reason"]))

# ---------- R4. denominators with player_count == 20 ----------
L13 = W.filter((pl.col("k") == 13) & pl.col("rw").is_not_null() & pl.col("bomb_bs").is_not_null() & (pl.col("bomb_bs") >= 0))
pc = RE.select(["match_key", "round", "player_count"]).rename({"round": "k"})
L13 = L13.join(pc, on=["match_key", "k"], how="left", suffix="_re").with_columns(mm=pl.col("bomb_bs") != pl.col("rw"))
d12 = [f"{d:02d}" for d in range(12, 26)]
d01 = [f"{d:02d}" for d in range(1, 8)]
print("\n== R4. label-13 bomb rows with pc==20: folder days 12..25 and 01..07 ==")
print(L13.filter(pl.col("player_count") == 20).with_columns(
    grp=pl.when(pl.col("day").is_in(d12)).then(pl.lit("12..25")).when(pl.col("day").is_in(d01)).then(pl.lit("01..07")).otherwise(pl.lit("other")))
      .group_by("grp").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")))

# ---------- P1. 2026 tome cross-check ----------
print("\n== P1. 2026 tome: label-13 bomb outcome vs round_end winner, by r1 regime ==")
base26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
dirs = sorted([p.name for p in base26.iterdir() if p.is_dir()])
re_dirs = [d for d in dirs if d.startswith("basic_round_end")]
bs_dirs = [d for d in dirs if d.startswith("basic_bomb_state")]
print("2026 round_end dirs:", re_dirs, " bomb_state dirs:", bs_dirs)
if re_dirs and bs_dirs:
    fr = sorted(glob.glob(str(base26 / re_dirs[0] / "dataframe_*")))
    fb = sorted(glob.glob(str(base26 / bs_dirs[0] / "dataframe_*")))
    RE26 = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
                      for f in fr], how="vertical_relaxed").collect()
    BS26 = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type"])
                      .filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])) for f in fb],
                     how="vertical_relaxed").collect()
    print("2026 round_end rows:", RE26.height, " matches:", RE26["match_key"].n_unique())
    r1_26 = set(RE26.filter(pl.col("round") == 1)["match_key"].to_list())
    RE26 = RE26.with_columns(r1=pl.col("match_key").is_in(list(r1_26)))
    B26 = (BS26.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
           .with_columns(bomb=pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2))))
    J = (RE26.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
            pl.col("winner_team_code").first().alias("rw"), pl.col("win_reason_code").first().alias("reason"),
            pl.col("r1").first().alias("r1"))
         .join(B26.select(["match_key", "round", "bomb"]), on=["match_key", "round"], how="inner"))
    print("2026 rounds with bomb outcome, mismatch by r1 (all labels):")
    print(J.with_columns(mm=pl.col("bomb") != pl.col("rw")).group_by("r1").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")))
    print("2026 label-13 rows with bomb outcome:")
    J13 = J.filter(pl.col("round") == 13).with_columns(mm=pl.col("bomb") != pl.col("rw"))
    print(J13.group_by("r1").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")))
    print("2026 reason x winner by r1 (all labels):")
    print(RE26.with_columns(rw=pl.col("winner_team_code")).group_by(["r1", "rw", "win_reason_code"]).agg(pl.len().alias("n"))
          .sort(["r1", "rw", "win_reason_code"]))
