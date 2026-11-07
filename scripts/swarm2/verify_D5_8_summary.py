r"""D5-8 verifier summary: reproducible key numbers, written to logs\swarm2\verify_D5_8.log (stdout is tee'd).

Pipeline (run in order): verify_D5_8_extract.py -> verify_D5_8_analysis.py -> this script.
Inputs: raw 2024 tome channels (round_end, round_state, bomb_state, header) and scratch parquet from the extract
step (not the claimed D5 scratchpad tables). Intermediate parquet lives in the scratchpad, not in the repo.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_summary.py
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

LOG = Path(r"G:\pureskill\csds-cs2\logs\swarm2\verify_D5_8.log")
LOG.parent.mkdir(parents=True, exist_ok=True)
_LOG_FH = open(LOG, "w", encoding="utf-8")


def out(*parts):
    line = " ".join(str(p) for p in parts)
    sys.__stdout__.write(line + "\n")
    _LOG_FH.write(line + "\n")


pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
OUT = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_8v")
S_SWAPS = [12, 24, 27, 30, 33]
TREASON = [1, 9, 13, 18]
CTREASON = [7, 8, 11, 12, 17]

R = pl.read_parquet(OUT / "rounds_v2.parquet").sort(["match_key", "round"])
M = pl.read_parquet(OUT / "matches_v.parquet")
hfiles = sorted(glob.glob(os.path.join(BASE, "header", "dataframe_*")))
hdr = pl.concat([pl.read_parquet(f, columns=["match_id", "t_starters_score_final", "ct_starters_score_final"])
                 for f in hfiles], how="vertical_relaxed").unique("match_id")

out("D5-8 VERIFIER LOG (independent recomputation from raw 2024 tome channels)")
out("rounds:", R.height, " matches:", M.height, " regime A:", M.filter(pl.col("regime") == "A").height,
    " regime B:", M.filter(pl.col("regime") == "B").height)
rmin_tbl = (R.group_by("match_key").agg(pl.col("round").min().alias("rmin"))
            .join(M.select(["match_key", "regime"]), on="match_key").group_by(["regime", "rmin"]).len().sort(["regime", "rmin"]))
out("regime x first round (rmin):", rmin_tbl.rows())

# 1. reason vs bomb consistency (engine-internal), by regime and round
out("\n[1] round_end reason vs bomb outcome (engine-internal consistency)")
Q = R.filter(pl.col("bside").is_not_null()).with_columns(
    reason_side=pl.when(pl.col("reason").is_in(TREASON)).then(2).when(pl.col("reason").is_in(CTREASON)).then(3).otherwise(None))
Q = Q.with_columns(
    reason_bomb_conflict=(pl.col("reason_side") != pl.col("bside")).cast(pl.Int64),
    winner_bomb_conflict=(pl.col("winner") != pl.col("bside")).cast(pl.Int64),
    rb=pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")))
out(Q.group_by(["regime", "rb"]).agg(pl.len().alias("n"), pl.col("reason_bomb_conflict").sum().alias("reason_conflict"),
                                     pl.col("winner_bomb_conflict").sum().alias("winner_conflict")).sort(["regime", "rb"]).rows())

# 2. same-day contrast at round 13 (confound check)
out("\n[2] same-day contrast at round 13: (day, regime, rows with bomb outcome, winner-vs-bomb conflicts)")
b13 = R.filter((pl.col("round") == 13) & pl.col("bside").is_not_null()).with_columns(
    mm=(pl.col("winner") != pl.col("bside")).cast(pl.Int64))
dd = (b13.filter(pl.col("day").is_in(["12", "13", "14", "15", "16", "18", "19", "20", "21", "22", "23", "24"]))
      .group_by(["day", "regime"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["day", "regime"]))
out(dd.rows())

# 3. freeze-end step vs label (no phase change, clean +1 step)
out("\n[3] freeze-end step vs label (no phase change, clean +1 step)")
X = R.with_columns(nt=pl.col("fe_t").shift(-1).over("match_key"), nct=pl.col("fe_ct").shift(-1).over("match_key"),
                   nr=pl.col("round").shift(-1).over("match_key"), nph=pl.col("fe_phase").shift(-1).over("match_key"))
X = X.filter((pl.col("nr") == pl.col("round") + 1) & (pl.col("fe_phase") == pl.col("nph"))
             & pl.col("fe_t").is_not_null() & pl.col("nt").is_not_null())
X = X.with_columns(dt=pl.col("nt") - pl.col("fe_t"), dct=pl.col("nct") - pl.col("fe_ct")).with_columns(
    st=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(2)
    .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(3).otherwise(None).cast(pl.Int64))
X = X.filter(pl.col("st").is_not_null())
X = X.with_columns(dis=(pl.col("st") != pl.col("winner")).cast(pl.Int64))
X13 = X.filter(pl.col("round") == 13).with_columns(hasbomb=pl.col("bside").is_not_null())
out("round 13 by group and bomb presence (rows, step-vs-label disagreements):")
out(X13.group_by(["grp", "hasbomb"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("disagree")).sort(["grp", "hasbomb"]).rows())
for g in ["lateA", "lateB"]:
    c = X.filter((pl.col("grp") == g) & pl.col("round").is_between(6, 16)).group_by("round").agg(
        pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort("round")
    out(f"control rounds 6..16, {g} (round, rows, disagree):", c.rows())

# 4. header final totals (independent channel): raw vs flip13 vs claimed {13,25,28}
out("\n[4] team-A (starter-T) total vs header T-starter final, contiguous and header-sum-consistent regime-A matches")
agg = R.group_by("match_key").agg(pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"),
                                 pl.col("round").n_unique().alias("nr"))
agg = agg.with_columns(contig=(pl.col("rmin") == 1) & (pl.col("nr") == pl.col("rmax")),
                       mid=pl.col("match_key").str.split("/").list.get(4))
agg = (agg.join(hdr, left_on="mid", right_on="match_id", how="left")
       .with_columns(fsum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"))
       .join(M.select(["match_key", "regime"]), on="match_key", how="left"))
ok = agg.filter(pl.col("contig") & (pl.col("fsum") == pl.col("rmax")) & (pl.col("regime") == "A"))
RS = R.join(ok.select(["match_key", "rmax", "t_starters_score_final"]), on="match_key", how="inner")
RS = RS.with_columns(nsw=sum([(pl.col("round") > b).cast(pl.Int64) for b in S_SWAPS]))
RS = RS.with_columns(sideA=pl.when(pl.col("nsw") % 2 == 0).then(2).otherwise(3))
RS = RS.with_columns(w=(pl.col("winner") == pl.col("sideA")).cast(pl.Int64))
RS = RS.with_columns(wf13=pl.when(pl.col("round") == 13).then(1 - pl.col("w")).otherwise(pl.col("w")),
                     wfc=pl.when(pl.col("round").is_in([13, 25, 28])).then(1 - pl.col("w")).otherwise(pl.col("w")))
TT = RS.group_by("match_key").agg(pl.col("w").sum().alias("raw"), pl.col("wf13").sum().alias("f13"),
                                  pl.col("wfc").sum().alias("fc"),
                                  pl.col("t_starters_score_final").first().alias("t"),
                                  pl.col("rmax").first().alias("rmax"))
for lo, hi, name in [(18, 24, "regulation 18..24"), (30, 30, "overtime 30"), (13, 17, "short 13..17"),
                     (2, 12, "short 2..12")]:
    s = TT.filter(pl.col("rmax").is_between(lo, hi))
    out(f"   {name:18s} n={s.height:5d} raw_exact={int((s['raw'] == s['t']).sum()):5d} "
        f"flip13_exact={int((s['f13'] == s['t']).sum()):5d} "
        f"claim13_25_28_exact={int((s['fc'] == s['t']).sum()):5d}")

# 5. per-match step-disagreement sets vs header fit, regulation/overtime
out("\n[5] per-match step-disagreement set (regime A, header-validated, rmax 18..30)")
sel = ok.filter(pl.col("rmax").is_between(18, 30)).select(["match_key", "rmax", "t_starters_score_final"])
Dset = (X.filter(pl.col("dis") == 1).select(["match_key", "round"]).unique()
        .join(sel, on="match_key", how="inner").group_by("match_key").agg(pl.col("round").sort().alias("D")))
allsel = (sel.join(Dset, on="match_key", how="left")
          .with_columns(Dstr=pl.col("D").cast(pl.List(pl.String)).list.join(",").fill_null("none")))
out("   pattern counts (Dstr, n):", allsel.group_by("Dstr").len().sort("len", descending=True).rows())

# 6. round-9 step disagreements in regime A: rmax of those matches
out("\n[6] regime A matches with a round-9 step disagreement: rmax range")
b9 = X.filter((pl.col("round") == 9) & (pl.col("dis") == 1)).select("match_key").unique()
rmax_all = R.group_by("match_key").agg(pl.col("round").max().alias("rmax_all"))
A9 = (M.filter(pl.col("regime") == "A").join(rmax_all, on="match_key", how="left")
      .with_columns(h9=pl.col("match_key").is_in(b9["match_key"].implode())))
h9 = A9.filter(pl.col("h9"))
out("   count:", h9.height, " rmax range:", h9["rmax_all"].min(), "..", h9["rmax_all"].max())

# 7. where do the non-13 reason-vs-bomb conflicts in regime A fall (overtime 25 / 28 extension check)
out("\n[7] regime A reason-vs-bomb conflicts by round (rounds other than 13):")
conf = (Q.filter((pl.col("regime") == "A") & (pl.col("round") != 13) & (pl.col("reason_bomb_conflict") == 1))
        .group_by("round").len().sort("round"))
out("   (round, conflicts):", conf.rows())
conf_all = Q.filter((pl.col("regime") == "A") & (pl.col("reason_bomb_conflict") == 1)).group_by("round").len().sort("round")
out("   all regime-A conflicts by round:", conf_all.rows())

# 8. short-group fit: header-consistent contiguous regime-A matches with rmax 13..17
out("\n[8] short group (rmax 13..17, contiguous, header-sum-consistent): fit of raw vs step-disagreement set")
Dall = (X.filter(pl.col("dis") == 1).select(["match_key", "round"]).unique()
        .group_by("match_key").agg(pl.col("round").sort().alias("D")))
short = (agg.with_columns(sumok=pl.col("fsum") == pl.col("rmax"))
         .filter(pl.col("contig") & pl.col("sumok") & (pl.col("regime") == "A") & pl.col("rmax").is_between(13, 17))
         .select(["match_key", "rmax", "t_starters_score_final"]).join(Dall, on="match_key", how="left")
         .with_columns(D=pl.col("D").fill_null(pl.lit([], dtype=pl.List(pl.Int64)))))
RSs = R.join(short.select(["match_key", "rmax", "t_starters_score_final", "D"]), on="match_key", how="inner")
RSs = RSs.with_columns(nsw=sum([(pl.col("round") > b).cast(pl.Int64) for b in S_SWAPS]))
RSs = RSs.with_columns(sideA=pl.when(pl.col("nsw") % 2 == 0).then(2).otherwise(3))
RSs = RSs.with_columns(w=(pl.col("winner") == pl.col("sideA")).cast(pl.Int64),
                       inD=pl.col("D").list.contains(pl.col("round")).fill_null(False))
RSs = RSs.with_columns(wd=pl.when(pl.col("inD")).then(1 - pl.col("w")).otherwise(pl.col("w")))
TS = RSs.group_by("match_key").agg(pl.col("w").sum().alias("raw"), pl.col("wd").sum().alias("fixD"),
                                   pl.col("t_starters_score_final").first().alias("t"), pl.col("rmax").first().alias("rmax"),
                                   pl.col("D").first().alias("D"))
TS = TS.with_columns(Dstr=pl.col("D").cast(pl.List(pl.String)).list.join(",").fill_null("none"))
out("   n:", TS.height, " raw_fit:", int((TS["raw"] == TS["t"]).sum()), " fixD_fit:", int((TS["fixD"] == TS["t"]).sum()))
out("   (Dstr, n, raw_fit, fixD_fit):", TS.group_by("Dstr").agg(pl.len().alias("n"), (pl.col("raw") == pl.col("t")).sum().alias("raw_fit"),
                                                             (pl.col("fixD") == pl.col("t")).sum().alias("fixD_fit"))
    .sort("n", descending=True).head(10).rows())
_LOG_FH.close()
