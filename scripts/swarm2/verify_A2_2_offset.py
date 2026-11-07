# verify_A2_2_offset.py
# Independent verifier for finding A2-2: in the 2024-02-g tome, round_start-family labels are one lower than
# round_end / round_freeze_end for the same round; the 2026 tome is aligned.
# Reads tome parquet only (read-only). Prints results; the shell redirects stdout to logs\swarm2\verify_A2_2.log.
# Memory: lazy scan with column projection per page; largest frame is round_state (~1.0M rows x 6 cols).
import os
import json

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)

ROOT = r"G:\pureskill\csds-cs2"
T24 = os.path.join(ROOT, "tomes", "2024-02-g", "tome", "csds")
T26 = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
GROUPS = ["01", "tail02_07", "tail12_25"]


def say(*a):
    print(*a, flush=True)


def chdir(T, ch):
    c = [d for d in os.listdir(T) if d.split(".", 1)[0] == "basic_" + ch]
    return os.path.join(T, c[0]) if c else None


def load(T, ch, cols, tag24=True):
    dp = chdir(T, ch)
    if dp is None:
        return None
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = []
    for p in meta["pages"]:
        lf = pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"]))
        names = lf.collect_schema().names()
        parts.append(lf.select([c for c in cols if c in names]).collect())
    df = pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
    df = df.with_row_index("rid").with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
    if tag24:
        grp = (pl.when(pl.col("day") == "01").then(pl.lit("01"))
               .when(pl.col("day") <= "07").then(pl.lit("tail02_07"))
               .otherwise(pl.lit("tail12_25")))
    else:
        grp = pl.lit("2026")
    return df.with_columns(grp.alias("grp"))


def prev_start(left, starts):
    # latest round_start with tick <= left.tick, same match (backward as-of join)
    return left.sort("tick").join_asof(starts, left_on="tick", right_on="s_tick", by="match_key", strategy="backward")


def starts_of(rs):
    return rs.select(["match_key", pl.col("round").alias("s_round"), pl.col("tick").alias("s_tick")]).sort("s_tick")


def uniq_first(df, cols):
    return df.sort("tick").unique(cols, keep="first", maintain_order=True)


def gap_quant(X, label):
    for g in GROUPS + ["2026"]:
        v = X.filter((pl.col("grp") == g) & pl.col("gap").is_not_null())["gap"].to_numpy()
        if len(v) == 0:
            continue
        say(f"  {label} {g}: n={len(v)} min={v.min()} p01={np.percentile(v, 1):.0f} median={np.median(v):.0f} "
            f"p99={np.percentile(v, 99):.0f} max={v.max()}")


# ---------------------------------------------------------------- load 2024
say("== load 2024-02-g tome (projected columns only) ==")
rs = load(T24, "round_start", ["match_key", "round", "tick"])
re_ = load(T24, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
st = load(T24, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
bs = load(T24, "bomb_state", ["match_key", "round", "tick", "event_type"])
ps = load(T24, "player_spawn", ["match_key", "round", "tick"])
pdth = load(T24, "player_death", ["match_key", "round", "tick"])
pinf = load(T24, "player_info", ["match_key", "round", "tick", "player_id", "team_code"])
pdc = load(T24, "player_disconnect", ["match_key", "round", "tick"])
say("rows: round_start", rs.height, "round_end", re_.height, "round_state", st.height, "bomb_state", bs.height,
    "player_spawn", ps.height, "player_death", pdth.height, "player_info", pinf.height, "player_disconnect", pdc.height)
say("player_info columns:", pinf.columns)
say("matches: round_end", re_["match_key"].n_unique(), "| round_start", rs["match_key"].n_unique())
rs_s = starts_of(rs)

# ---------------------------------------------------------------- B. round_end vs latest round_start (A2 key number)
say("\n== B. round_end label vs latest round_start label (claim: d = +1 for all 81,020) ==")
A = prev_start(re_, rs_s).with_columns((pl.col("round") - pl.col("s_round")).alias("d"))
F = re_.sort("tick").join_asof(rs_s.rename({"s_round": "f_round", "s_tick": "f_tick"}), left_on="tick",
                               right_on="f_tick", by="match_key", strategy="forward").select(["rid", "f_round", "f_tick"])
A = A.join(F, on="rid", how="left")
ok = A.filter(pl.col("s_round").is_not_null())
pre = A.filter(pl.col("s_round").is_null())
say("round_end rows:", A.height, "| with preceding round_start:", ok.height, "| pre-start:", pre.height)
say("pre-start labels (label:count):", pre.group_by("round").len().sort("round").rows())
say("d histogram (own recompute):", ok.group_by("d").len().sort("d").rows())
say("d histogram by group:", ok.group_by(["grp", "d"]).len().sort(["grp", "d"]).rows())
say("matches with any d != 1:", ok.filter(pl.col("d") != 1)["match_key"].n_unique())
say("tick ties round_end vs round_start:", ok.filter(pl.col("s_tick") == pl.col("tick")).height)
ub = ok.with_columns(pl.when(pl.col("f_round").is_null()).then(pl.lit("no_next_start"))
                     .when(pl.col("f_round") == pl.col("round")).then(pl.lit("next_start_is_L"))
                     .otherwise(pl.lit("other")).alias("ub"))
say("upper bound (next round_start label):", ub.group_by("ub").len().rows())
say("pre-start rows with label >= 2 (match, label, tick, first round_start label/tick):")
odd = pre.filter(pl.col("round") >= 2)
first_rs = rs.sort("tick").group_by("match_key").agg(pl.col("round").first().alias("first_s_label"),
                                                      pl.col("tick").first().alias("first_s_tick"), pl.len().alias("n_rs"))
for r in odd.join(first_rs, on="match_key", how="left").select(["match_key", "round", "tick", "first_s_label", "first_s_tick", "n_rs"]).iter_rows(named=True):
    say("  ", r)

# ---------------------------------------------------------------- C. freeze_end vs round_start (ordering / gap test)
say("\n== C. round_freeze_end(k) vs latest round_start (ordering and gap) ==")
fe = uniq_first(st.filter(pl.col("event_type") == "round_freeze_end"), ["match_key", "round"])
FE = prev_start(fe, rs_s).with_columns([(pl.col("round") - pl.col("s_round")).alias("d"),
                                        (pl.col("tick") - pl.col("s_tick")).alias("gap")])
say("freeze_end rows (unique match,label):", FE.height, "| no preceding round_start:", FE.filter(pl.col("s_round").is_null()).height)
say("d = freeze_end label - latest round_start label:", FE.filter(pl.col("s_round").is_not_null()).group_by("d").len().sort("d").rows())
say("gap ticks (freeze_end tick - latest round_start tick), ASSUMPTION tick rate 64 Hz for seconds:")
gap_quant(FE, "2024")
say("top gap values:", FE.group_by("gap").len().sort("len", descending=True).head(5).rows())
alt_fe = (fe.select(["match_key", "round", "tick"])
          .join(rs.select(["match_key", "round", pl.col("tick").alias("rs_tick")]), on=["match_key", "round"], how="inner")
          .filter(pl.col("rs_tick") < pl.col("tick")).select(["match_key", "round", "tick"]).unique())
say("ALT test: freeze_end(k) rows with a round_start labelled k BEFORE them (labels-correct hypothesis):", alt_fe.height, "of", fe.height)
alt_re = (re_.select(["match_key", "round", "tick"])
          .join(rs.select(["match_key", "round", pl.col("tick").alias("rs_tick")]), on=["match_key", "round"], how="inner")
          .filter(pl.col("rs_tick") < pl.col("tick")).select(["match_key", "round", "tick"]).unique())
say("ALT test: round_end(k) rows with a round_start labelled k BEFORE them:", alt_re.height, "of", re_.height)

# ---------------------------------------------------------------- D. 2026 control
say("\n== D. 2026 tome control ==")
rs6 = load(T26, "round_start", ["match_key", "round", "tick"], tag24=False)
re6 = load(T26, "round_end", ["match_key", "round", "tick"], tag24=False)
st6 = load(T26, "round_state", ["match_key", "round", "tick", "event_type"], tag24=False)
rs6_s = starts_of(rs6)
fe6 = uniq_first(st6.filter(pl.col("event_type") == "round_freeze_end"), ["match_key", "round"])
FE6 = prev_start(fe6, rs6_s).with_columns([(pl.col("round") - pl.col("s_round")).alias("d"),
                                           (pl.col("tick") - pl.col("s_tick")).alias("gap")])
say("2026 freeze_end rows:", FE6.height, "d histogram:", FE6.filter(pl.col("s_round").is_not_null()).group_by("d").len().sort("d").rows())
gap_quant(FE6, "2026")
A6 = prev_start(re6, rs6_s).with_columns((pl.col("round") - pl.col("s_round")).alias("d"))
say("2026 round_end rows:", A6.height, "d histogram:", A6.filter(pl.col("s_round").is_not_null()).group_by("d").len().sort("d").rows(),
    "pre-start:", A6.filter(pl.col("s_round").is_null()).height)
alt6 = (fe6.select(["match_key", "round", "tick"])
        .join(rs6.select(["match_key", "round", pl.col("tick").alias("rs_tick")]), on=["match_key", "round"], how="inner")
        .filter(pl.col("rs_tick") < pl.col("tick")).select(["match_key", "round", "tick"]).unique())
say("2026 freeze_end(k) with round_start labelled k before it:", alt6.height, "of", fe6.height)

# ---------------------------------------------------------------- E. score ledger (claim's cross-check)
say("\n== E. score ledger: freeze_end(k) -> freeze_end(k+1) delta vs round_end(k) winner (2024) ==")
FEu = fe.select(["match_key", "round", "t_score", "ct_score"]).sort(["match_key", "round"])
nx = FEu.with_columns([pl.col("round").shift(-1).over("match_key").alias("r_next"),
                       pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
                       pl.col("ct_score").shift(-1).over("match_key").alias("ct_next")])
nx = nx.filter(pl.col("r_next") == pl.col("round") + 1).with_columns([
    (pl.col("t_next") - pl.col("t_score")).alias("dT"), (pl.col("ct_next") - pl.col("ct_score")).alias("dCT")])
nx = nx.with_columns(pl.when(pl.col("dT") == 1).then(2).when(pl.col("dCT") == 1).then(3).otherwise(0).alias("w_score"))
REu = uniq_first(re_, ["match_key", "round"]).select(["match_key", "round", "winner_team_code", "grp"])
J = nx.join(REu.select(["match_key", "round", pl.col("winner_team_code").alias("w_end_k")]), on=["match_key", "round"], how="left")
J = J.join(REu.select(["match_key", pl.col("round").alias("r_next"), pl.col("winner_team_code").alias("w_end_k1")]),
           on=["match_key", "r_next"], how="left")
J = J.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day")).with_columns(
    pl.when(pl.col("day") == "01").then(pl.lit("01")).when(pl.col("day") <= "07").then(pl.lit("tail02_07"))
    .otherwise(pl.lit("tail12_25")).alias("grp"))
nz = J.filter(pl.col("w_score") != 0)
say("consecutive freeze_end pairs:", J.height, "| zero-delta pairs (excluded by claim):", J.height - nz.height)
ak = nz.filter(pl.col("w_score") == pl.col("w_end_k")).height
ak1 = nz.filter(pl.col("w_score") == pl.col("w_end_k1")).height
say(f"nonzero pairs: {nz.height} | agree with round_end(k): {ak} ({100*ak/nz.height:.2f}%) | agree with round_end(k+1): {ak1} ({100*ak1/nz.height:.2f}%)")
say(f"all consecutive pairs agree with round_end(k): {J.filter(pl.col('w_score') == pl.col('w_end_k')).height} of {J.height}")
for g in GROUPS:
    s = nz.filter(pl.col("grp") == g)
    a = s.filter(pl.col("w_score") == pl.col("w_end_k")).height
    inv = s.filter((pl.col("w_score") != pl.col("w_end_k")) & (pl.col("w_end_k") == 5 - pl.col("w_score"))).height
    say(f"  {g}: n={s.height} agree_k={a} ({100*a/max(s.height,1):.2f}%) disagree={s.height-a} of which side-flipped={inv}")
dis = nz.filter(pl.col("w_score") != pl.col("w_end_k")).group_by(["grp", "round"]).len().rename({"len": "bad"})
tot = nz.group_by(["grp", "round"]).len().rename({"len": "n"})
bk = tot.join(dis, on=["grp", "round"], how="left").with_columns(pl.col("bad").fill_null(0)).sort(["grp", "round"])
say("disagreement by group and round label (rows with bad>0):")
say(bk.filter(pl.col("bad") > 0).select(["grp", "round", "n", "bad"]).rows())

# ---------------------------------------------------------------- F. channel conventions vs round_start windows
say("\n== F. 2024 channel label convention: d = event label - latest round_start label ==")
def conv(name, df):
    X = prev_start(df, rs_s).with_columns((pl.col("round") - pl.col("s_round")).alias("d"))
    say(f"{name}: rows={X.height} d counts={X.group_by('d').len().sort('d').rows()}")
    say(f"   by group d: {X.group_by(['grp', 'd']).len().sort(['grp', 'd']).rows()}")
    return X
conv("player_death", pdth)
conv("player_spawn", ps)
conv("player_disconnect", pdc)
Xst = prev_start(st, rs_s).with_columns((pl.col("round") - pl.col("s_round")).alias("d"))
say("round_state by event_type (d counts):")
for et, sub in Xst.group_by("event_type"):
    say("  ", et[0], sub.group_by("d").len().sort("d").rows())
Xbs = prev_start(bs, rs_s).with_columns((pl.col("round") - pl.col("s_round")).alias("d"))
say("bomb_state by event_type (d counts):")
for et, sub in Xbs.group_by("event_type"):
    say("  ", et[0], sub.group_by("d").len().sort("d").rows())
if "tick" in pinf.columns:
    Xpi = prev_start(pinf, rs_s).with_columns((pl.col("round") - pl.col("s_round")).alias("d"))
    say(f"player_info: rows={Xpi.height} d counts={Xpi.group_by('d').len().sort('d').rows()}")
    say(f"   by group d: {Xpi.group_by(['grp', 'd']).len().sort(['grp', 'd']).rows()}")

# player_info label coverage vs round_end labels (round-number convention would match)
say("\n== G. player_info label coverage vs round_end label (bake-off uses PI labels for sides) ==")
pi_rounds = pinf.select(["match_key", "round"]).unique()
re_rounds = re_.select(["match_key", "round"]).unique()
say("round_end (match,label) with no player_info row:", re_rounds.join(pi_rounds, on=["match_key", "round"], how="anti").height)
say("player_info (match,label) with no round_end row:", pi_rounds.join(re_rounds, on=["match_key", "round"], how="anti").height)
say("player_info labels with no round_end row (label:count):",
    pi_rounds.join(re_rounds, on=["match_key", "round"], how="anti").group_by("round").len().sort("round").rows()[:8])
say("round_end (match,label) with no player_info row (label:count, first 8):",
    re_rounds.join(pi_rounds, on=["match_key", "round"], how="anti").group_by("round").len().sort("round").rows()[:8])
pmm = pinf.group_by("match_key").agg(pl.col("round").min().alias("pmin"), pl.col("round").max().alias("pmax"))
rmm = re_.group_by("match_key").agg(pl.col("round").max().alias("rmax"))
mm = rmm.join(pmm, on="match_key", how="left").with_columns((pl.col("pmax") - pl.col("rmax")).alias("x"))
say("player_info max label minus round_end max label:", mm.group_by("x").len().sort("x").rows())
say("player_info min label histogram:", pmm.group_by("pmin").len().sort("pmin").rows())

# ---------------------------------------------------------------- H. round 1 and the proposed key shift
say("\n== H. round 1 and the proposed rst_keys shift (+1) ==")
say("round_start labels present (min label):", rs["round"].min(), "| round_start rows with label 0:", rs.filter(pl.col("round") == 0).height)
for g in GROUPS:
    m_re1 = re_.filter((pl.col("grp") == g) & (pl.col("round") == 1))["match_key"].n_unique()
    m_fe1 = fe.filter((pl.col("grp") == g) & (pl.col("round") == 1))["match_key"].n_unique()
    m_all = re_.filter(pl.col("grp") == g)["match_key"].n_unique()
    say(f"  {g}: matches={m_all} round_end(1) present={m_re1} freeze_end(1) present={m_fe1}")
cand = re_.select(["match_key", "round"]).unique().join(fe.select(["match_key", "round", "grp"]), on=["match_key", "round"], how="inner")
rst_now = rs.select(["match_key", "round"]).unique()
rst_shift = rs.select(["match_key", (pl.col("round") + 1).alias("round")]).unique()
ex_now = cand.join(rst_now, on=["match_key", "round"], how="anti")
ex_shift = cand.join(rst_shift, on=["match_key", "round"], how="anti")
say("candidate rounds (round_end and freeze_end both present):", cand.height)
say("excluded by current rst_keys:", ex_now.height, "| label 1 excluded:", ex_now.filter(pl.col("round") == 1).height)
say("excluded by shifted rst_keys (+1):", ex_shift.height, "| label 1 excluded:", ex_shift.filter(pl.col("round") == 1).height)
say("  shifted exclusions by group:", ex_shift.group_by("grp").len().sort("grp").rows())
last = cand.group_by("match_key").agg(pl.col("round").max().alias("last"))
ex_now_last = ex_now.join(last, on="match_key").filter(pl.col("round") == pl.col("last")).height
say("current exclusions that are the last round of their match:", ex_now_last, "of", ex_now.height)

# ---------------------------------------------------------------- I. round 13 bomb-side check (e2 reproduction) and the label question
say("\n== I. round 13 bomb outcome vs round_end winner (e2 method, player_count == 20) ==")
outcome = (bs.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"])
           .agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("bomb_tick"), pl.col("grp").first())
           .with_columns([pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
                          pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit("defused")).otherwise(pl.lit("exploded")).alias("bomb_name")]))
J13 = (re_.filter(pl.col("player_count") == 20)
       .join(outcome.drop("grp"), on=["match_key", "round"], how="inner")
       .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"),
                     pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
                     .otherwise(pl.lit("other")).alias("rbin")))
say("rounds with bomb outcome:", J13.height, "| mismatches:", J13.filter(pl.col("mm")).height)
say("mismatch by round bin:", J13.group_by("rbin").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("rbin").rows())
r13 = J13.filter(pl.col("round") == 13)
say("round 13 by group (n, mm):", r13.group_by("grp").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("grp").rows())
r13t = r13.filter(pl.col("grp") == "tail12_25")
say("round 13 tail12_25 n:", r13t.height, "mm:", r13t.filter(pl.col("mm")).height)
# within-window: is the bomb event inside round 13's window (d = 1) and how far from round_end?
bw = r13.with_columns(pl.col("bomb_tick").alias("tick")).sort("tick").join_asof(rs_s, left_on="tick", right_on="s_tick", by="match_key", strategy="backward")
bw = bw.with_columns((pl.col("round") - pl.col("s_round")).alias("d_bomb"))
say("round 13 bomb event window d (1 = inside round 13 window):", bw.group_by(["mm", "d_bomb"]).len().sort(["mm", "d_bomb"]).rows())
say("round 13 gap round_end tick - bomb tick, mismatched rows (quantiles):",
    np.percentile(r13.filter(pl.col("mm"))["tick"].to_numpy() - r13.filter(pl.col("mm"))["bomb_tick"].to_numpy(), [5, 50, 95]).tolist())
# does the round_end row's own win_reason agree with its winner, and with the bomb event?
WT, WCT = [1, 9], [7, 8, 12]
RE_ok = re_.with_columns(pl.when(pl.col("win_reason_code").is_in(WT)).then(2)
                         .when(pl.col("win_reason_code").is_in(WCT)).then(3).otherwise(0).alias("w_reason"))
say("round_end rows: winner_team_code == winner implied by win_reason (2 for {1,9}, 3 for {7,8,12}):",
    RE_ok.filter(pl.col("w_reason") == pl.col("winner_team_code")).height, "of", RE_ok.height,
    "| other reason codes:", RE_ok.filter(pl.col("w_reason") == 0).group_by("win_reason_code").len().rows())
ct = (J13.filter((pl.col("round") == 13) & (pl.col("grp") == "tail12_25"))
      .group_by(["bomb_name", "win_reason_code", "winner_team_code"]).len().sort(["bomb_name", "win_reason_code", "winner_team_code"]))
say("round 13 tail12_25 crosstab bomb_name x win_reason_code x winner_team_code (n):", ct.rows())

say("\nDONE")
