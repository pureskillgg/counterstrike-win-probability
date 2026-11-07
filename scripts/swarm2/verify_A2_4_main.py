# verify_A2_4_main.py (v2)
# Independent recomputation of finding A2-4: "round 1 (freeze_end and round_end labels) is missing for all
# 1,957 build-9842 matches; the round-1 winner can be recovered from the score at freeze_end(2)".
# Read-only. Lazy parquet scans, column projection. Prints numbers only.
import os, json
import polars as pl

pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_width_chars(250)

TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
UUID = r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"


def sec(t):
    print("\n==== " + t + " ====", flush=True)


def channel_files(ch):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    assert len(d) == 1, (ch, d)
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    return [os.path.join(dp, p["dataframe"]["dataframe"]) for p in meta["pages"]]


def lazy(ch, cols):
    return pl.concat([pl.scan_parquet(f).select(cols) for f in channel_files(ch)], how="vertical_relaxed")


def dgrp(d):
    if d is None:
        return "NA"
    if d == "01":
        return "d01"
    if d <= "07":
        return "d02_07"
    if d <= "11":
        return "d08_11"
    return "d12_25"


# ---------------- 1. header, key mapping ----------------
sec("1. header and key mapping")
hdr = (pl.scan_parquet(os.path.join(TOME, "header", "dataframe_00000"))
       .select(["key", "match_id", "build_num", "map_name", "match_date", "tick_rate"]).collect())
hdr = hdr.with_columns(pl.col("key").str.extract(UUID, 1).alias("u_key"),
                       pl.col("match_id").str.extract(UUID, 1).alias("u_mid"))
keys = pl.concat([lazy(ch, ["match_key"]).unique() for ch in ["round_state", "round_start", "round_end"]]).unique().collect()
keys = keys.with_columns(pl.col("match_key").str.extract(UUID, 1).alias("u"),
                         pl.col("match_key").str.split("/").list.get(3).alias("day"))
print("header rows:", hdr.height, " distinct match_key over round channels:", keys.height,
      " uuid unique:", keys["u"].n_unique(), flush=True)
ov_key = len(set(hdr["u_key"].drop_nulls()) & set(keys["u"]))
ov_mid = len(set(hdr["u_mid"].drop_nulls()) & set(keys["u"]))
print("overlap header.key uuid:", ov_key, " overlap header.match_id uuid:", ov_mid, flush=True)
use = "u_key" if ov_key >= ov_mid else "u_mid"
hdr = hdr.rename({use: "u"})
mp = hdr.join(keys.select(["u", "match_key", "day"]), on="u", how="left")
print("header rows with a round-channel match_key:", mp.filter(pl.col("match_key").is_not_null()).height, " of", mp.height, flush=True)
um_hdr = mp.filter(pl.col("match_key").is_null())
print("header rows WITHOUT round data:", um_hdr.select(["key", "match_id", "build_num", "match_date", "map_name"]).rows(), flush=True)
um_keys = keys.filter(~pl.col("u").is_in(hdr["u"].drop_nulls().to_list()))
print("round-channel match_keys WITHOUT a header row:", um_keys.height, um_keys.select(["match_key", "day"]).rows(), flush=True)
mp = mp.filter(pl.col("match_key").is_not_null()).with_columns(
    pl.col("build_num").cast(pl.Utf8).alias("build"),
    pl.col("day").map_elements(dgrp, return_dtype=pl.Utf8).alias("dgrp"))
print("build_num counts (mapped):", mp.group_by("build").len().sort("build").rows(), flush=True)
print("build x day-group:", mp.group_by(["build", "dgrp"]).len().sort(["build", "dgrp"]).rows(), flush=True)
print("build x folder-day:", mp.group_by(["build", "day"]).len().sort(["build", "day"]).rows(), flush=True)
print("tick_rate by build:", mp.group_by(["build", "tick_rate"]).len().sort(["build", "tick_rate"]).rows(), flush=True)
mp = mp.with_columns(pl.col("match_date").str.slice(0, 10).alias("mdate"))
print("match_date range by build:", mp.group_by("build").agg(pl.col("mdate").min().alias("min_date"), pl.col("mdate").max().alias("max_date")).sort("build").rows(), flush=True)
print("folder day == match_date day:", mp.filter(pl.col("mdate").str.slice(8, 2) == pl.col("day")).height, "of", mp.height, flush=True)
MK = mp.select(["match_key", "build", "dgrp", "day", "tick_rate"])

# ---------------- 2. round channels: census, nulls, duplicates ----------------
sec("2. round_state event census; round<=1 rows; nulls and duplicates")
ST_COLS = ["round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase", "is_warmup", "match_key"]
st = lazy("round_state", ST_COLS).collect()
print("round_state event_type counts:", st.group_by("event_type").len().sort("len", descending=True).rows(), flush=True)
print("round_state rows:", st.height, " null round:", st["round"].null_count(), flush=True)
st = st.join(MK.select(["match_key", "build"]), on="match_key", how="left")
print("round_state rows with no build:", st.filter(pl.col("build").is_null()).height, flush=True)
low = (st.filter(pl.col("round") <= 1)
       .group_by(["build", "event_type", "round"]).agg(pl.len().alias("rows"), pl.col("match_key").n_unique().alias("matches"))
       .sort(["build", "event_type", "round"]))
print("round_state rows with round<=1, by build/event/round (build, event, round, rows, matches):", low.rows(), flush=True)
fe_raw = st.filter(pl.col("event_type") == "round_freeze_end").drop(["event_type"])
rs_raw = lazy("round_start", ["round", "tick", "match_key"]).collect()
re_raw = lazy("round_end", ["round", "tick", "winner_team_code", "win_reason_code", "match_key"]).collect()
rs_raw = rs_raw.join(MK.select(["match_key", "build"]), on="match_key", how="left")
re_raw = re_raw.join(MK.select(["match_key", "build"]), on="match_key", how="left")
for name, df in [("round_start", rs_raw), ("round_end", re_raw), ("freeze_end", fe_raw)]:
    dd = df.drop_nulls("round").group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
    print(f"{name}: rows={df.height} null_round={df['round'].null_count()} round<=0={(df['round'] <= 0).sum()} "
          f"dup(match,round) groups={dd.height} distinct(match,round)={df.drop_nulls('round').unique(['match_key','round']).height}", flush=True)
    if dd.height:
        print("   duplicate groups:", dd.rows(), flush=True)
        for mk_, r_ in dd.select(["match_key", "round"]).rows():
            sub = df.filter((pl.col("match_key") == mk_) & (pl.col("round") == r_))
            print("     ", sub.drop(["match_key"]).rows(), " build=", sub["build"][0], flush=True)
print("round_end winner_team_code by build:", re_raw.group_by(["build", "winner_team_code"]).len().sort(["build", "winner_team_code"]).rows(), flush=True)
print("round_end label 1/2 by build, win_reason_code:", re_raw.filter(pl.col("round") <= 2)
      .group_by(["build", "round", "win_reason_code"]).len().sort(["build", "round", "win_reason_code"]).rows(), flush=True)

# dedup as the bake-off/A2 scripts do (keep first) but report what was dropped
fe = fe_raw.unique(["match_key", "round"], keep="first", maintain_order=True)
re_ = re_raw.unique(["match_key", "round"], keep="first", maintain_order=True)
rs = rs_raw.unique(["match_key", "round"], keep="first", maintain_order=True)
print("after dedup (keep first): freeze_end", fe.height, " round_end", re_.height, " round_start", rs.height, flush=True)

# ---------------- 3. per-match label ranges ----------------
sec("3. per-match label ranges by build")


def agg(df, nm):
    return (df.group_by("match_key")
            .agg(pl.col("round").min().alias(nm + "_min"), pl.col("round").max().alias(nm + "_max"),
                 pl.col("round").n_unique().cast(pl.Int64).alias(nm + "_n")))


M = MK.join(agg(fe, "fe"), on="match_key", how="left").join(agg(re_, "re"), on="match_key", how="left").join(agg(rs, "rs"), on="match_key", how="left")
M = M.with_columns(
    ((pl.col("fe_n") == pl.col("fe_max") - pl.col("fe_min") + 1)).alias("fe_contig"),
    ((pl.col("re_n") == pl.col("re_max") - pl.col("re_min") + 1)).alias("re_contig"),
    ((pl.col("rs_n") == pl.col("rs_max") - pl.col("rs_min") + 1)).alias("rs_contig"),
)
print("matches by build x day-group:", M.group_by(["build", "dgrp"]).len().sort(["build", "dgrp"]).rows(), flush=True)
print("(build, fe_min, re_min, rs_min) -> matches:",
      M.group_by(["build", "fe_min", "re_min", "rs_min"]).len().sort(["build", "fe_min", "re_min", "rs_min"]).rows(), flush=True)
for nm in ["fe", "re", "rs"]:
    print(f"{nm} label 1 present (build, has1, matches):",
          M.with_columns((pl.col(f"{nm}_min") == 1).alias("has1")).group_by(["build", "has1"]).len().sort(["build", "has1"]).rows(), flush=True)
print("label-set gaps: (build, fe_gap, re_gap, rs_gap) -> matches:",
      M.with_columns((~pl.col("fe_contig")).alias("fg"), (~pl.col("re_contig")).alias("rg"), (~pl.col("rs_contig")).alias("sg"))
      .group_by(["build", "fg", "rg", "sg"]).len().sort(["build", "fg", "rg", "sg"]).rows(), flush=True)
gap_m = M.filter(~pl.col("fe_contig") | ~pl.col("re_contig") | ~pl.col("rs_contig")).select(["match_key", "build", "fe_min", "fe_max", "fe_n", "re_min", "re_max", "re_n", "rs_min", "rs_max", "rs_n"])
print("matches with a gap in a label set:", gap_m.rows(), flush=True)
for mk_ in gap_m["match_key"].to_list():
    print("   fe labels", sorted(fe.filter(pl.col("match_key") == mk_)["round"].to_list()),
          " re labels", sorted(re_.filter(pl.col("match_key") == mk_)["round"].to_list()),
          " rs labels", sorted(rs.filter(pl.col("match_key") == mk_)["round"].to_list()), flush=True)
print("rs_n - fe_n, rs_n - re_n by build (Int64):",
      M.with_columns((pl.col("rs_n") - pl.col("fe_n")).alias("d_rs_fe"), (pl.col("rs_n") - pl.col("re_n")).alias("d_rs_re"))
      .group_by(["build", "d_rs_fe", "d_rs_re"]).len().sort(["build", "d_rs_fe", "d_rs_re"]).rows(), flush=True)
M9842 = M.filter(pl.col("build") == "9842")
print("build 9842 matches:", M9842.height, " by dgrp:", M9842.group_by("dgrp").len().sort("dgrp").rows(), flush=True)
print("build 9842: round_start label 1 present:", M9842.filter(pl.col("rs_min") == 1).height,
      " | freeze_end label 1 present:", M9842.filter(pl.col("fe_min") == 1).height,
      " | round_end label 1 present:", M9842.filter(pl.col("re_min") == 1).height, flush=True)
M9961 = M.filter(pl.col("build") == "9961")
print("build 9961 matches:", M9961.height, " | freeze_end label 1 present:", M9961.filter(pl.col("fe_min") == 1).height,
      " | round_end label 1 present:", M9961.filter(pl.col("re_min") == 1).height, " | round_start label 1:", M9961.filter(pl.col("rs_min") == 1).height, flush=True)
print("build 9961 matches with no round_start rows:", M9961.filter(pl.col("rs_min").is_null()).select(["match_key", "fe_min", "re_min"]).rows(), flush=True)

# ---------------- 4. score sums at freeze_end(k) ----------------
sec("4. score sums at freeze_end(k), processed and raw")
fe_b = fe.with_columns((pl.col("t_score") + pl.col("ct_score")).alias("sum_p"),
                       (pl.col("t_score_raw") + pl.col("ct_score_raw")).alias("sum_r"))
print("freeze_end (deduped) rows: sum_p == k-1 by build:",
      fe_b.with_columns((pl.col("sum_p") == pl.col("round") - 1).alias("ok")).group_by(["build", "ok"]).len().sort(["build", "ok"]).rows(), flush=True)
print("freeze_end (deduped) rows: sum_r == k-1 by build:",
      fe_b.with_columns((pl.col("sum_r") == pl.col("round") - 1).alias("ok")).group_by(["build", "ok"]).len().sort(["build", "ok"]).rows(), flush=True)
print("processed != raw score rows by build:",
      fe_b.with_columns(((pl.col("t_score") != pl.col("t_score_raw")) | (pl.col("ct_score") != pl.col("ct_score_raw"))).alias("diff"))
      .group_by(["build", "diff"]).len().sort(["build", "diff"]).rows(), flush=True)
print("freeze_end label range by build:", fe_b.group_by("build").agg(pl.col("round").min().alias("fmin"), pl.col("round").max().alias("fmax")).sort("build").rows(), flush=True)
print("freeze_end rounds 1-2 by build (round, phase, is_warmup, rows):",
      fe_b.filter(pl.col("round") <= 2).group_by(["build", "round", "phase", "is_warmup"]).len().sort(["build", "round", "phase", "is_warmup"]).rows(), flush=True)

# ---------------- 5. side attachment: score delta k->k+1 vs round_end label k ----------------
sec("5. score-delta winner vs round_end label k (processed and raw), by build")
fs = fe_b.sort(["match_key", "round"]).with_columns(
    pl.col("round").shift(-1).over("match_key").alias("r_next"),
    pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
    pl.col("ct_score").shift(-1).over("match_key").alias("ct_next"),
    pl.col("t_score_raw").shift(-1).over("match_key").alias("tr_next"),
    pl.col("ct_score_raw").shift(-1).over("match_key").alias("ctr_next"))
fs = fs.filter(pl.col("r_next") == pl.col("round") + 1).with_columns(
    (pl.col("t_next") - pl.col("t_score")).alias("dT"), (pl.col("ct_next") - pl.col("ct_score")).alias("dCT"),
    (pl.col("tr_next") - pl.col("t_score_raw")).alias("dTr"), (pl.col("ctr_next") - pl.col("ct_score_raw")).alias("dCTr"))
fs = fs.with_columns(pl.when(pl.col("dT") == 1).then(2).when(pl.col("dCT") == 1).then(3).otherwise(0).alias("w_p"),
                     pl.when(pl.col("dTr") == 1).then(2).when(pl.col("dCTr") == 1).then(3).otherwise(0).alias("w_r"))
fs = fs.join(re_.select(["match_key", "round", "winner_team_code"]).rename({"winner_team_code": "w_end"}), on=["match_key", "round"], how="left")
print("consecutive freeze_end pairs:", fs.height, " with round_end(k) present:", fs.filter(pl.col("w_end").is_not_null()).height, flush=True)
okp = fs.filter(pl.col("w_p") != 0).with_columns((pl.col("w_end") != pl.col("w_p")).alias("bad"))
print("processed: score-derived winner (nonzero) by build x disagree:", okp.group_by(["build", "bad"]).len().sort(["build", "bad"]).rows(), flush=True)
bk = okp.group_by(["build", "round"]).agg(pl.len().alias("n"), pl.col("bad").sum().cast(pl.Int64).alias("bad")).sort(["build", "round"])
for b in ["9842", "9961"]:
    rows = bk.filter((pl.col("build") == b) & (pl.col("round") <= 30)).rows()
    print(f"  build {b}: k with bad>0 (k, n, bad):", [(r[1], r[2], r[3]) for r in rows if r[3] > 0], flush=True)
    print(f"  build {b}: k=1..12 (k, n, bad):", [(r[1], r[2], r[3]) for r in rows if r[1] <= 12], flush=True)
okr = fs.filter(pl.col("w_r") != 0).with_columns((pl.col("w_end") != pl.col("w_r")).alias("bad"))
print("raw: score-derived winner (nonzero) by build x disagree:", okr.group_by(["build", "bad"]).len().sort(["build", "bad"]).rows(), flush=True)
print("processed vs raw delta winner differ, by build:", fs.filter(pl.col("w_p") != pl.col("w_r")).group_by("build").len().sort("build").rows(), flush=True)

# ---------------- 6. recovery test: side with 1 point at freeze_end(2) ----------------
sec("6. side with 1 point at freeze_end(2) vs round_end label 1 winner")
f2 = fe_b.filter(pl.col("round") == 2).select(["match_key", "build", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
e1 = re_.filter(pl.col("round") == 1).select(["match_key", "winner_team_code"]).rename({"winner_team_code": "w1_end"})
rec = f2.join(e1, on="match_key", how="left").with_columns(
    pl.when((pl.col("t_score") == 1) & (pl.col("ct_score") == 0)).then(2)
      .when((pl.col("ct_score") == 1) & (pl.col("t_score") == 0)).then(3).otherwise(0).alias("side_rec"),
    pl.when((pl.col("t_score_raw") == 1) & (pl.col("ct_score_raw") == 0)).then(2)
      .when((pl.col("ct_score_raw") == 1) & (pl.col("t_score_raw") == 0)).then(3).otherwise(0).alias("side_rec_raw"))
print("build x side_rec x w1_end (processed):", rec.group_by(["build", "side_rec", "w1_end"]).len().sort(["build", "side_rec", "w1_end"]).rows(), flush=True)
print("build x side_rec_raw x w1_end (raw):", rec.group_by(["build", "side_rec_raw", "w1_end"]).len().sort(["build", "side_rec_raw", "w1_end"]).rows(), flush=True)
r61 = rec.filter(pl.col("build") == "9961")
print("9961 agreement processed:", r61.filter(pl.col("side_rec") == pl.col("w1_end")).height, "of", r61.height,
      " | raw:", r61.filter(pl.col("side_rec_raw") == pl.col("w1_end")).height, flush=True)
f1 = fe_b.filter(pl.col("round") == 1)
print("9961 freeze_end(1) with t+ct==0:", f1.filter((pl.col("build") == "9961") & (pl.col("sum_p") == 0)).height, "of", f1.filter(pl.col("build") == "9961").height, flush=True)
r42 = rec.filter(pl.col("build") == "9842")
print("9842 recovered side (processed):", r42.group_by("side_rec").len().sort("side_rec").rows(), " raw:", r42.group_by("side_rec_raw").len().sort("side_rec_raw").rows(), flush=True)

# ---------------- 6b. state-level score change across round 1 events (independent of freeze_end(2)) ----------------
sec("6b. score change between first and last round_state event labelled 1 (per match)")
s1 = st.filter(pl.col("round") == 1).sort(["match_key", "tick"])
first1 = s1.group_by("match_key").first().select(["match_key", "build", "tick", "event_type", "t_score", "ct_score"]).rename(
    {"tick": "t0", "event_type": "ev0", "t_score": "t0s", "ct_score": "c0s"})
last1 = s1.group_by("match_key").last().select(["match_key", "tick", "event_type", "t_score", "ct_score"]).rename(
    {"tick": "t1", "event_type": "ev1", "t_score": "t1s", "ct_score": "c1s"})
sc = first1.join(last1, on="match_key", how="left").with_columns(
    (pl.col("t1s") - pl.col("t0s")).alias("dT1"), (pl.col("c1s") - pl.col("c0s")).alias("dC1"))
sc = sc.with_columns(pl.when((pl.col("dT1") == 1) & (pl.col("dC1") == 0)).then(2)
                     .when((pl.col("dC1") == 1) & (pl.col("dT1") == 0)).then(3).otherwise(0).alias("side_state"))
print("first/last round-1 event names by build:", sc.group_by(["build", "ev0", "ev1"]).len().sort(["build", "ev0", "ev1"]).rows(), flush=True)
print("state-side gained by build (2=T,3=CT,0=none/other):", sc.group_by(["build", "side_state"]).len().sort(["build", "side_state"]).rows(), flush=True)
cmp = sc.join(rec.select(["match_key", "side_rec", "w1_end"]), on="match_key", how="left")
print("9842: state-side vs fe(2)-side agreement:", cmp.filter(pl.col("build") == "9842").with_columns((pl.col("side_state") == pl.col("side_rec")).alias("ok")).group_by("ok").len().sort("ok").rows(), flush=True)
print("9961: state-side vs round_end(1) winner agreement:", cmp.filter(pl.col("build") == "9961").with_columns((pl.col("side_state") == pl.col("w1_end")).alias("ok")).group_by("ok").len().sort("ok").rows(), flush=True)

# ---------------- 7. tick timing ----------------
sec("7. tick timing (seconds = tick delta / tick_rate)")
rs_t = rs.select(["match_key", "build", "round", "tick"]).rename({"tick": "rs_tick"})
fe_t = fe.select(["match_key", "build", "round", "tick"]).rename({"tick": "fe_tick"})
re_t = re_.select(["match_key", "build", "round", "tick"]).rename({"tick": "re_tick"})
TR = MK.select(["match_key", "tick_rate"])


def sec_pair(left, right, lab_l, lab_r, tick_l, tick_r, label):
    a = left.filter(pl.col("round") == lab_l).select(["match_key", "build", tick_l])
    b = right.filter(pl.col("round") == lab_r).select(["match_key", "build", tick_r])
    j = a.join(b, on=["match_key", "build"], how="inner").join(TR, on="match_key", how="left")
    j = j.with_columns(((pl.col(tick_r) - pl.col(tick_l)) / pl.col("tick_rate")).alias("sec"))
    out = j.group_by("build").agg(pl.len().alias("n"), pl.col("sec").min().alias("min"),
                                   pl.col("sec").quantile(0.05).alias("p05"), pl.col("sec").median().alias("med"),
                                   pl.col("sec").quantile(0.95).alias("p95"), pl.col("sec").max().alias("max"),
                                   (pl.col("sec") < 0).sum().cast(pl.Int64).alias("neg")).sort("build")
    print(f"{label}:", [tuple(round(x, 2) if isinstance(x, float) else x for x in r) for r in out.rows()], flush=True)


sec_pair(rs_t, fe_t, 2, 2, "rs_tick", "fe_tick", "rs(2)->fe(2)  [freeze length]")
sec_pair(rs_t, fe_t, 1, 1, "rs_tick", "fe_tick", "rs(1)->fe(1)  [freeze length, 9961 only]")
sec_pair(rs_t, re_t, 1, 1, "rs_tick", "re_tick", "rs(1)->re(1)  [round length, 9961]")
sec_pair(fe_t, re_t, 1, 1, "fe_tick", "re_tick", "fe(1)->re(1)  [in-round length, 9961]")
sec_pair(re_t, fe_t, 1, 2, "re_tick", "fe_tick", "re(1)->fe(2)  [post-round to next freeze, 9961]")
sec_pair(rs_t, fe_t, 1, 2, "rs_tick", "fe_tick", "rs(1)->fe(2)  [9842 span if round 1 is real]")
sec_pair(rs_t, re_t, 2, 2, "rs_tick", "re_tick", "rs(2)->re(2)  [round length, both builds]")
sec_pair(fe_t, re_t, 2, 2, "fe_tick", "re_tick", "fe(2)->re(2)  [in-round length, both builds]")
sec_pair(re_t, fe_t, 2, 3, "re_tick", "fe_tick", "re(2)->fe(3)  [post-round, both builds]")
o = fe_t.join(re_t, on=["match_key", "build", "round"], how="inner").with_columns((pl.col("re_tick") <= pl.col("fe_tick")).alias("v"))
print("fe(k).tick < re(k).tick violations (build, v, n):", o.group_by(["build", "v"]).len().sort(["build", "v"]).rows(), flush=True)
o2 = rs_t.join(fe_t, on=["match_key", "build", "round"], how="inner").with_columns((pl.col("fe_tick") <= pl.col("rs_tick")).alias("v"))
print("rs(k).tick < fe(k).tick violations (build, v, n):", o2.group_by(["build", "v"]).len().sort(["build", "v"]).rows(), flush=True)
o3 = re_t.with_columns((pl.col("round") + 1).alias("round")).join(fe_t, on=["match_key", "build", "round"], how="inner",
                                                                    suffix="_nx").with_columns((pl.col("fe_tick") <= pl.col("re_tick")).alias("v"))
print("re(k).tick < fe(k+1).tick violations (build, v, n):", o3.group_by(["build", "v"]).len().sort(["build", "v"]).rows(), flush=True)

# ---------------- 8. event timeline of round 1 vs round 2 (round_state) ----------------
sec("8. round_state event timeline per round (seconds after round_prestart of that round)")
pre = st.filter(pl.col("event_type") == "round_prestart").sort(["match_key", "tick"]).unique(["match_key", "round"], keep="first") \
        .select(["match_key", "round", "tick"]).rename({"tick": "pre_tick"})
tl = st.filter(pl.col("round").is_in([1, 2])).join(pre, on=["match_key", "round"], how="left").join(TR, on="match_key", how="left")
tl = tl.with_columns(((pl.col("tick") - pl.col("pre_tick")) / pl.col("tick_rate")).alias("dt"))
tlsum = (tl.group_by(["build", "round", "event_type"])
         .agg(pl.col("match_key").n_unique().cast(pl.Int64).alias("matches"), pl.len().alias("rows"),
              pl.col("dt").median().alias("med_s"), pl.col("dt").min().alias("min_s"), pl.col("dt").max().alias("max_s"))
         .sort(["build", "round", "med_s"]))
for r in tlsum.rows():
    print("  ", r[0], "round", r[1], f"{r[2]:<32}", "matches", r[3], "rows", r[4],
          f"med {r[5]:.2f}s  min {r[6]:.2f}  max {r[7]:.2f}" if r[5] is not None else "", flush=True)
print("round_prestart(1) present by build:", pre.join(MK.select(["match_key", "build"]), on="match_key").filter(pl.col("round") == 1).group_by("build").len().rows(), flush=True)

# ---------------- 9. label-only checks shared with A2 scripts ----------------
sec("9. A2 bake-off-style label convention (keep first) reproduced")
print("freeze_end (deduped) distinct (match,label):", fe.height, " sum test ok:",
      fe_b.with_columns((pl.col("sum_p") == pl.col("round") - 1).alias("ok")).filter(pl.col("ok")).height, flush=True)
print("build 9842 round_end label-2 rows with win_reason:", re_.filter((pl.col("build") == "9842") & (pl.col("round") == 2)).group_by("win_reason_code").len().sort("win_reason_code").rows(), flush=True)
print("DONE", flush=True)
