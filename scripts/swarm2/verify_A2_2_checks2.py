# verify_A2_2_checks2.py - second-pass checks for finding A2-2 (run after verify_A2_2_offset.py).
# (a) freeze_end score-sum invariant; (b) round-13 label test on the score ledger (is the mismatch a label shift
# or a side inversion?); (c) last-round rows vs player_count (is the current rst_keys gate reached by last rounds?);
# (d) tick equality of round_prestart / round_poststart / round_officially_ended / round_mvp with round_start;
# (e) one raw E: match folder for the day-12 tail match, to see whether the offset is already in the source.
import os
import json

import polars as pl

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)

ROOT = r"G:\pureskill\csds-cs2"
T24 = os.path.join(ROOT, "tomes", "2024-02-g", "tome", "csds")


def say(*a):
    print(*a, flush=True)


def chdir(T, ch):
    c = [d for d in os.listdir(T) if d.split(".", 1)[0] == "basic_" + ch]
    return os.path.join(T, c[0]) if c else None


def load(T, ch, cols):
    dp = chdir(T, ch)
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = []
    for p in meta["pages"]:
        lf = pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"]))
        names = lf.collect_schema().names()
        parts.append(lf.select([c for c in cols if c in names]).collect())
    df = pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
    df = df.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
    return df.with_columns(
        pl.when(pl.col("day") == "01").then(pl.lit("01"))
        .when(pl.col("day") <= "07").then(pl.lit("tail02_07"))
        .otherwise(pl.lit("tail12_25")).alias("grp"))


rs = load(T24, "round_start", ["match_key", "round", "tick"])
st = load(T24, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
re_ = load(T24, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
fe = (st.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .unique(["match_key", "round"], keep="first", maintain_order=True))

# ---------------------------------------------------------------- (a) score-sum invariant
say("== (a) freeze_end: t_score + ct_score == label - 1 (completed rounds before this round) ==")
fs = fe.with_columns((pl.col("t_score") + pl.col("ct_score")).alias("sum"))
say("all groups:", fs.filter(pl.col("sum") == pl.col("round") - 1).height, "of", fs.height)
for g in ["01", "tail02_07", "tail12_25"]:
    s = fs.filter(pl.col("grp") == g)
    say(f"  {g}: {s.filter(pl.col('sum') == pl.col('round') - 1).height} of {s.height}")

# ---------------------------------------------------------------- (b) round-13 label test on the score ledger
say("\n== (b) score ledger delta at label k vs round_end at label k+o, o = -1, 0, +1 (non-zero deltas) ==")
REu = re_.sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True).select(
    ["match_key", "round", "winner_team_code", "grp"])
fes = fe.select(["match_key", "round", "t_score", "ct_score"]).sort(["match_key", "round"])
nx = fes.with_columns([pl.col("round").shift(-1).over("match_key").alias("r_next"),
                       pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
                       pl.col("ct_score").shift(-1).over("match_key").alias("ct_next")])
nx = nx.filter(pl.col("r_next") == pl.col("round") + 1).with_columns([
    (pl.col("t_next") - pl.col("t_score")).alias("dT"), (pl.col("ct_next") - pl.col("ct_score")).alias("dCT")])
nx = nx.with_columns(pl.when(pl.col("dT") == 1).then(2).when(pl.col("dCT") == 1).then(3).otherwise(0).alias("w_score"))
nx = nx.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day")).with_columns(
    pl.when(pl.col("day") == "01").then(pl.lit("01")).when(pl.col("day") <= "07").then(pl.lit("tail02_07"))
    .otherwise(pl.lit("tail12_25")).alias("grp"))
nz = nx.filter(pl.col("w_score") != 0)


def end_at(o):
    # winner of round_end labelled (k + o), keyed by k
    return REu.select(["match_key", (pl.col("round") - o).alias("round"), pl.col("winner_team_code").alias(f"w{o}")])


J2 = nz.join(end_at(-1), on=["match_key", "round"], how="left").join(end_at(0), on=["match_key", "round"], how="left") \
       .join(end_at(1), on=["match_key", "round"], how="left")
for g, ks in [("01", [9, 13]), ("tail02_07", [9, 13]), ("tail12_25", [9, 13, 25, 27, 28]), ("tail12_25", [5])]:
    for k in ks:
        s = J2.filter((pl.col("grp") == g) & (pl.col("round") == k))
        n = s.height
        if n == 0:
            continue
        say(f"  {g} k={k}: n={n} | agree w(k-1)={s.filter(pl.col('w_score') == pl.col('w-1')).height} "
            f"agree w(k)={s.filter(pl.col('w_score') == pl.col('w0')).height} "
            f"agree w(k+1)={s.filter(pl.col('w_score') == pl.col('w1')).height} "
            f"agree inverse of w(k)={s.filter(pl.col('w_score') == 5 - pl.col('w0')).height}")

# ---------------------------------------------------------------- (c) last rounds vs player_count (rst gate reach)
say("\n== (c) round_end rows: last round of match vs round_end.player_count (2024) ==")
lastR = re_.group_by("match_key").agg(pl.col("round").max().alias("last"))
R2 = re_.join(lastR, on="match_key").with_columns((pl.col("round") == pl.col("last")).alias("is_last"))
say("is_last x player_count (n):", R2.group_by(["is_last", "player_count"]).len().sort(["is_last", "player_count"]).rows()[:20])
say("last-round rows with player_count == 20 (full roster per e2):", R2.filter(pl.col("is_last") & (pl.col("player_count") == 20)).height,
    "of", R2.filter(pl.col("is_last")).height)

# ---------------------------------------------------------------- (d) start-family tick equality with round_start
say("\n== (d) start-family events: same (match, label) as round_start, same tick? ==")
rs_key = rs.select(["match_key", "round", pl.col("tick").alias("rs_tick")]).unique(["match_key", "round"])
for ev in ["round_prestart", "round_poststart", "round_officially_ended"]:
    e = st.filter(pl.col("event_type") == ev).select(["match_key", "round", "tick"]).unique(["match_key", "round"])
    j = e.join(rs_key, on=["match_key", "round"], how="inner")
    say(f"  {ev}: rows={e.height} with round_start at same (match,label)={j.height} same tick={j.filter(pl.col('tick') == pl.col('rs_tick')).height}")
mvp = load(T24, "round_mvp", ["match_key", "round", "tick"])
if mvp is not None:
    mv = mvp.select(["match_key", "round", "tick"]).unique(["match_key", "round"])
    say(f"  round_mvp channel: rows={mvp.height}; label-matched with round_start={mv.join(rs_key, on=['match_key','round'], how='inner').height}")
    # round_mvp vs round_end label (same label, same match)
    re_key = re_.select(["match_key", "round", pl.col("tick").alias("re_tick")]).unique(["match_key", "round"])
    say(f"  round_mvp label-matched with round_end: {mv.join(re_key, on=['match_key', 'round'], how='inner').height}")

# ---------------------------------------------------------------- (e) raw E: source spot check (tail match, day 12)
say("\n== (e) raw source spot check: E: folder for tail match 734e42e0 (day 12) ==")
RAW = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12\734e42e0-3ad6-4c1e-af4d-dc3385177620\csds"
say("raw match folder exists:", os.path.isdir(RAW))
if os.path.isdir(RAW):
    names = sorted(os.listdir(RAW))
    say("entries (first 60):", names[:60])
    cand = [n for n in names if "round_start" in n]
    say("round_start entries:", cand)
    for n in cand:
        p = os.path.join(RAW, n)
        try:
            if os.path.isdir(p):
                files = sorted(f for f in os.listdir(p) if f.endswith(".parquet") or f.startswith("dataframe"))
                frames = [pl.read_parquet(os.path.join(p, f)) for f in files]
                df = pl.concat(frames, how="vertical_relaxed") if frames else None
            else:
                df = pl.read_parquet(p)
            if df is not None:
                say("  raw", n, "columns:", df.columns, "rows:", df.height)
                say("  raw round_start (round, tick):", df.select([c for c in ["round", "tick"] if c in df.columns]).sort("tick").head(4).rows())
        except Exception as ex:  # noqa: BLE001
            say("  raw read failed:", n, str(ex)[:200])
say("tome (same match) round_start (round, tick):",
    rs.filter(pl.col("match_key") == "csds/2024/02/12/734e42e0-3ad6-4c1e-af4d-dc3385177620/csds").sort("tick").head(4).select(["round", "tick"]).rows())
say("\nDONE")
