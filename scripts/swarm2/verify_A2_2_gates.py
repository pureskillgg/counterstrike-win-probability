# verify_A2_2_gates.py - third-pass checks for finding A2-2.
# (f) raw E: source for the day-12 tail match (734e42e0): is the round_start offset already in the source?
# (g) per-round gate simulation for 2024 (bake-off order, simplified): current rst_keys vs the proposed +1 shift.
#     ASSUMPTIONS: full roster = round_end.player_count == 20 (as e2_side_check_r13.py uses); team codes T=2, CT=3;
#     5 T and 5 CT among the 10 distinct player_info rows at the round label; disconnect rows keyed by their label.
#     Not reproduced here: channel-kept match filter and duplicate-copy removal (so counts are per round, not per kept set).
import os
import json
import glob

import polars as pl

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)

ROOT = r"G:\pureskill\csds-cs2"
T24 = os.path.join(ROOT, "tomes", "2024-02-g", "tome", "csds")
RAWM = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12\734e42e0-3ad6-4c1e-af4d-dc3385177620"
MK = "csds/2024/02/12/734e42e0-3ad6-4c1e-af4d-dc3385177620/csds"


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


# ---------------------------------------------------------------- (f) raw source for one tail match
say("== (f) raw E: source vs tome, match 734e42e0 (day 12) ==")
say("raw match folder exists:", os.path.isdir(RAWM))


def raw_channel(ch):
    # raw channels in the match folder are single files named by channel (no extension)
    f = os.path.join(RAWM, ch)
    if os.path.isfile(f):
        try:
            return pl.read_parquet(f), "single file read as parquet"
        except Exception as ex:  # noqa: BLE001
            return None, f"read failed: {str(ex)[:160]}"
    d = os.path.join(RAWM, ch)
    if os.path.isdir(d):
        frames = [pl.read_parquet(os.path.join(d, e)) for e in sorted(os.listdir(d))]
        return (pl.concat(frames, how="vertical_relaxed") if frames else None), f"folder with {len(frames)} files"
    return None, "missing"


raw_rs, info = raw_channel("round_start")
say("raw round_start:", info, "| columns:", None if raw_rs is None else raw_rs.columns)
if raw_rs is not None:
    say("raw round_start (round, tick) first 6 by tick:", raw_rs.sort("tick").select(["round", "tick"]).head(6).rows())
raw_st, info2 = raw_channel("round_state")
say("raw round_state:", info2)
if raw_st is not None and "event_type" in raw_st.columns:
    fz = raw_st.filter(pl.col("event_type") == "round_freeze_end").sort("tick").select(["round", "tick"]).head(4).rows()
    say("raw round_freeze_end (round, tick) first 4:", fz)
tome_rs = load(T24, "round_start", ["match_key", "round", "tick"]).filter(pl.col("match_key") == MK).sort("tick")
say("tome round_start (round, tick) first 6 for same match:", tome_rs.select(["round", "tick"]).head(6).rows())
if raw_rs is not None:
    same = raw_rs.sort("tick").select(["round", "tick"]).rows() == tome_rs.sort("tick").select(["round", "tick"]).rows()
    say("raw round_start rows identical to tome rows for this match:", same, "| raw rows", raw_rs.height, "tome rows", tome_rs.height)

# ---------------------------------------------------------------- (g) gate simulation, current vs proposed rst_keys
say("\n== (g) per-round gate simulation, 2024 (ASSUMPTIONS in header) ==")
re_ = load(T24, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
rs = load(T24, "round_start", ["match_key", "round", "tick"])
pinf = load(T24, "player_info", ["match_key", "round", "player_id", "team_code"])
dc = load(T24, "player_disconnect", ["match_key", "round", "tick"])
st = load(T24, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = st.filter(pl.col("event_type") == "round_freeze_end").sort("tick").unique(["match_key", "round"], keep="first",
                                                                                maintain_order=True)
cand = re_.sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True).select(
    ["match_key", "round", "player_count", "grp"])
say("candidate rounds (unique round_end):", cand.height)

pi_sum = (pinf.unique(["match_key", "round", "player_id"], keep="first")
          .group_by(["match_key", "round"])
          .agg(pl.len().alias("n_players"),
               (pl.col("team_code") == 2).sum().alias("n_t"),
               (pl.col("team_code") == 3).sum().alias("n_ct")))
roster_ok = pi_sum.filter((pl.col("n_players") == 10) & (pl.col("n_t") == 5) & (pl.col("n_ct") == 5)).select(
    ["match_key", "round"])
disc = dc.select(["match_key", "round"]).unique()
rst_now = rs.select(["match_key", "round"]).unique()
rst_shift = rs.select(["match_key", (pl.col("round") + 1).alias("round")]).unique()
live = fe.filter(pl.col("t_score").is_not_null() & pl.col("ct_score").is_not_null()).select(["match_key", "round"])


def run_gates(rst, label):
    c = cand
    drops = {}
    a = c.filter(pl.col("player_count") != 20)
    drops["A player_count != 20"] = a.height
    c = c.join(a.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
    b = c.join(roster_ok, on=["match_key", "round"], how="anti")
    drops["B not 5T/5CT (player_info)"] = b.height
    c = c.join(b.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
    d_ = c.join(disc, on=["match_key", "round"], how="semi")
    drops["C disconnect in round"] = d_.height
    c = c.join(d_.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
    e = c.join(rst, on=["match_key", "round"], how="anti")
    drops["D no round_start row (rst)"] = e.height
    c = c.join(e.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
    f = c.join(live, on=["match_key", "round"], how="anti")
    drops["E no live freeze_end"] = f.height
    c = c.join(f.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
    say(f"{label}: survivors={c.height} drops={drops}")
    say(f"   survivors by group: {c.group_by('grp').len().sort('grp').rows()}")
    say(f"   survivors with round label 1: {c.filter(pl.col('round') == 1).height}")
    return c


cur = run_gates(rst_now, "CURRENT rst_keys (labels as recorded)")
sh = run_gates(rst_shift, "PROPOSED rst_keys shifted +1")
lost = cur.join(sh.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
gained = sh.join(cur.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
LAST = cand.group_by("match_key").agg(pl.col("round").max().alias("last"))
lastset = cand.join(LAST, on="match_key").filter(pl.col("round") == pl.col("last")).select(["match_key", "round"])
say("candidate last-of-match rounds:", lastset.height, "| candidate round-1 rounds:", cand.filter(pl.col("round") == 1).height)
say("rounds kept now but dropped by the shift:", lost.height, "| by label:", lost.group_by("round").len().sort("round").rows(),
    "| groups:", lost.group_by("grp").len().rows())
say("  of those, last-of-match:", lost.join(lastset, on=["match_key", "round"], how="semi").height)
say("rounds dropped now but kept by the shift:", gained.height, "| by label:", gained.group_by("round").len().sort("round").rows(),
    "| groups:", gained.group_by("grp").len().rows())
say("  of those, last-of-match:", gained.join(lastset, on=["match_key", "round"], how="semi").height)
say("net change in survivors (shift minus current):", sh.height - cur.height)
say("\nDONE")
