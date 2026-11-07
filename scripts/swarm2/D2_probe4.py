"""D2 probe 4: round-by-round trace for example matches (2024 affected and unaffected), plus
validation of the death-based elimination side against the reason-based side in all rounds.

Trace columns per round: valid sides (sx), recorded winner, reason side, bomb outcome event and tick,
round_start tick, round_end tick, planter side, death-based elimination side, score-based side.

Also: agreement of the death-based elimination side with the reason-based side, by year and round bin.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\D2_probe4.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(30)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def read(label, ch, cols):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    return pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*"))).select(cols).collect()


def elim_table(label, base):
    pi = (read(label, "player_info", ["match_key", "round", "player_id", "team_code"])
          .unique(["match_key", "round", "player_id"], keep="first"))
    first_valid = base.filter(pl.col("sx").is_not_null()).group_by("match_key").agg(pl.col("round").min().alias("F"))
    pim = pi.join(first_valid, on="match_key", how="inner").filter(pl.col("round") == pl.col("F"))
    mem = pim.select(["match_key", "player_id", pl.col("team_code").alias("tc_F")]).unique(["match_key", "player_id"])
    mem = mem.with_columns((pl.col("tc_F") == 2).alias("isX"))
    dth = read(label, "player_death", ["match_key", "round", "tick", "player_id"])
    d = dth.join(mem.select(["match_key", "player_id", "isX"]), on=["match_key", "player_id"], how="left")
    vf = base.select(["match_key", "round", "sx"]).filter(pl.col("sx").is_not_null())
    d = d.join(vf, on=["match_key", "round"], how="left")
    d = d.with_columns(pl.when(pl.col("isX")).then(pl.col("sx")).otherwise(5 - pl.col("sx")).alias("victim_side"))
    elim = d.group_by(["match_key", "round"]).agg(
        (pl.col("victim_side") == 2).sum().alias("t_deaths"),
        (pl.col("victim_side") == 3).sum().alias("ct_deaths"))
    elim = elim.with_columns(
        pl.when((pl.col("ct_deaths") == 5) & (pl.col("t_deaths") < 5)).then(pl.lit(2))
        .when((pl.col("t_deaths") == 5) & (pl.col("ct_deaths") < 5)).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_elim"))
    return elim


def main():
    for label in ["2024", "2026"]:
        base = pl.read_parquet(ROOT / "logs" / "swarm2" / f"D2_rounds_{label}.parquet")
        elim = elim_table(label, base)
        b = base.join(elim.select(["match_key", "round", "s_elim", "t_deaths", "ct_deaths"]), on=["match_key", "round"], how="left")
        b = b.filter(pl.col("s_reason").is_not_null() & pl.col("s_elim").is_not_null())
        print(f"\n== {label}: elimination-side vs reason-side agreement (rounds with a 5-death wipe on one side)")
        print(b.group_by("bin").agg(
            pl.len().alias("n"),
            (pl.col("s_elim") == pl.col("s_reason")).sum().alias("elim==reason"),
            (pl.col("s_elim") != pl.col("s_reason")).sum().alias("elim!=reason")).sort("bin"))
        print(b.group_by("day").agg(
            pl.len().alias("n"),
            (pl.col("s_elim") != pl.col("s_reason")).sum().alias("elim!=reason")).sort("day").head(30))

    label = "2024"
    base = pl.read_parquet(ROOT / "logs" / "swarm2" / f"D2_rounds_{label}.parquet")
    elim = elim_table(label, base)
    rs_start = read(label, "round_start", ["match_key", "round", "tick"]).sort("tick").unique(["match_key", "round"], keep="first").rename({"tick": "tick_start"})
    re_ = read(label, "round_end", ["match_key", "round", "tick"]).sort("tick").unique(["match_key", "round"], keep="first").rename({"tick": "tick_end"})
    bs = read(label, "bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
    bo = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
          .select(["match_key", "round", "tick", "event_type"]).sort("tick"))
    bo_all = bo.group_by(["match_key", "round"]).agg(
        pl.col("event_type").first().alias("bomb_first"), pl.col("tick").first().alias("bomb_tick"), pl.len().alias("n_bomb"))
    pl_ = bs.filter(pl.col("event_type") == "bomb_planted").sort("tick").unique(["match_key", "round"], keep="first").select(
        ["match_key", "round", pl.col("player_team_code").alias("planter"), pl.col("tick").alias("plant_tick")])
    j = base.join(rs_start, on=["match_key", "round"], how="left").join(re_, on=["match_key", "round"], how="left")
    j = j.join(bo_all, on=["match_key", "round"], how="left").join(pl_, on=["match_key", "round"], how="left")
    j = j.join(elim.select(["match_key", "round", "s_elim"]), on=["match_key", "round"], how="left")

    # pick examples: 2024 matches (days 12-25) with a round-13 bomb mismatch, and days 01-07 with a bomb outcome in round 13
    aff = (j.filter((pl.col("round") == 13) & pl.col("bomb_first").is_not_null())
           .with_columns(pl.when(pl.col("bomb_first") == "bomb_exploded").then(2).otherwise(3).alias("s_bomb_r13")))
    aff = aff.filter(pl.col("s_bomb_r13") != pl.col("winner"))
    ok = j.filter((pl.col("round") == 13) & pl.col("bomb_first").is_not_null())
    ok = ok.with_columns(pl.when(pl.col("bomb_first") == "bomb_exploded").then(2).otherwise(3).alias("s_bomb_r13")).filter(pl.col("s_bomb_r13") == pl.col("winner"))
    ex_aff = aff.select("match_key").head(2).to_series().to_list()
    ex_ok = ok.select("match_key").head(1).to_series().to_list()
    cols = ["round", "sx", "sx_prev", "winner", "s_reason", "s_bomb", "s_score", "s_elim", "win_reason_code",
            "bomb_first", "n_bomb", "tick_start", "bomb_tick", "tick_end", "plant_tick", "planter", "t0", "ct0", "t1", "ct1"]
    for mk in ex_aff + ex_ok:
        tag = "AFFECTED" if mk in ex_aff else "CLEAN"
        print(f"\n==== {tag} match {mk}")
        t = j.filter((pl.col("match_key") == mk) & (pl.col("round") >= 10) & (pl.col("round") <= 16)).sort("round")
        print(t.select(cols))

    # all 2024 round-13 bomb-mismatch rows: where is the bomb tick relative to round 12 window?
    aff_all = j.filter((pl.col("round") == 13) & pl.col("bomb_first").is_not_null()).with_columns(
        pl.when(pl.col("bomb_first") == "bomb_exploded").then(2).otherwise(3).alias("s_bomb_r13")).filter(pl.col("s_bomb_r13") != pl.col("winner"))
    r12 = j.filter(pl.col("round") == 12).select(["match_key", pl.col("tick_start").alias("t12s"), pl.col("tick_end").alias("t12e"),
                                                  pl.col("bomb_first").alias("b12"), pl.col("bomb_tick").alias("b12tick"), pl.col("n_bomb").alias("n12")])
    x = aff_all.join(r12, on="match_key", how="left")
    print("\n== round-13 bomb-mismatch rows: bomb tick vs round-12 window")
    print(x.with_columns(
        ((pl.col("bomb_tick") >= pl.col("t12s")) & (pl.col("bomb_tick") <= pl.col("t12e"))).alias("in_r12_window"),
        ((pl.col("bomb_tick") >= pl.col("tick_start")) & (pl.col("bomb_tick") <= pl.col("tick_end"))).alias("in_r13_window"),
    ).group_by(["in_r12_window", "in_r13_window"]).len())
    print("\n== round-13 bomb-mismatch rows: bomb tick minus round-13 start tick (describe)")
    print(x.select((pl.col("bomb_tick") - pl.col("tick_start")).alias("d")).describe())
    print("\n== round-13 bomb-mismatch rows: bomb tick minus round-12 end tick (describe)")
    print(x.select((pl.col("bomb_tick") - pl.col("t12e")).alias("d")).describe())
    print("\n== round-13 bomb-mismatch rows: bomb_first and n_bomb (round 13)")
    print(x.group_by(["bomb_first", "n_bomb"]).len())
    print("\n== round-12 bomb outcomes for the same matches (b12, n12)")
    print(x.group_by(["b12", "n12"]).len())


if __name__ == "__main__":
    main()
