"""E1 step 2: validation of keys, round timeline, side mapping, and item-event placement (from the cache).

Reads only logs\\swarm\\E1\\cache (built from G: tomes). Prints diagnostics; writes nothing else.
Run: cd G:\\pureskill\\csds-cs2 ; pixi run python -I -X utf8 scripts\\swarm\\E1_validate.py
"""

import os

import polars as pl

CACHE = r"G:\pureskill\csds-cs2\logs\swarm\E1\cache"


def c(name):
    return pl.read_parquet(os.path.join(CACHE, name + ".parquet"))


def main():
    hdr = c("header")
    print("header rows", hdr.height, "unique keys", hdr["key"].n_unique())
    print("is_wingman", hdr.group_by("is_wingman").len().to_dicts())
    print("max_rounds", hdr.group_by("max_rounds").len().to_dicts())
    print("match_type", hdr.group_by("match_type").len().to_dicts())
    days = hdr.with_columns(pl.col("key").str.extract(r"csds/2026/10/(\d{2})/", 1).alias("day"))
    print("day counts", days.group_by("day").len().sort("day").to_dicts())

    re_ = c("round_end")
    print("\nround_end rows", re_.height, "unique (match,round)", re_.select(["match_key", "round"]).unique().height)
    print("player_count", re_.group_by("player_count").len().to_dicts())
    print("winner", re_.group_by("winner_team_code").len().to_dicts())
    print("win_reason", re_.group_by("win_reason_code").len().sort("len", descending=True).head(12).to_dicts())
    dup_re = re_.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
    print("round_end duplicate (match,round) pairs:", dup_re.height)

    rs = c("round_state")
    fe = rs.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"])
    print("\nround_freeze_end rows", fe.height, "unique (match,round)", fe.select(["match_key", "round"]).unique().height)
    fe_dup = fe.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
    print("freeze_end duplicate pairs", fe_dup.height)
    bt = rs.filter(pl.col("event_type") == "buytime_ended").select(["match_key", "round", "tick"])
    print("buytime_ended rows", bt.height, "unique", bt.select(["match_key", "round"]).unique().height)
    bt_dup = bt.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
    print("buytime duplicate pairs", bt_dup.height)
    rnd = re_.select(["match_key", "round"]).unique()
    miss_fe = rnd.join(fe.select(["match_key", "round"]).unique(), on=["match_key", "round"], how="anti")
    print("rounds (round_end) without freeze_end:", miss_fe.height)
    fe_u = fe.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("fe_tick"))
    bt_u = bt.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("bt_tick"))
    gap = fe_u.join(bt_u, on=["match_key", "round"]).with_columns((pl.col("bt_tick") - pl.col("fe_tick")).alias("gap"))
    print("buytime - freeze_end gap quantiles:", gap["gap"].quantile(0.05), gap["gap"].quantile(0.5), gap["gap"].quantile(0.95))

    rstart = c("round_start").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("rs_tick"))
    tl = fe_u.join(rstart, on=["match_key", "round"], how="left")
    print("freeze_end - round_start quantiles:",
          (tl["fe_tick"] - tl["rs_tick"]).quantile(0.05), (tl["fe_tick"] - tl["rs_tick"]).quantile(0.5),
          (tl["fe_tick"] - tl["rs_tick"]).quantile(0.95), "null rs:", tl["rs_tick"].null_count())

    pi = c("player_info")
    print("\nplayer_info rows", pi.height)
    print("team_code", pi.group_by("team_code").len().to_dicts())
    pk = pi.group_by(["match_key", "round", "player_id"]).len().filter(pl.col("len") > 1)
    print("player_info duplicate (match,round,player) keys:", pk.height)
    pid_range = pi.select(pl.col("player_id").min().alias("min"), pl.col("player_id").max().alias("max"))
    print("player_id range", pid_range.to_dicts())
    print("player_id==0 rows in player_info", pi.filter(pl.col("player_id") == 0).height)
    per_round = pi.group_by(["match_key", "round"]).agg(
        pl.len().alias("n"), (pl.col("team_code") == 2).sum().alias("nt"), (pl.col("team_code") == 3).sum().alias("nct"))
    print("players per round:", per_round.group_by("n").len().sort("n").to_dicts())
    print("rounds with 5T/5CT:", per_round.filter((pl.col("nt") == 5) & (pl.col("nct") == 5)).height,
          "of", per_round.height)
    print("rounds with 10 players but not 5/5:", per_round.filter((pl.col("n") == 10) & ~((pl.col("nt") == 5) & (pl.col("nct") == 5))).height)

    for name in ["item_equip", "item_pickup", "item_dropped", "item_refund"]:
        it = c(name)
        print("\n==", name, "rows", it.height)
        print("player_id==0 rows", it.filter(pl.col("player_id") == 0).height)
        j = it.join(pi.select(["match_key", "round", "player_id", "team_code"]),
                    on=["match_key", "round", "player_id"], how="left")
        print("unmapped to side (no player_info row):", j.filter(pl.col("team_code").is_null()).height)
        mism = j.filter(pl.col("team_code").is_not_null() & pl.col("player_team_code").is_not_null()
                        & (pl.col("team_code") != pl.col("player_team_code")))
        both = j.filter(pl.col("team_code").is_not_null() & pl.col("player_team_code").is_not_null())
        print("player_team_code agrees with player_info team_code: %d of %d (mismatch %d)" % (
            both.height - mism.height, both.height, mism.height))
        print("player_team_code null rows:", it.filter(pl.col("player_team_code").is_null()).height)
        rnd_keys = rnd
        missing_round = it.select(["match_key", "round"]).unique().join(rnd_keys, on=["match_key", "round"], how="anti")
        print("item (match,round) pairs with no round_end:", missing_round.height)

    ie = c("item_equip")
    print("\nitem_equip item counts top 25:", ie.group_by("item").len().sort("len", descending=True).head(25).to_dicts())
    print("weapon_type_code", ie.group_by("weapon_type_code").len().sort("weapon_type_code").to_dicts())


if __name__ == "__main__":
    main()
