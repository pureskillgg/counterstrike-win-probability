"""Single rule set for the per-second loadout (serving rule from scripts/serving/win_probability.py, lines 40-57 and 249-291).

Grid: tick = freeze + 64*k for k >= 0 (freeze = export round_freeze), k < steps of the round in the export (serving loop bound).
Per (step, side): the 20 fields below, computed exactly as serving does (null -> 0, inv_primary null -> 0, false for null flags).
Side: team_code 3 = CT, 2 = T (serving: any code other than 3 goes to T; here any other code is dropped and counted).
"""
import polars as pl

TICK = 64
RIFLE = (7, 8, 10, 13, 16, 39, 60)
M4 = (16, 60)
AK = (7,)
AWP = (9,)
SNIPER_OTHER = (11, 38, 40)
SMG = (17, 19, 23, 24, 26, 33, 34)
HEAVY = (14, 25, 27, 28, 29, 35)
FIELDS = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle",
          "sniper_other", "smg", "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy", "equip"]
INPUT_COLS = ["health", "armor", "has_helmet", "has_defuser", "is_defusing", "inv_primary", "inv_flashbang",
              "inv_smokegrenade", "inv_hegrenade", "inv_molotov", "inv_incgrenade", "inv_decoy", "equipment_value_calc"]


def aggregate(df):
    """df: one row per (step s, side_idx, player) after grid filter and dedupe. Returns one row per (s, side_idx)."""
    inv = pl.col("inv_primary").fill_null(0).cast(pl.Int64)
    arm = pl.col("armor").fill_null(0).cast(pl.Int64)
    return (df.group_by(["s", "side_idx"]).agg([
        pl.col("health").fill_null(0).cast(pl.Int64).sum().alias("hp"),
        (arm > 0).cast(pl.Int64).sum().alias("armor_n"),
        arm.sum().alias("armor_sum"),
        pl.col("has_helmet").fill_null(False).cast(pl.Int64).sum().alias("helmet_n"),
        pl.col("has_defuser").fill_null(False).cast(pl.Int64).sum().alias("kit_n"),
        pl.col("is_defusing").fill_null(False).cast(pl.Int64).sum().alias("defusing"),
        (inv == AK[0]).cast(pl.Int64).sum().alias("ak"),
        inv.is_in(list(M4)).cast(pl.Int64).sum().alias("m4"),
        (inv == AWP[0]).cast(pl.Int64).sum().alias("awp"),
        inv.is_in(list(RIFLE)).cast(pl.Int64).sum().alias("rifle"),
        inv.is_in(list(SNIPER_OTHER)).cast(pl.Int64).sum().alias("sniper_other"),
        inv.is_in(list(SMG)).cast(pl.Int64).sum().alias("smg"),
        inv.is_in(list(HEAVY)).cast(pl.Int64).sum().alias("heavy"),
        (inv == 0).cast(pl.Int64).sum().alias("no_primary"),
        pl.col("inv_flashbang").fill_null(0).cast(pl.Int64).sum().alias("flash"),
        pl.col("inv_smokegrenade").fill_null(0).cast(pl.Int64).sum().alias("smoke"),
        pl.col("inv_hegrenade").fill_null(0).cast(pl.Int64).sum().alias("he"),
        (pl.col("inv_molotov").fill_null(0).cast(pl.Int64) + pl.col("inv_incgrenade").fill_null(0).cast(pl.Int64)).sum().alias("fire"),
        pl.col("inv_decoy").fill_null(0).cast(pl.Int64).sum().alias("decoy"),
        pl.col("equipment_value_calc").fill_null(0).cast(pl.Int64).sum().alias("equip"),
    ]))


def side_index(side_col):
    """CT (3) -> 0, T (2) -> 1, anything else -> -1 (dropped and counted by the caller)."""
    return (pl.when(side_col == 3).then(0).when(side_col == 2).then(1).otherwise(-1)).cast(pl.Int64)
