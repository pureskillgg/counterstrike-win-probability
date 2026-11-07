"""Check round_buy_types round classes to see how the report's rifle-side subset (n=266) is defined.
Also tests whether 'other heavy' in the report includes G3SG1 (11) and SCAR-20 (38)."""
import numpy as np
import polars as pl

HZ = r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"
RD = r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet"
BT = r"G:\pureskill\csds-cs2\data\state_sample\round_buy_types.parquet"

bt = pl.read_parquet(BT)
print("round_buy_types rows:", bt.height)
print("is_pistol counts:", bt["is_pistol"].value_counts().rows())
print("round_class counts (all rounds):")
print(bt["round_class"].value_counts(sort=True))
print("round_class counts (non-pistol):")
print(bt.filter(~pl.col("is_pistol"))["round_class"].value_counts(sort=True))
print("higher_side counts (non-pistol):")
print(bt.filter(~pl.col("is_pistol"))["higher_side"].value_counts(sort=True))
print("ct_buy values (top):")
print(bt["ct_buy"].value_counts(sort=True).head(12))
print("t_buy values (top):")
print(bt["t_buy"].value_counts(sort=True).head(12))

# check pistol flag vs round number
print("is_pistol by round:")
print(bt.group_by(["round", "is_pistol"]).agg(pl.len().alias("n")).sort(["round", "is_pistol"]).head(30))
