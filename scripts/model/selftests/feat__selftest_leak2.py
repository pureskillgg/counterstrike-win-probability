"""SELF-TEST ONLY: reads the round labels directly. Must be REJECTED by the static guard."""
import polars as pl


def compute(G, store):
    R = pl.read_parquet(r"G:\pureskill\csds-cs2\data\combined_test\rounds.parquet").select("match_key", "round", "final")
    return G.select("match_key", "round").join(R, on=["match_key", "round"], how="left").select(pl.col("final").cast(pl.Float32).alias("leak"))
