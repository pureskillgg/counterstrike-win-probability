import glob
import polars as pl
base = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
def pages(ch):
    return sorted(glob.glob(base + "/basic_" + ch + ".COMBINED/dataframe_*"))
for ch in ["round_end", "player_death"]:
    ps = pages(ch)
    print("===", ch, "pages", len(ps), ps[:1])
    lf = pl.concat([pl.scan_parquet(p) for p in ps], how="diagonal_relaxed")
    print(lf.collect_schema())
    print(lf.head(4).collect())
    print("rows", lf.select(pl.len()).collect().item())
    print("distinct match_key prefixes", lf.select(pl.col("match_key").str.slice(0, 9).n_unique()).collect().item())
