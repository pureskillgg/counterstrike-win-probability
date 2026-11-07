import glob, polars as pl
BASE = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
def files(ch):
    return sorted(glob.glob(BASE + "/basic_" + ch + ".COMBINED/dataframe_*"))
def lf26(ch, cols=None):
    lf = pl.scan_parquet(files(ch)).filter(pl.col("match_key").str.starts_with("csds/2026/"))
    if cols: lf = lf.select(cols)
    return lf
rs = lf26("round_state").collect()
print("round_state cols:", rs.columns)
print("round_state rows:", rs.height)
print(rs.group_by("event_type").len().sort("len", descending=True).head(30).rows() if "event_type" in rs.columns else "no event_type")
