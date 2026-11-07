# A1: breakdown of the 90 header-vs-round disagreements from A1_header_vs_rounds.csv
import polars as pl
d = pl.read_csv(r"G:\pureskill\csds-cs2\logs\swarm2\A1_header_vs_rounds.csv", infer_schema_length=0).with_columns(
    [pl.col(c).cast(pl.Int64) for c in ["max_rounds", "n", "S", "rnd_t", "rnd_ct", "hdr_t", "hdr_ct", "max_team_round", "threshold"]])
bad = d.filter(pl.col("round_match") == "false")
print("disagree total:", bad.height)
a = bad.filter(pl.col("n") < pl.col("S"))
b = bad.filter(pl.col("n") == pl.col("S"))
print("A (n<S):", a.height, "| reached threshold on last rnd:", a.filter(pl.col("max_team_round") >= pl.col("threshold")).height,
      "| header-team-swap in A (rnd_t==hdr_ct & rnd_ct==hdr_t):", a.filter((pl.col("rnd_t") == pl.col("hdr_ct")) & (pl.col("rnd_ct") == pl.col("hdr_t"))).height)
print("A gap distribution S-n:", a.group_by((pl.col("S") - pl.col("n")).alias("g")).len().sort("g").rows())
print("A: header other-team minus recorded other-team (hdr_t - rnd_t if winner is ct...):")
a2 = a.with_columns(
    pl.when(pl.col("hdr_t") >= pl.col("hdr_ct")).then(pl.col("hdr_t") - pl.col("rnd_t")).otherwise(pl.col("hdr_ct") - pl.col("rnd_ct")).alias("excess_on_loser"),
    pl.when(pl.col("hdr_t") >= pl.col("hdr_ct")).then(pl.col("hdr_ct") - pl.col("rnd_ct")).otherwise(pl.col("hdr_t") - pl.col("rnd_t")).alias("winner_side_diff"))
print("   winner_side_diff (should be 0 if only loser total inflated):", a2.group_by("winner_side_diff").len().sort("winner_side_diff").rows())
print("   excess on loser distribution:", a2.group_by("excess_on_loser").len().sort("excess_on_loser").rows())
print("B (n==S, team split disagrees):", b.height, "| max_rounds:", b.group_by("max_rounds").len().rows())
print(b.select(["key","max_rounds","n","S","rnd_t","rnd_ct","hdr_t","hdr_ct","max_team_round","threshold"]).rows())
print("A not reaching threshold:", a.filter(pl.col("max_team_round") < pl.col("threshold")).select(["key","max_rounds","n","S","rnd_t","rnd_ct","hdr_t","hdr_ct"]).rows())
