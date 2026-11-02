import polars as pl

SCR = "C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
p = pl.read_parquet(SCR + "/v4_test_preds_with_flags.parquet")
total_ll = p["ll"].mean()
total_loss = p["ll"].sum()
print("rows", p.height, "matches", p["match_key"].n_unique(), "v4 LL", round(total_ll, 5), "AUC-free")
print("t min/max", p["t"].min(), p["t"].max())

p = p.with_columns(
    (pl.col("t") // 10 * 10).cast(pl.Int64).alias("tb"),
    pl.when(pl.col("planted") == 1).then(pl.lit("post-plant")).otherwise(pl.lit("pre-plant")).alias("phase"),
)
print("\nby seconds into round (10 s buckets): rows share, mean LL, share of loss, mean |p-0.5|")
g = p.group_by("tb").agg(
    pl.len().alias("rows"),
    pl.col("ll").mean().alias("mean_ll"),
    pl.col("ll").sum().alias("loss_sum"),
    (pl.col("p") - 0.5).abs().mean().alias("mean_conf"),
).sort("tb")
for r in g.iter_rows(named=True):
    print("  t %4d-%4d  rows %5.1f%%  LL %.4f  loss share %5.1f%%  mean|p-0.5| %.3f" % (
        r["tb"], r["tb"] + 9, 100 * r["rows"] / p.height, r["mean_ll"], 100 * r["loss_sum"] / p["ll"].sum(), r["mean_conf"]))

print("\nby phase and 5v5 status")
g2 = p.with_columns(
    pl.when((pl.col("ac") == 5) & (pl.col("at") == 5)).then(pl.lit("5v5")).otherwise(pl.lit("not 5v5")).alias("strength")
).group_by(["phase", "strength"]).agg(
    pl.len().alias("rows"), pl.col("ll").mean().alias("mean_ll"), pl.col("ll").sum().alias("loss_sum")).sort(["phase", "strength"])
for r in g2.iter_rows(named=True):
    print("  %-10s %-8s rows %5.1f%%  LL %.4f  loss share %5.1f%%" % (
        r["phase"], r["strength"], 100 * r["rows"] / p.height, r["mean_ll"], 100 * r["loss_sum"] / p["ll"].sum()))

print("\nreference: flat 0.5 guess LL = 0.6931; v4 pooled LL = %.4f" % total_ll)
print("share of rows with v4 p in 0.4-0.6 (near coin flip):", round(float(((p["p"] > 0.4) & (p["p"] < 0.6)).mean()), 4))
print("share of rows with v4 p above 0.9 or below 0.1:", round(float(((p["p"] > 0.9) | (p["p"] < 0.1)).mean()), 4))
