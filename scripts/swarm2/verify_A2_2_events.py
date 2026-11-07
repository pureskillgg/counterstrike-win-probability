# verify_A2_2_events.py
# Per-event-type label offset against the latest round_end before the event, both tomes.
# Also: freeze_end min label per match by day group, and score test agreement by round bucket (label k and k+1).
import os, json
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
TOMES = {"2024": os.path.join(ROOT, "tomes", "2024-02-g", "tome", "csds"),
         "2026": os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")}
EVENTS = ["round_start", "round_poststart", "round_prestart", "round_officially_ended"]


def read(tome, base, cols, filt=None):
    names = [x for x in os.listdir(tome) if x.split(".", 1)[0] == base]
    assert len(names) == 1, (base, names)
    dp = os.path.join(tome, names[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = []
    for p in meta["pages"]:
        lf = pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols)
        if filt is not None:
            lf = lf.filter(filt)
        parts.append(lf.collect())
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]


def run(tag):
    tome = TOMES[tag]
    RE = read(tome, "basic_round_end", ["match_key", "round", "tick", "winner_team_code"])
    RE = RE.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
                         pl.col("winner_team_code").cast(pl.Int64, strict=False))
    re_s = RE.select(["match_key", pl.col("tick").alias("te"), pl.col("round").alias("j")]).sort("te")
    RS = read(tome, "basic_round_start", ["match_key", "round", "tick"])
    ST = read(tome, "basic_round_state", ["match_key", "round", "tick", "event_type"],
              filt=pl.col("event_type").is_in(EVENTS[1:] + ["round_freeze_end"]))
    ST = ST.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    RS = RS.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).with_columns(
        pl.lit("round_start").alias("event_type"))
    allev = pl.concat([RS.select(["match_key", "round", "tick", "event_type"]),
                       ST.select(["match_key", "round", "tick", "event_type"])], how="vertical_relaxed")
    print(f"\n== [{tag}] label offset of each boundary event vs latest round_end before it (d = event label - round_end label) ==")
    for ev in EVENTS:
        E = allev.filter(pl.col("event_type") == ev).sort("tick")
        if E.height == 0:
            print(f"  {ev}: no rows")
            continue
        J = E.join_asof(re_s, left_on="tick", right_on="te", by="match_key", strategy="backward")
        J = J.with_columns((pl.col("round") - pl.col("j")).alias("d"))
        print(f"  {ev}: rows={E.height} d-distribution={J.group_by('d').len().sort('d').rows()}")
    # freeze_end min label per match, by day group (2024 only: day groups; 2026: per day)
    FE = ST.filter(pl.col("event_type") == "round_freeze_end")
    mins = FE.group_by("match_key").agg(pl.col("round").min().alias("fe_min"))
    mins = mins.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
    if tag == "2024":
        mins = mins.with_columns(pl.when(pl.col("day") == "01").then(pl.lit("d01")).when(pl.col("day") <= "07")
                                 .then(pl.lit("d02-07")).otherwise(pl.lit("d12-25")).alias("grp"))
        print("  freeze_end min label per match by group:", mins.group_by(["grp", "fe_min"]).len().sort(["grp", "fe_min"]).rows()[:10], "...")
    else:
        print("  freeze_end min label per match (all):", mins.group_by("fe_min").len().sort("fe_min").rows())
    # score test by round bucket, label k and k+1
    sc = read(tome, "basic_round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"],
              filt=pl.col("event_type") == "round_freeze_end")
    sc = sc.with_columns(pl.col("round").cast(pl.Int64), pl.col("t_score").cast(pl.Int64, strict=False),
                         pl.col("ct_score").cast(pl.Int64, strict=False))
    sc = sc.sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True).sort(["match_key", "round"])
    nxt = sc.with_columns([pl.col("round").shift(-1).over("match_key").alias("r2"),
                           pl.col("t_score").shift(-1).over("match_key").alias("t2"),
                           pl.col("ct_score").shift(-1).over("match_key").alias("c2")])
    nxt = nxt.filter(pl.col("r2") == pl.col("round") + 1).with_columns(
        [(pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")])
    nxt = nxt.with_columns(pl.when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(2)
                           .when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(3).otherwise(0).alias("ws"))
    rk = RE.select(["match_key", pl.col("round").alias("k"), pl.col("winner_team_code").alias("wk")])
    rk1 = RE.select(["match_key", pl.col("round").alias("k1"), pl.col("winner_team_code").alias("wk1")])
    nxt = nxt.join(rk, left_on=["match_key", "round"], right_on=["match_key", "k"], how="left")
    nxt = nxt.join(rk1, left_on=["match_key", "r2"], right_on=["match_key", "k1"], how="left")
    ok = nxt.filter(pl.col("ws") != 0).with_columns(
        pl.when(pl.col("round") <= 12).then(pl.lit("r01-12")).when(pl.col("round") <= 24).then(pl.lit("r13-24"))
        .otherwise(pl.lit("r25+")).alias("bk"))
    tab = ok.group_by("bk").agg(pl.len().alias("n"), (pl.col("ws") == pl.col("wk")).sum().alias("eq_k"),
                                (pl.col("ws") == pl.col("wk1")).sum().alias("eq_k1")).sort("bk")
    print("  score test by round bucket (n, eq_k, eq_k1):", tab.rows())
    print("  pairs with ws=0 (no single-side +1 change):", nxt.filter(pl.col("ws") == 0).height, "of", nxt.height)


if __name__ == "__main__":
    run("2024")
    run("2026")
    print("\nDONE")
