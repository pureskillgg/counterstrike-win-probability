# verify_A2_2_r13.py
# Score-vs-round_end agreement by individual round number (k), 2024 and 2026, to see whether the
# 2024 disagreement (outside the A2-2 label shift) starts at a specific round.
import os, json
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
TOMES = {"2024": os.path.join(ROOT, "tomes", "2024-02-g", "tome", "csds"),
         "2026": os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")}


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
    RE = read(tome, "basic_round_end", ["match_key", "round", "winner_team_code"])
    RE = RE.with_columns(pl.col("round").cast(pl.Int64), pl.col("winner_team_code").cast(pl.Int64, strict=False))
    sc = read(tome, "basic_round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"],
              filt=pl.col("event_type") == "round_freeze_end")
    sc = sc.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
                         pl.col("t_score").cast(pl.Int64, strict=False), pl.col("ct_score").cast(pl.Int64, strict=False))
    sc = sc.sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True).sort(["match_key", "round"])
    nxt = sc.with_columns([pl.col("round").shift(-1).over("match_key").alias("r2"),
                           pl.col("t_score").shift(-1).over("match_key").alias("t2"),
                           pl.col("ct_score").shift(-1).over("match_key").alias("c2")])
    nxt = nxt.filter(pl.col("r2") == pl.col("round") + 1).with_columns(
        [(pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")])
    nxt = nxt.with_columns(pl.when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(2)
                           .when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(3).otherwise(0).alias("ws"))
    rk = RE.select(["match_key", pl.col("round").alias("k"), pl.col("winner_team_code").alias("wk")]).unique(
        ["match_key", "k"], keep="first", maintain_order=True)
    nxt = nxt.join(rk, left_on=["match_key", "round"], right_on=["match_key", "k"], how="left")
    ok = nxt.filter(pl.col("ws") != 0)
    tab = ok.group_by("round").agg(pl.len().alias("n"), (pl.col("ws") == pl.col("wk")).sum().alias("eq_k")).sort("round")
    rows = [(r["round"], r["n"], r["n"] - r["eq_k"]) for r in tab.iter_rows(named=True)]
    print(f"\n== [{tag}] score-vs-round_end disagreements by round k: (k, pairs, disagree) ==")
    print(rows)
    dis = ok.filter(pl.col("ws") != pl.col("wk")).height
    print(f"  total disagreements: {dis} of {ok.height}")
    print("  disagreement by side of score sign (ws=2 T, 3 CT), (ws, wk):",
          ok.filter(pl.col("ws") != pl.col("wk")).group_by(["ws", "wk"]).len().sort(["ws", "wk"]).rows())


if __name__ == "__main__":
    run("2024")
    run("2026")
    print("\nDONE")
