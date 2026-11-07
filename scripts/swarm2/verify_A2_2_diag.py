# verify_A2_2_diag.py
# Diagnostic for verify_A2_2: event order for early rounds in 2024 and 2026 matches, grouped by where the
# first round_start row sits relative to round_end 1 and round_freeze_end 1. Reads only labels <= 3 and ticks.
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
    RE = read(tome, "basic_round_end", ["match_key", "round", "tick", "winner_team_code"])
    RS = read(tome, "basic_round_start", ["match_key", "round", "tick"])
    ST = read(tome, "basic_round_state", ["match_key", "round", "tick", "event_type"],
              filt=pl.col("round") <= 3)
    RE = RE.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    RS = RS.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    ST = ST.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    RE = RE.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
    rs1 = RS.group_by("match_key").agg(pl.col("tick").min().alias("rs_t1"), pl.col("round").min().alias("rs_l1"))
    re1 = RE.filter(pl.col("round") == 1).group_by("match_key").agg(pl.col("tick").min().alias("re1_t"),
                                                                   pl.col("tick").max().alias("re1_tmax"), pl.len().alias("re1_n"))
    fe1 = ST.filter((pl.col("event_type") == "round_freeze_end") & (pl.col("round") == 1)).group_by(
        "match_key").agg(pl.col("tick").min().alias("fe1_t"))
    fe2 = ST.filter((pl.col("event_type") == "round_freeze_end") & (pl.col("round") == 2)).group_by(
        "match_key").agg(pl.col("tick").min().alias("fe2_t"))
    days = RE.select(["match_key", "day"]).unique("match_key")
    M = days.join(rs1, on="match_key", how="left").join(re1, on="match_key", how="left") \
            .join(fe1, on="match_key", how="left").join(fe2, on="match_key", how="left")
    M = M.with_columns(
        pl.when(pl.col("rs_t1").is_null()).then(pl.lit("no_rs"))
        .when(pl.col("re1_t").is_null()).then(pl.lit("no_re1"))
        .when(pl.col("rs_t1") < pl.col("re1_t")).then(pl.lit("firstRS_before_RE1"))
        .otherwise(pl.lit("RE1_before_firstRS")).alias("type"))
    M = M.with_columns(pl.when(pl.col("day") == "01").then(pl.lit("d01")).when(pl.col("day") <= "07").then(pl.lit("d02-07")).otherwise(pl.lit("d12-25")).alias("grp")) if tag == "2024" else M.with_columns(pl.lit("all").alias("grp"))
    print(f"\n== [{tag}] first round_start vs round_end 1 and round_freeze_end 1/2 ==")
    print("matches:", M.height)
    print("type counts by group:", M.group_by(["grp", "type"]).len().sort(["grp", "type"]).rows())
    print("round_freeze_end 1 present (by type):", M.group_by(["type"]).agg(pl.col("fe1_t").is_not_null().sum().alias("fe1_present"), pl.len().alias("n")).rows())
    print("round_end 1 row count per match (value_counts):", M.group_by(["re1_n"]).len().sort("re1_n").rows())
    tyX = M.filter(pl.col("type") == "RE1_before_firstRS")
    tyY = M.filter(pl.col("type") == "firstRS_before_RE1")
    print("type Y: first round_start tick vs freeze_end 1 tick (rs_t1 < fe1_t count):",
          tyY.filter(pl.col("rs_t1") < pl.col("fe1_t")).height, "of", tyY.filter(pl.col("fe1_t").is_not_null()).height)
    print("type X: round_end 1 tick vs freeze_end 1 tick (re1 > fe1 count):",
          tyX.filter(pl.col("re1_t") > pl.col("fe1_t")).height, "of", tyX.filter(pl.col("fe1_t").is_not_null()).height)
    # sample event sequences for one match of each type
    for typ in ["firstRS_before_RE1", "RE1_before_firstRS", "no_rs"]:
        sub = M.filter(pl.col("type") == typ).sort("match_key")
        if sub.height == 0:
            print(f"-- no match of type {typ}")
            continue
        mk = sub["match_key"][0]
        ev = []
        for r in RE.filter((pl.col("match_key") == mk) & (pl.col("round") <= 3)).iter_rows(named=True):
            ev.append((r["tick"], "round_end", r["round"]))
        for r in RS.filter((pl.col("match_key") == mk) & (pl.col("round") <= 3)).iter_rows(named=True):
            ev.append((r["tick"], "round_start", r["round"]))
        for r in ST.filter((pl.col("match_key") == mk) & pl.col("event_type").is_in(["round_freeze_end", "round_prestart", "round_officially_ended", "round_poststart", "buytime_ended"])).iter_rows(named=True):
            ev.append((r["tick"], r["event_type"], r["round"]))
        ev.sort()
        print(f"-- sample match type {typ}: {mk}")
        for e in ev[:14]:
            print(f"   tick={e[0]:>7}  {e[1]:<24} label={e[2]}")


if __name__ == "__main__":
    run("2024")
    run("2026")
    print("\nDONE")
