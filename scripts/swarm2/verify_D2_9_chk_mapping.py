"""verify_D2_9 event-mapping check.

Question: which round_state game events (event_type) do the basic_round_start and basic_round_end channels
coincide with, per match and round, in 2024 and 2026? If the 2024 basic_round_start channel is an exact copy of
a different game event (for example round_prestart, which fires after the previous round ends), then the
"round_start after round_end" finding is a channel-mapping property, not a property of the match timeline.

For each (match, round) key we take the first tick of each event_type in round_state and of each anchor
channel, then count exact tick matches and median differences. Read-only; one channel at a time.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_chk_mapping.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def parts(label, ch):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))
    assert len(fd) == 1, (label, ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


def first_per_key(df):
    return df.sort("tick").unique(["match_key", "round"], keep="first")


for label in ["2024", "2026"]:
    print(f"\n==================== {label} ====================")
    rst = pl.scan_parquet(parts(label, "round_state")).select(
        ["match_key", "round", "tick", "event_type"]).collect()
    anchors = {}
    for ch in ["round_start", "round_end"]:
        anchors[ch] = first_per_key(pl.scan_parquet(parts(label, ch)).select(
            ["match_key", "round", "tick"]).collect()).rename({"tick": f"{ch}_tick"})

    for ch in ["round_start", "round_end"]:
        a = anchors[ch]
        n_keys = a.height
        print(f"\n-- anchor basic_{ch}: keys={n_keys:,}  (rows in channel used as anchor, first per key)")
        rows = []
        for et in rst["event_type"].unique().to_list():
            e = first_per_key(rst.filter(pl.col("event_type") == et)).select(
                ["match_key", "round", pl.col("tick").alias("e_tick")])
            j = a.join(e, on=["match_key", "round"], how="inner")
            if j.height == 0:
                continue
            d = (j["e_tick"] - j[f"{ch}_tick"])
            exact = int((d == 0).sum())
            rows.append({
                "event_type": et,
                "keys_matched": j.height,
                "frac_of_anchor_keys": round(j.height / n_keys, 4),
                "exact_tick_equal": exact,
                "frac_exact": round(exact / j.height, 4),
                "median_e_minus_anchor": float(d.median()),
                "p10": float(d.quantile(0.10)),
                "p90": float(d.quantile(0.90)),
            })
        out = pl.DataFrame(rows).sort("frac_exact", descending=True)
        print(out.head(12))

    # round_end channel vs round_start channel directly: does round_start(r) line up with round_end(r-1)?
    rs = anchors["round_start"]
    re_ = anchors["round_end"]
    prev = re_.with_columns((pl.col("round") + 1).alias("round"))  # round_end of r-1 keyed as r
    jp = rs.join(prev.rename({"round_end_tick": "end_prev_tick"}), on=["match_key", "round"], how="inner")
    dp = jp["round_start_tick"] - jp["end_prev_tick"]
    print(f"\n-- round_start(r) - round_end(r-1) ticks: keys={jp.height:,}  "
          f"median={dp.median()}  frac<0={float((dp < 0).mean()):.4f}  frac==0={float((dp == 0).mean()):.4f}")
    print(f"   quantiles p01={dp.quantile(0.01)} p10={dp.quantile(0.10)} p50={dp.median()} "
          f"p90={dp.quantile(0.90)} p99={dp.quantile(0.99)}")
