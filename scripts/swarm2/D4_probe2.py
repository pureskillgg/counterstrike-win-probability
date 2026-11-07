"""D4 probe 2: event types, per-round row counts, duplicate live-start markers, and tick/second consistency.

Reads 2024 and 2026 tomes only (projected columns). Output goes to logs\\swarm2\\D4_probe2.log via the caller.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D4_probe2.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(12)
pl.Config.set_fmt_str_lengths(50)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"),
}


def read(base, ch, cols):
    fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


for era, base in TOMES.items():
    print(f"\n######## {era} ########")
    bs = read(base, "bomb_state", ["match_key", "round", "tick", "second", "event_type"])
    print("[bomb_state] event_type counts:")
    print(bs.group_by("event_type").len().sort("event_type"))
    # tick vs second consistency
    chk = bs.with_columns(((pl.col("second") * 64) - pl.col("tick")).abs().alias("d"))
    print("[bomb_state] |second*64 - tick| quantiles (0.5, 0.99, 1.0):",
          chk["d"].quantile(0.5), chk["d"].quantile(0.99), chk["d"].max())
    # rows per (match, round, event_type)
    per = bs.group_by(["match_key", "round", "event_type"]).len().rename({"len": "n_rows"})
    print("[bomb_state] rows per (match, round, event_type) distribution:")
    print(per.group_by(["event_type", "n_rows"]).len().sort(["event_type", "n_rows"]).head(30))
    # does a (match, round) have both planted and an outcome?
    outs = bs.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
    print("[bomb_state] outcome rows per (match,round) >1:",
          outs.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height)
    print("[bomb_state] distinct (match,round) with outcome:",
          outs.select(["match_key", "round"]).unique().height)

    bd = read(base, "bomb_defuse", ["match_key", "round", "tick", "event_type"])
    print("[bomb_defuse] event_type counts:")
    print(bd.group_by("event_type").len().sort("event_type"))

    ba = read(base, "bomb_action", ["match_key", "round", "tick", "event_type"])
    print("[bomb_action] event_type counts:")
    print(ba.group_by("event_type").len().sort("event_type"))

    rs = read(base, "round_state", ["match_key", "round", "tick", "event_type", "phase", "is_warmup"])
    print("[round_state] event_type counts:")
    print(rs.group_by("event_type").len().sort("event_type"))
    fe = rs.filter(pl.col("event_type") == "round_freeze_end")
    print("[round_state] round_freeze_end rows per (match,round) >1:",
          fe.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height,
          " distinct (match,round):", fe.select(["match_key", "round"]).unique().height)

    re_ = read(base, "round_end", ["match_key", "round", "tick", "winner_team_code"])
    print("[round_end] rows per (match,round) >1:",
          re_.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height,
          " distinct (match,round):", re_.select(["match_key", "round"]).unique().height)
    print("[round_end] round value counts (top 30 by round):")
    print(re_.group_by("round").len().sort("round").head(30))
    print("[round_end] max round per match quantiles:")
    print(re_.group_by("match_key").agg(pl.col("round").max().alias("mx"))["mx"].quantile(0.5),
          re_.group_by("match_key").agg(pl.col("round").max().alias("mx"))["mx"].max())

    # does the round_end tick sit after the freeze end tick for the same round?
    j = fe.select(["match_key", "round", pl.col("tick").alias("live")]).unique(["match_key", "round"]).join(
        re_.select(["match_key", "round", pl.col("tick").alias("end")]).unique(["match_key", "round"]),
        on=["match_key", "round"], how="inner")
    print("[window] live<end share:", (j["live"] < j["end"]).mean(), " n:", j.height)

    # bomb_state rows where the round index does not exist in round_end
    rk = re_.select(["match_key", "round"]).unique()
    orphan = bs.join(rk, on=["match_key", "round"], how="anti")
    print("[bomb_state] rows whose (match, round) is not in round_end:", orphan.height)
