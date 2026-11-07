"""A6 final numbers on the fingerprint-deduplicated match set (both eras).

N1. 2024 round 13 / 25 inversion (W != S_ns) on deduped matches, and bomb-side inversion.
N2. Share of round_end rows whose tick lags the last death / bomb end (delayed round_end), per era.
N3. Round-1 absence in round_end on deduped matches (2024).
N4. Bomb-ended rounds coded 1/7 vs 8/9, deduped, per era.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A6_final_numbers.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

ROOT = Path(r"G:\pureskill\csds-cs2")
ERAS = {
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
}
T, CT = 2, 3


def files(base, ch):
    return sorted(glob.glob(str(sorted(base.glob(f"basic_{ch}.*"))[0] / "dataframe_*")))


def load(base, ch, cols):
    return pl.scan_parquet(files(base, ch)).select(cols).collect()


def main():
    for era, base in ERAS.items():
        print("=" * 90, f"\nERA {era}\n" + "=" * 90)
        RE = load(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).unique(
            ["match_key", "round"], keep="first")
        fp = (RE.sort(["match_key", "round"])
              .with_columns(pl.concat_str([pl.col("round").cast(pl.Utf8), pl.col("tick").cast(pl.Utf8),
                                           pl.col("winner_team_code").cast(pl.Utf8),
                                           pl.col("win_reason_code").cast(pl.Utf8)], separator=":").alias("part"))
              .group_by("match_key").agg(pl.col("part").alias("parts"))
              .with_columns(pl.col("parts").list.join("|").hash().alias("fph")))
        grp = fp.group_by("fph").agg(pl.col("match_key").sort().alias("keys"), pl.len().alias("n"))
        keep = set(grp["keys"].list.first().to_list())
        print(f"raw matches={fp.height}; fingerprint groups={grp.filter(pl.col('n') > 1).height}; "
              f"kept={len(keep)}; dropped copies={fp.height - len(keep)}")
        REk = RE.filter(pl.col("match_key").is_in(list(keep))).with_columns(
            pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day"))
        print(f"deduped round_end rows={REk.height}")

        # N3 round 1 absence on deduped matches
        mk = REk.group_by("match_key").agg((pl.col("round") == 1).any().alias("has_r1"))
        print(f"N3 deduped matches without round_end round 1: {mk.filter(~pl.col('has_r1')).height} of {mk.height}")

        # N2 delayed round_end: gap to last death, and bomb end position
        PD = load(base, "player_death", ["match_key", "round", "tick", "player_id"]).filter(
            pl.col("player_id").is_not_null())
        lastd = PD.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("last_death"))
        BS = load(base, "bomb_state", ["match_key", "round", "tick", "event_type"]).filter(
            pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        bend = BS.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("bomb_end_tick"),
                                                      pl.col("event_type").first().alias("bomb_type"))
        j = REk.join(lastd, on=["match_key", "round"], how="left").join(bend, on=["match_key", "round"], how="left")
        el = j.filter(pl.col("win_reason_code").is_in([8, 9]) & pl.col("last_death").is_not_null())
        n_el = el.height
        print(f"N2 code 8/9 rounds with a death: {n_el}; round_end tick == last death tick: "
              f"{el.filter(pl.col('tick') == pl.col('last_death')).height}; round_end > last death by >0 ticks: "
              f"{el.filter(pl.col('tick') > pl.col('last_death')).height}; "
              f"round_end < last death: {el.filter(pl.col('tick') < pl.col('last_death')).height}")
        bj = j.filter(pl.col("bomb_end_tick").is_not_null())
        print(f"N2 bomb-ended rounds: {bj.height}; round_end tick == bomb end tick: "
              f"{bj.filter(pl.col('tick') == pl.col('bomb_end_tick')).height}; round_end later: "
              f"{bj.filter(pl.col('tick') > pl.col('bomb_end_tick')).height}; round_end earlier: "
              f"{bj.filter(pl.col('tick') < pl.col('bomb_end_tick')).height}")

        # N4 bomb-ended rounds coded 1/7 vs 8/9 (deduped)
        bj2 = bj.with_columns(pl.when(pl.col("bomb_type") == "bomb_defused").then(pl.lit("defused"))
                              .otherwise(pl.lit("exploded")).alias("outcome"))
        n_b = bj2.height
        c17 = bj2.filter(pl.col("win_reason_code").is_in([1, 7])).height
        c89 = bj2.filter(pl.col("win_reason_code").is_in([8, 9])).height
        print(f"N4 deduped bomb-ended rounds={n_b}; coded 1/7={c17} ({c17 / n_b:.4f}); coded 8/9={c89} "
              f"({c89 / n_b:.4f})")
        print(bj2.group_by(["outcome", "win_reason_code"]).len().sort(["outcome", "win_reason_code"]))

        # N1 2024 inversion on deduped set
        if era == "2024":
            P2 = pl.read_parquet(Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad") / "A6_threeway_2024.parquet")
            P2k = P2.filter(pl.col("match_key").is_in(list(keep)))
            for r in (13, 25):
                s = P2k.filter((pl.col("round") == r) & pl.col("S_ns").is_not_null())
                inv = s.filter(pl.col("W") != pl.col("S_ns")).height
                print(f"N1 deduped round {r}: rows with score-derived winner={s.height}; inverted={inv} "
                      f"({inv / max(s.height, 1):.4f})")
            s13b = P2k.filter((pl.col("round") == 13) & pl.col("B").is_not_null())
            print(f"N1 deduped round 13 with bomb side: n={s13b.height}; W!=B: "
                  f"{s13b.filter(pl.col('W') != pl.col('B')).height}; S_ns!=B: "
                  f"{s13b.filter(pl.col('S_ns').is_not_null() & (pl.col('S_ns') != pl.col('B'))).height}")
            ot = P2k.filter((pl.col("rbin") == "other") & pl.col("S_ns").is_not_null())
            print(f"N1 deduped other rounds: score-derived winner known={ot.height}; "
                  f"disagree with W={ot.filter(pl.col('S_ns') != pl.col('W')).height}")
            print(f"N1 deduped round 13 and 25 total labels (round_end rows): "
                  f"{REk.filter(pl.col('round').is_in([13, 25])).height}")
            inv_rows = P2k.filter((pl.col("round").is_in([13, 25])) & pl.col("S_ns").is_not_null() &
                                  (pl.col("W") != pl.col("S_ns")))
            print("N1 inverted r13/25 rows (deduped) by round:", inv_rows.group_by("round").len().sort("round").rows())
            print("N1 inverted rows with bomb side (deduped):", inv_rows.filter(pl.col("B").is_not_null()).height,
                  " of which W==B-consistent-with-score? (B==S_ns):",
                  inv_rows.filter(pl.col("B").is_not_null() & (pl.col("B") == pl.col("S_ns"))).height)


if __name__ == "__main__":
    main()
