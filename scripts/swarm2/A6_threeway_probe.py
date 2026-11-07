"""A6 follow-up probes for the label audit (both eras).

P1. Round-number gaps in round_end per match (2024 and 2026): missing-round histogram.
P2. Three-way concordance per round: round_end winner (W), score-derived winner without swap (S_ns),
    score-derived winner with swap (S_sw), bomb side (B: defused -> CT, exploded -> T).
    Split by round bin (13, 25, other) and by match day (2024).
P3. Absolute death check: distinct victims per side in rounds coded 8 / 9 (5 per side expected),
    not depending on the roster table.
P4. 2026: the rounds where two score_update rows show +1 for different sides.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A6_threeway_probe.py
"""
import collections
import glob
import re
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
ERAS = {
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
}
T, CT = 2, 3


def files(base, ch):
    folder = base / "header" if ch == "header" else sorted(base.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(folder / "dataframe_*")))


def load(base, ch, cols):
    return pl.scan_parquet(files(base, ch)).select(cols).collect()


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def main():
    for era, base in ERAS.items():
        section(f"ERA {era}")
        RE = load(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).unique(
            ["match_key", "round"], keep="first").with_columns(
            pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day"))

        # P1 gaps
        pm = RE.group_by("match_key").agg(pl.col("round").sort().alias("rs"), pl.col("round").max().alias("mx"),
                                          pl.col("round").min().alias("mn"), pl.len().alias("n"))
        pm = pm.with_columns(
            (pl.col("mx") - pl.col("n")).alias("gap_count"),
            pl.col("rs").list.eval(pl.element().diff().fill_null(1)).list.max().alias("max_step"))
        print("P1 matches by (min round, max round, rows) top 10 patterns:")
        print(pm.group_by(["mn", "mx", "n"]).len().sort("len", descending=True).head(10))
        print("P1 matches with a gap (max_step>1):", pm.filter(pl.col("max_step") > 1).height, "of", pm.height)
        miss = []
        for rs, mx in pm.select(["rs", "mx"]).iter_rows():
            have = set(rs)
            miss.extend(r for r in range(1, mx + 1) if r not in have)
        mh = collections.Counter(miss)
        print("P1 missing round numbers (within 1..max), top 15:", mh.most_common(15))

        # P2 three-way concordance
        RS = load(base, "round_state", ["match_key", "round", "event_type", "t_score", "ct_score"]).filter(
            pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first")
        sc = {(m, r): (t, c) for m, r, t, c in RS.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
        BS = load(base, "bomb_state", ["match_key", "round", "event_type"]).filter(
            pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        bo = BS.group_by(["match_key", "round"]).agg(
            (pl.col("event_type") == "bomb_defused").sum().alias("nd"),
            (pl.col("event_type") == "bomb_exploded").sum().alias("ne"))
        bside = {(m, r): (CT if nd > 0 else T) for m, r, nd, ne in bo.select(["match_key", "round", "nd", "ne"]).iter_rows()
                 if (nd > 0) != (ne > 0)}
        rows = []
        for m, r, w, code, day in RE.select(["match_key", "round", "winner_team_code", "win_reason_code", "day"]).iter_rows():
            a, b = sc.get((m, r)), sc.get((m, r + 1))
            s_ns = s_sw = None
            if a is not None and b is not None:
                dt_, dc_ = b[0] - a[0], b[1] - a[1]
                s_ns = T if (dt_, dc_) == (1, 0) else (CT if (dt_, dc_) == (0, 1) else None)
                s_sw = T if (dt_, dc_) == (0, 1) else (CT if (dt_, dc_) == (1, 0) else None)
            rows.append((m, r, day, w, code, s_ns, s_sw, bside.get((m, r))))
        P2 = pl.DataFrame(rows, schema=["match_key", "round", "day", "W", "code", "S_ns", "S_sw", "B"], orient="row")
        P2 = P2.with_columns(
            pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
            .otherwise(pl.lit("other")).alias("rbin"))
        P2.write_parquet(Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad") / f"A6_threeway_{era}.parquet")
        bb = P2.filter(pl.col("B").is_not_null())
        print(f"P2 rounds with bomb side: {bb.height}")
        print("P2 bomb-side concordance by round bin (W==B, S_ns==B, S_sw==B; over rounds with B and that source):")
        out = bb.group_by("rbin").agg(
            pl.len().alias("n"),
            (pl.col("W") == pl.col("B")).sum().alias("W_eq_B"),
            (pl.col("S_ns") == pl.col("B")).sum().alias("Sns_eq_B"),
            pl.col("S_ns").is_not_null().sum().alias("Sns_known"),
            (pl.col("S_sw") == pl.col("B")).sum().alias("Ssw_eq_B"),
            pl.col("S_sw").is_not_null().sum().alias("Ssw_known")).sort("rbin")
        print(out)
        if era == "2024":
            print("P2 2024 r13 by match day (n, W==B, S_ns==B, S_sw==B):")
            r13 = bb.filter(pl.col("round") == 13).group_by("day").agg(
                pl.len().alias("n"),
                (pl.col("W") == pl.col("B")).sum().alias("W_eq_B"),
                (pl.col("S_ns") == pl.col("B")).sum().alias("Sns_eq_B"),
                (pl.col("S_sw") == pl.col("B")).sum().alias("Ssw_eq_B")).sort("day")
            print(r13)
            print("P2 2024 r13 three-way pattern (W vs S_ns vs B), rows with S_ns and B known:")
            r13s = P2.filter((pl.col("round") == 13) & pl.col("S_ns").is_not_null() & pl.col("B").is_not_null())
            print(r13s.group_by(["day", (pl.col("W") == pl.col("S_ns")).alias("W_eq_Sns"),
                                 (pl.col("W") == pl.col("B")).alias("W_eq_B"),
                                 (pl.col("S_ns") == pl.col("B")).alias("Sns_eq_B")]).len().sort(
                ["day", "W_eq_Sns", "W_eq_B", "Sns_eq_B"]))
            print("P2 2024 'other' rounds with bomb side, by code (n, W==B, S_ns==W):")
            oth = P2.filter((pl.col("rbin") == "other") & pl.col("B").is_not_null())
            print(oth.group_by("code").agg(pl.len().alias("n"), (pl.col("W") == pl.col("B")).sum().alias("W_eq_B"),
                                           (pl.col("S_ns") == pl.col("W")).sum().alias("Sns_eq_W"),
                                           pl.col("S_ns").is_not_null().sum().alias("Sns_known")).sort("code"))
            print("P2 2024 'other' rounds, code-8/9 defused or exploded: S_ns vs W vs B")
            ce = P2.filter((pl.col("rbin") == "other") & pl.col("code").is_in([8, 9]) & pl.col("B").is_not_null())
            print(ce.group_by(["code", "B"]).agg(pl.len().alias("n"), (pl.col("S_ns") == pl.col("W")).sum().alias("Sns_eq_W"),
                                                 pl.col("S_ns").is_not_null().sum().alias("Sns_known")).sort(["code", "B"]))

        # P3 absolute victims
        PD = load(base, "player_death", ["match_key", "round", "player_id", "player_team_code"]).filter(
            pl.col("player_id").is_not_null()).unique(["match_key", "round", "player_id"])
        vic = PD.group_by(["match_key", "round", "player_team_code"]).agg(pl.len().alias("nv"))
        vw = vic.pivot(on="player_team_code", index=["match_key", "round"], values="nv").rename(
            {"2": "t_victims", "3": "ct_victims"})
        for c in ["t_victims", "ct_victims"]:
            if c not in vw.columns:
                vw = vw.with_columns(pl.lit(0).cast(pl.UInt32).alias(c))
        P3 = RE.select(["match_key", "round", "win_reason_code", "winner_team_code", "day"]).join(
            vw, on=["match_key", "round"], how="left").with_columns(
            pl.col("t_victims").fill_null(0), pl.col("ct_victims").fill_null(0))
        print("P3 absolute victims check (5 per side expected; victim side from player_death.player_team_code)")
        c8 = P3.filter(pl.col("win_reason_code") == 8)
        c9 = P3.filter(pl.col("win_reason_code") == 9)
        print(f"  code 8 rounds={c8.height}: T victims < 5: {c8.filter(pl.col('t_victims') < 5).height}; "
              f"T victims == 0: {c8.filter(pl.col('t_victims') == 0).height}")
        print(f"  code 9 rounds={c9.height}: CT victims < 5: {c9.filter(pl.col('ct_victims') < 5).height}; "
              f"CT victims == 0: {c9.filter(pl.col('ct_victims') == 0).height}")
        print("  code 8 T-victims distribution:", c8.group_by("t_victims").len().sort("t_victims").rows())
        print("  code 9 CT-victims distribution:", c9.group_by("ct_victims").len().sort("ct_victims").rows())
        print("  victim sides observed:", vic.group_by("player_team_code").len().sort("player_team_code").rows())
        if era == "2024":
            bbm = P3.join(P2.select(["match_key", "round", "B"]), on=["match_key", "round"], how="left")
            defc8 = bbm.filter((pl.col("B") == CT) & (pl.col("win_reason_code") == 8))
            print(f"  2024 defused rounds coded 8: n={defc8.height}; T victims<5: {defc8.filter(pl.col('t_victims') < 5).height}")
            print("  2024 defused rounds coded 8 T-victims distribution:",
                  defc8.group_by("t_victims").len().sort("t_victims").rows())
            defc7 = bbm.filter((pl.col("B") == CT) & (pl.col("win_reason_code") == 7))
            print(f"  2024 defused rounds coded 7: n={defc7.height}; T victims distribution:",
                  defc7.group_by("t_victims").len().sort("t_victims").rows())
        else:
            bbm = P3.join(P2.select(["match_key", "round", "B"]), on=["match_key", "round"], how="left")
            defc7 = bbm.filter((pl.col("B") == CT) & (pl.col("win_reason_code") == 7))
            print(f"  2026 defused rounds coded 7: n={defc7.height}; T victims distribution:",
                  defc7.group_by("t_victims").len().sort("t_victims").rows())

        # P4 2026 multi-side score_update rounds
        if era == "2026":
            section("P4 2026 rounds with +1 score_update for both sides")
            su = load(base, "score_update", ["match_key", "round", "tick", "team_code", "old_score", "new_score"])
            up = su.filter(pl.col("new_score") - pl.col("old_score") == 1)
            multi = up.group_by(["match_key", "round"]).agg(pl.col("team_code").n_unique().alias("nt"),
                                                            pl.len().alias("n")).filter(pl.col("nt") > 1)
            print("P4 rounds with +1 for both sides:", multi.height)
            print("P4 n +1 rows per such round:", multi.group_by("n").len().rows())
            ex = multi.head(3).select(["match_key", "round"]).rows()
            for m, r in ex:
                print("  example", m[-40:], "round", r)
                print(up.filter((pl.col("match_key") == m) & (pl.col("round") == r)).sort("tick").select(
                    ["tick", "team_code", "old_score", "new_score"]))
            P4 = P2.join(multi.select(["match_key", "round"]).with_columns(pl.lit(True).alias("multi")),
                         on=["match_key", "round"], how="left").with_columns(pl.col("multi").fill_null(False))
            print("P4 multi-side rounds vs noswap score disagreement:")
            print(P4.filter(pl.col("S_ns").is_not_null()).group_by(["multi", (pl.col("S_ns") != pl.col("W"))
                                                                    .alias("Sns_ne_W")]).len().sort(["multi", "Sns_ne_W"]))


if __name__ == "__main__":
    main()
