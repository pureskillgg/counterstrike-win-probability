"""E4 verification: (1) does the 2024 round_end processing-version boundary line up with key-day 2024/02/12;
(2) winner_team_code vs win_reason_code consistency (documented enum: 2 = T, 3 = CT);
(3) key-path day vs header match_date agreement and mismatch pattern, both tomes.

Read-only, G: only. Run: cd G:\\pureskill\\csds-cs2 ; pixi run python -I -X utf8 scripts\\swarm\\e4_verify_version_dates.py
"""
import collections
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}
# Documented enum (notes/spec/csgo_and_docs_digest.md section 2.6): side codes 2 = T, 3 = CT.
WIN_T_REASONS = {1, 9, 13, 18}   # bomb exploded (T); CTs eliminated (T); time out hostage map (T); CTs surrendered (T)
WIN_CT_REASONS = {7, 8, 11, 12, 17}  # bomb defused; Ts eliminated; hostages rescued; time out no bomb; Ts surrendered (CT)


def read(tk, ch, cols):
    base, suf = TOMES[tk]
    folder = base / ("header" if ch == "header" else f"basic_{ch}{suf}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def main():
    for tk in ["2026", "2024"]:
        print(f"\n==================== TOME {tk} ====================")
        hdr = read(tk, "header", ["key", "match_date"])
        kd = [("/".join(k.split("/")[1:4])) for k in hdr["key"].to_list()]
        md = [(str(x)[:10].replace("-", "/") if x is not None else None) for x in hdr["match_date"].to_list()]
        agree = sum(1 for a, b in zip(kd, md) if a == b)
        print(f"header rows {hdr.height}; key-path day == match_date[:10] for {agree} ({agree / hdr.height:.4f})")
        mism = collections.Counter((a, b) for a, b in zip(kd, md) if a != b)
        print(f"mismatch pairs (key_day, match_date) top 12 of {sum(mism.values())}: " +
              "; ".join(f"{a}|{b}={n}" for (a, b), n in mism.most_common(12)))
        mdist = collections.Counter(b for b in md if b)
        print("match_date day counts (all): " + "; ".join(f"{d}={n}" for d, n in sorted(mdist.items())[:60]))

        re_ = read(tk, "round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "player_count"])
        re_ = re_.unique(subset=["match_key", "round"], keep="first")
        print(f"round_end unique rounds {re_.height}")
        print("winner_team_code counts: " + str(dict(sorted(collections.Counter(re_["winner_team_code"].to_list()).items(), key=lambda kv: str(kv[0])))))
        print("win_reason_code counts: " + str(dict(sorted(collections.Counter(re_["win_reason_code"].to_list()).items(), key=lambda kv: str(kv[0])))))
        pc = collections.Counter(re_["player_count"].to_list())
        print("player_count top: " + str(pc.most_common(6)))

        # winner side vs documented win reason
        ok = bad = other = 0
        bad_pairs = collections.Counter()
        for w, r in zip(re_["winner_team_code"].to_list(), re_["win_reason_code"].to_list()):
            if r in WIN_T_REASONS:
                exp = 2
            elif r in WIN_CT_REASONS:
                exp = 3
            else:
                other += 1
                continue
            if w == exp:
                ok += 1
            else:
                bad += 1
                bad_pairs[(w, r)] += 1
        print(f"winner vs reason (documented enum): consistent {ok}, inconsistent {bad}, reason not in enum {other}; "
              f"inconsistent pairs (winner, reason): {dict(bad_pairs.most_common(8))}")

        # rebuilt-version marker by key-day: share of rounds with player_count == 20 and reason in {8, 9}
        re_k = re_.with_columns(pl.col("match_key").str.split("/").list.slice(1, 3).list.join("/").alias("kday"))
        grp = (re_k.group_by("kday").agg([
            (pl.col("player_count") == 20).cast(pl.Int64).mean().alias("share_pc20"),
            pl.col("win_reason_code").is_in([8, 9]).cast(pl.Int64).mean().alias("share_reason_8_9"),
            pl.col("round").count().alias("rounds"),
        ]).sort("kday"))
        print("per key-day: kday | rounds | share player_count==20 | share win_reason in {8,9}")
        for row in grp.iter_rows():
            print(f"  {row[0]} | {row[3]} | {row[1]:.3f} | {row[2]:.3f}")


if __name__ == "__main__":
    main()
