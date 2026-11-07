"""E2 probe 4: inspect signature duplicate groups in 2024 and 2026 (sizes, rounds, header agreement). Reads G: only."""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}


def read(base, suffix, ch, cols=None):
    fd = base / ("header" if ch == "header" else f"basic_{ch}{suffix}")
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


for era, (base, suffix) in TOMES.items():
    print(f"===== {era}")
    re_ = read(base, suffix, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    de = read(base, suffix, "player_death", ["match_key", "round", "tick", "player_id"])
    pi = read(base, suffix, "player_info", ["match_key", "round", "player_id", "team_code"])
    hdr = read(base, suffix, "header", ["key", "map_name", "match_date", "platform", "match_type"]).rename({"key": "match_key"})
    sig = collections.defaultdict(list)
    for x in re_.iter_rows(named=True):
        sig[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"]))))
    for x in de.iter_rows(named=True):
        sig[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
    for x in pi.iter_rows(named=True):
        sig[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
    groups = collections.defaultdict(list)
    for m, parts in sig.items():
        groups[tuple(sorted(parts))].append(m)
    dupg = [sorted(v) for v in groups.values() if len(v) > 1]
    sizes = collections.Counter(len(g) for g in dupg)
    print("group sizes:", sorted(sizes.items()))
    nrounds = re_.group_by("match_key").len().rename({"len": "n_rounds"})
    npc = re_.group_by("match_key").agg(pl.col("player_count").mode().first().alias("pc"))
    H = hdr.join(nrounds, on="match_key", how="left").join(npc, on="match_key", how="left")
    # show a sample of groups with header fields
    for g in dupg[:2] + dupg[-2:]:
        print("GROUP")
        print(H.filter(pl.col("match_key").is_in(g)).select(["match_key", "map_name", "match_date", "platform", "n_rounds", "pc"]).with_columns(pl.col("match_key").str.slice(0, 60)))
    # aggregate: rounds in duplicate groups
    mem = [m for g in dupg for m in g]
    H = H.with_columns(pl.col("match_key").is_in(mem).alias("in_dup_group"))
    print(H.group_by(["in_dup_group"]).agg([pl.len().alias("matches"), pl.col("n_rounds").median().alias("median_rounds"),
                                             pl.col("n_rounds").mean().alias("mean_rounds")]).rows())
    print("member pc distribution:", H.filter(pl.col("in_dup_group")).group_by("pc").len().sort("len", descending=True).head(6).rows())
    # date folder of the dup groups
    print("same date within groups:", all(len({tuple(m.split("/")[2:5]) for m in g}) == 1 for g in dupg))
    print("n_rounds of dup members (first 15 groups):", [[int(H.filter(pl.col('match_key') == m)['n_rounds'][0]) if H.filter(pl.col('match_key') == m).height else -1 for m in g] for g in dupg[:15]])
