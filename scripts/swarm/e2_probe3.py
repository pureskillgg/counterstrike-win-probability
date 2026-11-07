"""E2 probe 3: match mode per era, death window, id mapping, team codes, duplicate signatures. Reads G: only."""
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
    re_ = read(base, suffix, "round_end")
    # match mode from modal player_count
    mm = (re_.group_by(["match_key", "player_count"]).len()
          .sort(["match_key", "len"], descending=[False, True])
          .group_by("match_key", maintain_order=True).first())
    print("modal player_count per match:", mm.group_by("player_count").len().sort("len", descending=True).head(12).rows())
    print("rounds with mixed player_count within match: matches =",
          re_.group_by("match_key").agg(pl.col("player_count").n_unique().alias("n")).filter(pl.col("n") > 1).height)
    fe = read(base, suffix, "round_state", ["match_key", "round", "tick", "event_type"]).filter(pl.col("event_type") == "round_freeze_end")
    fe = fe.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("live_tick"))
    rs_round = re_.select(["match_key", "round", "tick", "winner_team_code", "player_count"]).rename({"tick": "end_tick"})
    rnd = rs_round.join(fe, on=["match_key", "round"], how="left")
    print("rounds total", rnd.height, "no live_tick", rnd.filter(pl.col("live_tick").is_null()).height)
    rnd = rnd.with_columns((pl.col("end_tick") - pl.col("live_tick")).alias("live_len_ticks"))
    print("live length ticks describe:", rnd["live_len_ticks"].describe().rows())
    print("live length negative count:", rnd.filter(pl.col("live_len_ticks") < 0).height)
    rst = read(base, suffix, "round_start", ["match_key", "round", "tick"]).rename({"tick": "start_tick"})
    j = rnd.join(rst, on=["match_key", "round"], how="left")
    print("round_start tick==66 share (all rounds):", round(j.filter(pl.col("start_tick") == 66).height / j.height, 4),
          " round1 share:", round(j.filter((pl.col("round") == 1) & (pl.col("start_tick") == 66)).height / max(1, j.filter(pl.col("round") == 1).height), 4))
    # deaths
    de = read(base, suffix, "player_death", ["match_key", "round", "tick", "player_id", "attacker_id", "assister_id",
                                             "player_id_fixed", "attacker_id_fixed", "player_team_code", "attacker_team_code"])
    print("death rows", de.height)
    dj = de.join(rnd.select(["match_key", "round", "live_tick", "end_tick"]), on=["match_key", "round"], how="left")
    print("deaths without round match:", dj.filter(pl.col("end_tick").is_null()).height)
    inwin = dj.filter((pl.col("tick") >= pl.col("live_tick")) & (pl.col("tick") <= pl.col("end_tick")))
    print("deaths in live window:", inwin.height, "share", round(inwin.height / dj.height, 4),
          " before live:", dj.filter(pl.col("tick") < pl.col("live_tick")).height,
          " after end:", dj.filter(pl.col("tick") > pl.col("end_tick")).height)
    print("team codes player:", de.group_by("player_team_code").len().rows(), " attacker:", de.group_by("attacker_team_code").len().rows())
    # null attacker analysis
    na = de.filter(pl.col("attacker_id_fixed").is_null())
    print("attacker_id_fixed null rows", na.height, "raw attacker_id top:", na.group_by("attacker_id").len().sort("len", descending=True).head(6).rows())
    print("attacker_id_fixed null & attacker_team_code null:", na.filter(pl.col("attacker_team_code").is_null()).height)
    pn = de.filter(pl.col("player_id_fixed").is_null())
    print("player_id_fixed null rows", pn.height, "raw player_id top:", pn.group_by("player_id").len().sort("len", descending=True).head(4).rows())
    # same slot kills
    same = de.filter(pl.col("attacker_id_fixed") == pl.col("player_id_fixed")).height
    print("self-kill (fixed ids equal) rows", same)
    tk = de.filter((pl.col("attacker_team_code").is_not_null()) & (pl.col("attacker_team_code") == pl.col("player_team_code"))
                   & (pl.col("attacker_id_fixed") != pl.col("player_id_fixed"))).height
    print("team-kill rows (same team codes, different slot)", tk)
    print("fixed id ranges: player", de["player_id_fixed"].min(), de["player_id_fixed"].max(), " attacker", de["attacker_id_fixed"].min(), de["attacker_id_fixed"].max())
    # signature duplicates
    sig = collections.defaultdict(list)
    for x in re_.iter_rows(named=True):
        sig[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"]))))
    for x in de.select(["match_key", "round", "tick", "player_id"]).iter_rows(named=True):
        sig[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
    pi = read(base, suffix, "player_info", ["match_key", "round", "player_id", "team_code"])
    for x in pi.iter_rows(named=True):
        sig[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
    groups = collections.defaultdict(list)
    for m, parts in sig.items():
        groups[tuple(sorted(parts))].append(m)
    dupg = [sorted(v) for v in groups.values() if len(v) > 1]
    print("signature duplicate groups:", len(dupg), "members:", sum(len(g) for g in dupg), "removed:", sum(len(g) - 1 for g in dupg))
    print("sig-matches-count", len(sig))
    print("example group", dupg[0] if dupg else None)
