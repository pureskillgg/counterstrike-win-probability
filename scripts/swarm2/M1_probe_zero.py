import glob
import collections
import polars as pl

ROOT = "G:/pureskill/csds-cs2/tomes/"
ERAS = {
    "2026": ("2026-10-01_to_08/tome/csds", ".2026-09-18,2026-10-08"),
    "2024": ("2024-02-g/tome/csds", ".2023-12-28,2024-02-25"),
}


def read(era, ch, cols):
    base, suf = ERAS[era]
    folder = ROOT + base + ("/header" if ch == "header" else f"/basic_{ch}{suf}")
    files = sorted(glob.glob(folder + "/dataframe_*"))
    frames = []
    for f in files:
        if len(pl.read_parquet_schema(f)) == 0:
            continue
        frames.append(pl.read_parquet(f, columns=cols))
    return pl.concat(frames, how="vertical_relaxed") if frames else None


for era in ["2026", "2024"]:
    H = read(era, "header", ["key", "map_name", "match_type", "game", "match_date"])
    cols = [c for c in ["match_type", "game"] if c in H.columns]
    print(f"== {era} header: map x match_type/game (counts)")
    print(H.group_by(["map_name"] + cols).agg(pl.len().alias("n")).sort(["map_name"] + cols))
    RE = read(era, "round_end", ["match_key", "round", "player_count"])
    pc = RE.join(H.select(["key", "map_name"]).rename({"key": "match_key"}), on="match_key", how="left")
    print(f"== {era} round_end player_count by map (top values)")
    for mp in sorted(pc["map_name"].drop_nulls().unique().to_list()):
        sub = pc.filter(pl.col("map_name") == mp)
        vc = sub.group_by("player_count").agg(pl.len().alias("n")).sort("n", descending=True).head(4)
        print(f"  {mp}: rounds={sub.height} top_pc={vc.rows()}")
    if era == "2026":
        BS = read(era, "bomb_state", ["match_key", "event_type"])
        bm = set(BS["match_key"].unique().to_list()) if BS is not None else set()
        allm = H.select(["key", "map_name"]).rename({"key": "match_key"})
        nob = allm.filter(~pl.col("match_key").is_in(list(bm)))
        print(f"== {era} header matches with zero bomb_state rows, by map:")
        print(nob.group_by("map_name").agg(pl.len().alias("n")).sort("n", descending=True))
    if era == "2024":
        BS = read(era, "bomb_state", ["match_key", "event_type"])
        bm = set(BS["match_key"].unique().to_list())
        allm = H.select(["key", "map_name"]).rename({"key": "match_key"})
        nob = allm.filter(~pl.col("match_key").is_in(list(bm)))
        print(f"== {era} header matches with zero bomb_state rows, by map:")
        print(nob.group_by("map_name").agg(pl.len().alias("n")).sort("n", descending=True))
        # planted rounds in matches that do have bomb rows, per map
        pl_ev = BS.filter(pl.col("event_type") == "bomb_planted")
        print("bomb_state event types:", BS["event_type"].unique().sort().to_list())
