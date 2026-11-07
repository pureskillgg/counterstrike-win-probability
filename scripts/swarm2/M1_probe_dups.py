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
    frames = []
    for f in sorted(glob.glob(folder + "/dataframe_*")):
        if len(pl.read_parquet_schema(f)) == 0:
            continue
        frames.append(pl.read_parquet(f, columns=cols))
    return pl.concat(frames, how="vertical_relaxed")


for era in ["2026", "2024"]:
    H = read(era, "header", ["key", "map_name", "match_id", "sharecode", "demo_id", "match_date"])
    hdr = {r["key"]: r for r in H.iter_rows(named=True)}
    RE = read(era, "round_end", ["match_key", "round", "tick", "winner_team_code"])
    DE = read(era, "player_death", ["match_key", "round", "tick", "player_id"])
    PI = read(era, "player_info", ["match_key", "round", "player_id", "team_code"])
    parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), str(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
    groups = collections.defaultdict(list)
    for m, v in parts.items():
        groups[tuple(sorted(v))].append(m)
    dg = [g for g in groups.values() if len(g) > 1]
    same_sc = same_demo = same_mid = same_date = 0
    maps = collections.Counter()
    for g in dg:
        rows = [hdr[m] for m in g if m in hdr]
        if len({r["sharecode"] for r in rows}) == 1:
            same_sc += 1
        if len({r["demo_id"] for r in rows}) == 1:
            same_demo += 1
        if len({r["match_id"] for r in rows}) == 1:
            same_mid += 1
        if len({r["match_date"] for r in rows}) == 1:
            same_date += 1
        maps[rows[0]["map_name"]] += len(g) - 1
    print(f"== {era}: dup groups (any size>1) = {len(dg)}; members beyond first = {sum(len(g)-1 for g in dg)}")
    print(f"   groups with identical sharecode: {same_sc}; identical demo_id: {same_demo}; "
          f"identical match_id: {same_mid}; identical match_date: {same_date}")
    print(f"   extra copies by map: {dict(maps)}")
    # sizes of signatures for a few groups
    sizes = sorted(len(next(iter(groups.keys()))) for _ in [0]) if False else None
    ex = [g for g in dg][:2]
    for g in ex:
        print("   example group:", [(m.split('/')[-2][:8], hdr[m]["map_name"], hdr[m]["match_date"], hdr[m]["match_id"][:8]) for m in g])
