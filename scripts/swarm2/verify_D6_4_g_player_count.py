"""D6-4 independent verifier, part G: what does round_end.player_count encode, by tome and by match format?
The claim's bomb filter is player_count == 20. Checks whether that filter is a 5v5 selector in 2024 and whether it
exists at all in 2026. Also checks the 16-round (format f16) roster size, and the build 9842 matches dated 2024-02-07..12.
"""
import sys, glob, collections
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
B24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
B26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
T, CT = 2, 3


def load(base, ch, cols):
    if ch == "header":
        fd = base / "header"
    else:
        fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f).select(cols) for f in files], how="vertical_relaxed").collect()


for name, base in [("2024", B24), ("2026", B26)]:
    RE = load(base, "round_end", ["match_key", "round", "player_count"])
    print(f"{name} round_end player_count values (count): {dict(collections.Counter(RE['player_count'].to_list()).most_common(8))}")

H = load(B24, "header", ["key", "build_num", "match_date"])
bm = {k: b for k, b, _ in H.iter_rows()}
md = {k: d[:10] for k, _, d in H.iter_rows()}
RS = load(B24, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"])
fe = RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick").unique(["match_key", "round"], keep="first")
scm = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    scm[m][r] = (t, c)


def sw_at(s, r):
    if r not in s or r + 1 not in s:
        return None
    t0, c0 = s[r]
    t1, c1 = s[r + 1]
    no = (t1 - t0, c1 - c0) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    if sw and not no:
        return "swap"
    if no and not sw:
        return "no"
    return None


fmt = {}
for m, s in scm.items():
    if sw_at(s, 8) == "swap":
        fmt[m] = "f16"
    elif sw_at(s, 12) == "swap":
        fmt[m] = "f24"
    else:
        fmt[m] = "none"
RE24 = load(B24, "round_end", ["match_key", "round", "player_count"]).unique(["match_key", "round"], keep="first")
pc_fmt = collections.defaultdict(collections.Counter)
for m, r, pc in RE24.iter_rows():
    if m in fmt:
        pc_fmt[fmt[m]][pc] += 1
for f in ["f16", "f24", "none"]:
    print(f"2024 player_count by format {f}: {dict(pc_fmt[f].most_common(6))}")

PI = load(B24, "player_info", ["match_key", "round", "player_id", "team_code"]).unique(["match_key", "round", "player_id"], keep="first")
size = collections.defaultdict(lambda: collections.Counter())
for m, r, t in PI.select(["match_key", "round", "team_code"]).iter_rows():
    pass
cnt = collections.defaultdict(collections.Counter)
for (m, r, t), n in collections.Counter(PI.select(["match_key", "round", "team_code"]).iter_rows()).items():
    cnt[(m, r)][t] = n
for f in ["f16", "f24"]:
    c = collections.Counter()
    for (m, r), d in cnt.items():
        if fmt.get(m) == f and r == 9:
            c[(d.get(T, 0), d.get(CT, 0))] += 1
    print(f"player_info roster (T,CT) in round 9, format {f}: {dict(c.most_common(5))}")

print("\nBuild 9842 matches with header date 2024-02-07..2024-02-12 (for the build-vs-date question):")
sub = [(m, bm[m], md[m]) for m in bm if bm[m] == 9842 and "2024-02-07" <= md[m] <= "2024-02-12"]
print(f"  count={len(sub)}")
RE_all = load(B24, "round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
BS = load(B24, "bomb_state", ["match_key", "round", "event_type"])
outs = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
outs = outs.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(CT).otherwise(T).alias("bomb_side"))
J = RE_all.join(outs, on=["match_key", "round"], how="inner")
sel = {m for m, _, _ in sub}
Js = J.filter(pl.col("match_key").is_in(list(sel)) & (pl.col("round") == 13))
print(f"  round-13 bomb rows in those matches (any player_count): n={Js.height}; mismatch={int((Js['bomb_side'] != Js['winner_team_code']).sum())}")
print(f"  their player_count values: {dict(collections.Counter(Js['player_count'].to_list()))}")
Jb = J.filter(pl.col("match_key").is_in(list(sel)))
print(f"  all-round bomb rows in those matches: n={Jb.height}; mismatch={int((Jb['bomb_side'] != Jb['winner_team_code']).sum())}")
print(f"  their player_count values: {dict(collections.Counter(Jb['player_count'].to_list()))}")
