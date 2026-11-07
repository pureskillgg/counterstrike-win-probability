"""D3: row-level comparison of the 2024 G: tome against the raw E: tail copy.

Hypothesis D3: the G: tome differs from the raw E: source for the affected 2024-02-12..25 rounds
(round 13 side vs bomb outcome defect).

Test
  1. Inventory from the G: 2024 tome: round_end + bomb_state. Round-13 mismatch flag replicates
     scripts/swarm/e2_side_check_r13.py (player_count == 20; bomb side 3 if bomb_defused else 2;
     mismatch if bomb side != winner_team_code).
  2. Sample (seed 0, ASSUMPTION): 10 affected matches (day 12..25 with a round-13 mismatch) and
     10 clean matches (day 01..07 with a round-13 bomb outcome and no mismatch).
  3. For each sampled match, for channels round_end, round_state, bomb_state, bomb_action: load the G:
     rows (filter match_key) and the E: parquet file (E:\\...\\2024\\MM\\DD\\<uuid>\\<channel>), normalise
     E: (drop __index_level_0__, add match_key), then compare the full multiset of rows on all shared
     columns, using exact values (numeric and bool cast to Float64 on both sides, then to string).
  4. Round-13 detail: winner_team_code and bomb events for rounds 12, 13, 14 on both sides, and the
     e2 mismatch flag recomputed from each side.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D3_compare_tome_vs_raw.py [all]
  With no argument: 10 + 10 sample. With 'all': every 2024 match with a round-13 bomb outcome in
  days 01..25 (larger E: read, same checks).
"""
import collections
import glob
import random
import sys
import time
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

MODE = sys.argv[1] if len(sys.argv) > 1 else "sample"
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
E_BASE = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds")
CHANNELS = ["round_end", "round_state", "bomb_state", "bomb_action"]
SEED = 0
N_SAMPLE = 10
FOCUS = 13
T_CODE, CT_CODE = 2, 3
NON_DATA = {"match_key", "__index_level_0__"}

t0 = time.time()


def log(*a):
    print(*a, flush=True)


def g_read(ch, cols=None):
    folder = TOME / f"basic_{ch}{SUF}"
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    if not files:
        raise SystemExit(f"no pages for {ch}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


# ---------- 1. inventory from G: round_end and bomb_state
RE_G = g_read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
BS_G = g_read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
log("== 1. G: inventory ==")
log(f"round_end rows {RE_G.height}; matches {RE_G['match_key'].n_unique()}")
log("player_count value counts (round_end):")
log(RE_G.group_by("player_count").len().sort("player_count"))
dup = RE_G.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
log(f"duplicate (match, round) rows in round_end: {dup.height}")

outcome = (BS_G.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
flag = (RE_G.filter(pl.col("player_count") == 20)
        .join(outcome, on=["match_key", "round"], how="inner")
        .with_columns(
            pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT_CODE)).otherwise(pl.lit(T_CODE))
            .alias("bomb_side"),
            pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day"))
        .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch")))
log(f"round-with-outcome rows (player_count==20): {flag.height}; mismatches: {int(flag['mismatch'].sum())}")
r13 = flag.filter(pl.col("round") == FOCUS)
log(f"round-13 with outcome: {r13.height}; mismatches: {int(r13['mismatch'].sum())}")
mm_by_day = r13.group_by("day").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort("day")
log("round-13 by day (day, n, mm):", mm_by_day.rows())

aff_pool = sorted(r13.filter((pl.col("day") >= 12) & pl.col("mismatch"))["match_key"].unique().to_list())
clean_pool = sorted(r13.filter((pl.col("day") <= 7) & (~pl.col("mismatch")))["match_key"].unique().to_list())
log(f"affected pool (day 12..25, round-13 mismatch): {len(aff_pool)} matches")
log(f"clean pool (day 01..07, round-13 outcome, no mismatch): {len(clean_pool)} matches")

CH_FILES = ["round_end", "round_state", "bomb_state", "bomb_action"]


def e_has_all(key):
    parts = key.split("/")
    d = E_BASE / parts[1] / parts[2] / parts[3] / parts[4]
    return all((d / c).exists() for c in CH_FILES)


clean_usable = [k for k in clean_pool if e_has_all(k)]
aff_usable = [k for k in aff_pool if e_has_all(k)]
late_clean_pool = sorted(set(r13.filter(pl.col("day") >= 12)["match_key"].to_list()) - set(aff_pool))
late_clean_usable = [k for k in late_clean_pool if e_has_all(k)]
log(f"E: usable: affected {len(aff_usable)}/{len(aff_pool)}; clean day01-07 {len(clean_usable)}/{len(clean_pool)}; "
    f"late clean day12-25 {len(late_clean_usable)}/{len(late_clean_pool)}")

if MODE == "all":
    sel_aff = sorted(aff_usable)
    sel_clean = sorted(clean_usable + late_clean_usable)
    log(f"MODE all: affected {len(sel_aff)}; clean (day01-07 and day12-25 with no round-13 mismatch) {len(sel_clean)}")
else:
    rng = random.Random(SEED)
    sel_aff = sorted(rng.sample(aff_usable, N_SAMPLE))
    sel_clean = sorted(rng.sample(clean_usable, N_SAMPLE))
log("affected sample (ASSUMPTION seed 0):" if MODE != "all" else "affected (all usable):")
for k in sel_aff[:50]:
    log("  ", k)
log("clean sample (ASSUMPTION seed 0):" if MODE != "all" else "clean (all usable):")
for k in sel_clean[:50]:
    log("  ", k)

# ---------- preload G: channels (small) and split per match
want = set(sel_aff) | set(sel_clean)
G = {}
for ch in CHANNELS:
    df = g_read(ch).filter(pl.col("match_key").is_in(sorted(want)))
    G[ch] = {k: v for k, v in df.partition_by("match_key", as_dict=True).items()}
    G[ch] = {k[0] if isinstance(k, tuple) else k: v for k, v in G[ch].items()}
log(f"G: rows loaded for selected matches: " +
    ", ".join(f"{ch}={sum(v.height for v in G[ch].values())}" for ch in CHANNELS))


def e_path(key, ch):
    parts = key.split("/")  # csds / 2024 / MM / DD / uuid / csds
    return E_BASE / parts[1] / parts[2] / parts[3] / parts[4] / ch


def e_read(key, ch):
    p = e_path(key, ch)
    if not p.exists():
        return None, str(p)
    df = pl.read_parquet(str(p))
    return df, str(p)


def norm_E(df, key):
    df = df.drop([c for c in NON_DATA if c in df.columns and c != "match_key"])
    return df.with_columns(pl.lit(key).alias("match_key"))


def row_strings(df, cols):
    if df.height == 0:
        return []
    exprs = []
    for c in cols:
        dt = df.schema[c]
        if dt.is_numeric() or dt == pl.Boolean:
            e = pl.col(c).cast(pl.Float64).cast(pl.Utf8)
        else:
            e = pl.col(c).cast(pl.Utf8)
        exprs.append(e.fill_null("<NULL>"))
    return df.select(pl.concat_str(exprs, separator="|").alias("k"))["k"].to_list()


def compare_channel(g_df, e_df):
    """Return (n_g, n_e, shared_cols, cols_only_g, cols_only_e, only_g, only_e, sample rows)."""
    g_cols = [c for c in g_df.columns if c not in NON_DATA]
    e_cols = [c for c in e_df.columns if c not in NON_DATA]
    shared = [c for c in g_cols if c in e_cols]
    only_g_cols = [c for c in g_cols if c not in e_cols]
    only_e_cols = [c for c in e_cols if c not in g_cols]
    gk = row_strings(g_df.select(shared), shared)
    ek = row_strings(e_df.select(shared), shared)
    gc, ec = collections.Counter(gk), collections.Counter(ek)
    only_g = gc - ec
    only_e = ec - gc
    return {"n_g": g_df.height, "n_e": e_df.height, "shared": shared, "only_g_cols": only_g_cols,
            "only_e_cols": only_e_cols, "n_only_g": sum(only_g.values()), "n_only_e": sum(only_e.values()),
            "ex_g": list(only_g.keys())[:3], "ex_e": list(only_e.keys())[:3]}


def round_view(g_re, e_re, bs_g, bs_e, rnd):
    def rv(re_df, bs_df):
        r = re_df.filter(pl.col("round") == rnd)
        win = r["winner_team_code"].to_list()
        bs = bs_df.filter(pl.col("round") == rnd)
        evs = sorted(bs.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded", "bomb_planted"]))
                     .select(["tick", "event_type", "player_team_code"]).iter_rows())
        return win, evs
    return rv(g_re, bs_g), rv(e_re, bs_e)


summary = collections.defaultdict(lambda: collections.Counter())
per_match_lines = []
missing_e = []
round_detail = []

for group, keys in (("AFFECTED", sel_aff), ("CLEAN", sel_clean)):
    log(f"\n== 2. {group} matches: G: tome vs E: raw ({len(keys)} matches) ==")
    for key in keys:
        day = int(key.split("/")[3])
        line = {"key": key, "day": day, "group": group}
        e_data = {}
        skip = False
        for ch in CHANNELS:
            e_df, ep = e_read(key, ch)
            if e_df is None:
                missing_e.append((key, ch, ep))
                skip = True
                break
            e_data[ch] = norm_E(e_df, key)
        if skip:
            log(f"  {key} MISSING E: file(s); skipped")
            summary[group]["missing_e"] += 1
            continue
        g_data = {ch: G[ch].get(key, pl.DataFrame()) for ch in CHANNELS}
        diffs = {}
        for ch in CHANNELS:
            gd = g_data[ch]
            ed = e_data[ch]
            if gd.height == 0:
                # tome has no rows for this channel for this match
                diffs[ch] = {"n_g": 0, "n_e": ed.height, "n_only_g": 0, "n_only_e": ed.height,
                             "only_g_cols": [], "only_e_cols": [], "shared": [], "ex_g": [], "ex_e": []}
                continue
            diffs[ch] = compare_channel(gd, ed)
        n_rows_g = {ch: diffs[ch]["n_g"] for ch in CHANNELS}
        n_rows_e = {ch: diffs[ch]["n_e"] for ch in CHANNELS}
        n_only = {ch: diffs[ch]["n_only_g"] + diffs[ch]["n_only_e"] for ch in CHANNELS}
        line.update({"rows_g": n_rows_g, "rows_e": n_rows_e, "row_diff": n_only})
        summary[group]["matches"] += 1
        summary[group]["any_row_diff"] += int(sum(n_only.values()) > 0)
        for ch in CHANNELS:
            summary[group][f"rows_diff_{ch}"] += n_only[ch]
            summary[group][f"rows_g_{ch}"] += n_rows_g[ch]
            summary[group][f"rows_e_{ch}"] += n_rows_e[ch]
            if diffs[ch].get("only_g_cols") or diffs[ch].get("only_e_cols"):
                summary[group][f"colset_diff_{ch}"] += 1
        # round-13 neighbourhood: any row diff in rounds 12-14 for round_end/round_state/bomb channels
        nb_diff = 0
        for ch in ["round_end", "round_state", "bomb_state", "bomb_action"]:
            gd = g_data[ch]
            ed = e_data[ch]
            if gd.height == 0 and ed.height == 0:
                continue
            if "round" in gd.columns:
                nb_g = gd.filter(pl.col("round").is_between(12, 14)).height
                nb_e = ed.filter(pl.col("round").is_between(12, 14)).height
                nb_diff += abs(nb_g - nb_e)
        line["nb_rows_diff_12_14_count_only"] = nb_diff
        (g_win, g_evs), (e_win, e_evs) = round_view(g_data["round_end"], e_data["round_end"],
                                                    g_data["bomb_state"], e_data["bomb_state"], FOCUS)
        g_flag = None
        e_flag = None
        if g_win and any(ev[1] in ("bomb_defused", "bomb_exploded") for ev in g_evs):
            side = CT_CODE if any(ev[1] == "bomb_defused" for ev in g_evs) else T_CODE
            g_flag = int(side != g_win[0])
        if e_win and any(ev[1] in ("bomb_defused", "bomb_exploded") for ev in e_evs):
            side = CT_CODE if any(ev[1] == "bomb_defused" for ev in e_evs) else T_CODE
            e_flag = int(side != e_win[0])
        line.update({"r13_win_g": g_win, "r13_win_e": e_win, "r13_evs_g": g_evs, "r13_evs_e": e_evs,
                     "r13_mm_g": g_flag, "r13_mm_e": e_flag})
        per_match_lines.append(line)
        tag = "DIFF" if sum(n_only.values()) else "same"
        log(f"  {key} day={day:02d} {tag} rows_g={n_rows_g} rows_e={n_rows_e} row_diff={n_only}")
        log(f"     r13 winner G={g_win} E={e_win} | bomb G={g_evs} | bomb E={e_evs} | mm G={g_flag} E={e_flag}")
        for ch in CHANNELS:
            d = diffs[ch]
            if d.get("only_g_cols") or d.get("only_e_cols"):
                log(f"     colset mismatch {ch}: only G {d['only_g_cols']} only E {d['only_e_cols']}")
            if d["n_only_g"] or d["n_only_e"]:
                log(f"     {ch}: example only-in-G {d['ex_g']}")
                log(f"     {ch}: example only-in-E {d['ex_e']}")

# ---------- 3. summary
log(f"\n== 3. Summary (mode={MODE}) ==")
for group in ("AFFECTED", "CLEAN"):
    s = summary[group]
    log(f"{group}: matches compared {s['matches']}; matches with any row difference {s['any_row_diff']}; "
        f"missing E: {s['missing_e']}")
    for ch in CHANNELS:
        log(f"   {ch:<13} rows G={s[f'rows_g_{ch}']:>7} E={s[f'rows_e_{ch}']:>7} "
            f"row-multiset difference={s[f'rows_diff_{ch}']:>5} colset-mismatch matches={s[f'colset_diff_{ch}']}")

log("\nRound-13 side check, per match (G: vs E:):")
g_mm = sum(1 for x in per_match_lines if x["r13_mm_g"] == 1)
e_mm = sum(1 for x in per_match_lines if x["r13_mm_e"] == 1)
g_out = sum(1 for x in per_match_lines if x["r13_mm_g"] is not None)
e_out = sum(1 for x in per_match_lines if x["r13_mm_e"] is not None)
same_win = sum(1 for x in per_match_lines if x["r13_win_g"] == x["r13_win_e"])
same_evs = sum(1 for x in per_match_lines if x["r13_evs_g"] == x["r13_evs_e"])
log(f"  round-13 with bomb outcome: G={g_out} E={e_out}; mismatches: G={g_mm} E={e_mm}")
log(f"  round-13 winner_team_code identical G vs E in {same_win} of {len(per_match_lines)} matches")
log(f"  round-13 bomb events identical G vs E in {same_evs} of {len(per_match_lines)} matches")
for grp in ("AFFECTED", "CLEAN"):
    xs = [x for x in per_match_lines if x["group"] == grp]
    log(f"  {grp}: G mm={sum(1 for x in xs if x['r13_mm_g'] == 1)}/{len(xs)}, E mm={sum(1 for x in xs if x['r13_mm_e'] == 1)}/{len(xs)}")

log(f"\nmissing E: files: {len(missing_e)}")
for m in missing_e[:10]:
    log("  ", m)
log(f"elapsed {time.time() - t0:.1f}s")
