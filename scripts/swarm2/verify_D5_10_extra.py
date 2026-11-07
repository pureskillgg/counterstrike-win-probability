"""verify D5_10, step 3: which tome the bake-off kept set comes from, bomb mismatch rate by round, and what-ifs.

Read-only on G:. Reuses the scratchpad build from verify_D5_10_build.py and the functions in
verify_D5_10_analysis.py (imported; that module's main() is not run).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_extra.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_D5_10_analysis as A  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}
# bakeoff.py DENSE / SPARSE (re-typed; bakeoff.py itself is not imported)
DENSE = ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state", "team_change"]
SPARSE = ["player_disconnect", "bot_takeover"]


def chan_coverage(year):
    """Channel filter of bakeoff_kept (extract_state_1hz.bakeoff_kept): a match is channel-kept if it has a keyset
    in every channel, and dense channels have rows for it. Returns (kept_count, per-channel missing)."""
    base, suf = TOMES[year]
    header = set()
    hf = sorted(glob.glob(str(base / "header" / "dataframe_*")))
    header = set(pl.concat([pl.scan_parquet(f).select(["key"]) for f in hf]).collect()["key"].to_list())
    missing = collections.Counter()
    kept = set(header)
    for ch in DENSE + SPARSE:
        folder = base / f"basic_{ch}{suf}"
        ksf = sorted(glob.glob(str(folder / "keyset_*")))
        if not ksf:
            print(f"  [{year}] channel {ch}: no folder or no keyset files -> every match missing this channel")
            missing[ch] = len(header)
            kept = set()
            continue
        ks = set(pl.concat([pl.read_parquet(f) for f in ksf]).to_series(0).to_list())
        kept &= ks
        missing[ch] += len(header - ks)
        if ch in DENSE:
            dfs = sorted(glob.glob(str(folder / "dataframe_*")))
            have = set(pl.concat([pl.scan_parquet(f).select(["match_key"]) for f in dfs]).unique().collect()
                       ["match_key"].to_list())
            kept &= have
            missing[ch + "_rows"] += len(header - have)
    return len(kept), missing


def main():
    print("== 1. Which tome feeds the bake-off kept set? (bakeoff -> feature_validation.read -> TOME) ==")
    print("  feature_validation.TOME = tomes/2026-10-01_to_08 (default for fv.read); bakeoff header read asks for max_rounds,")
    print("  which the 2024 header does not have (checked in verify_D5_10_probe / build).")
    for y in ["2024", "2026"]:
        n, miss = chan_coverage(y)
        print(f"  [{y}] matches passing the bakeoff channel filter (keyset in all channels, dense rows present): {n}")
        print(f"       missing counts: {dict(miss)}")
    print("  claim (bake-off): channel_kept 1,709 of the 2026 tome's 1,752 matches (score_lead_fix.json)")

    print("\n== 2. Bomb mismatch rate by round label (2024; one planted bomb, one resolution; round_end winner vs bomb) ==")
    R24, M24 = A.load("2024")
    tab = collections.defaultdict(lambda: [0, 0])
    for m, rr in R24.items():
        reg = A.regime_of(M24[m])
        for r, x in rr.items():
            if x["n_planted"] >= 1 and (x["n_exploded"] + x["n_defused"]) == 1 and x["winner"] is not None:
                truth = A.T if x["n_exploded"] else A.CT
                tab[(reg, r)][0] += 1
                tab[(reg, r)][1] += int(truth != x["winner"])
    for reg in ["A", "B"]:
        rows = [(r, v[0], v[1]) for (g, r), v in sorted(tab.items()) if g == reg and r <= 30]
        print(f"  regime {reg} (round, bomb rounds, mismatches): " + ", ".join(f"{r}:{n}/{k}" for r, n, k in rows))

    print("\n== 3. What-ifs on the regime-A team-A diff (no 8->9 swap) ==")
    regA_ns = lambda m: A.regime_of(M24[m]) == "A" and not A.swap8_flag(R24.get(m, {}))
    S = A.SCHEDS["D5_12_24_27"]
    A.diff_summary(R24, M24, S, regA_ns, "winner", "no flip")
    A.diff_summary(R24, M24, S, regA_ns, "winner", "flip 13", flip_rounds=frozenset({13}))
    A.diff_summary(R24, M24, S, regA_ns, "winner", "flip 13, 25, 28", flip_rounds=frozenset({13, 25, 28}))
    A.diff_summary(R24, M24, S, regA_ns, "winner", "flip 13, 25, 28 (BO map)", flip_rounds=frozenset({13, 25, 28}))

    print("\n== 4. 2026 control, MR16 matches with the BO MR16 swap rule (bake-off lead vs round_end winners) ==")
    R26, M26 = A.load("2026")
    mr16 = lambda m: M26[m]["max_rounds"] == 16
    A.diff_summary(R26, M26, A.SCHEDS["BO_MR16"], mr16, "winner", "2026 max_rounds 16 (BO_MR16 map)")
    A.diff_summary(R26, M26, A.SCHEDS["BO_MR24"], mr16, "winner", "2026 max_rounds 16 (BO_MR24 map, wrong)")
    A.header_test(R26, M26, lambda mat: A.sched(16), mr16, "2026 max_rounds 16, BO_MR16 map")

    print("\n== 5. Regime by day (2024) ==")
    days = collections.Counter((A.regime_of(M24[m]), M24[m]["day"]) for m in M24)
    print("  regime A by day:", sorted((d, n) for (g, d), n in days.items() if g == "A"))
    print("  regime B by day:", sorted((d, n) for (g, d), n in days.items() if g == "B"))
    print("  regime-marker A matches in 2026:", sum(1 for m in M26 if A.regime_of(M26[m]) == "A"), "of", len(M26))


if __name__ == "__main__":
    main()
