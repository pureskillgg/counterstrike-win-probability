"""verify_D2_9 raw-source spot check (read-only on E:, nothing written).

Picks 2 class-B matches (basic_round_end tick == round_state cs_pre_restart tick, not == round_end) and
2 class-A matches (== round_state round_end) from the 2024 tome, then looks up the same match in the
2024 tail-recovered raw copy on E: and compares round_end / round_start / round_state tick values for
rounds 1-3 (tome vs raw). Purpose: decide whether the class-B late round_end tick is already in the raw
data (so the proposed window inherits it) or was introduced when the tome was built.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_chk_rawcheck.py
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(14)
pl.Config.set_fmt_str_lengths(50)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
RAW24 = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")


def parts_tome(ch):
    fd = sorted(glob.glob(str(TOME24 / f"basic_{ch}.*")))
    assert len(fd) == 1, (ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


def first_key(df, name):
    return (df.sort("tick").unique(["match_key", "round"], keep="first")
            .select(["match_key", "round", pl.col("tick").alias(name)]))


rst = pl.scan_parquet(parts_tome("round_state")).select(["match_key", "round", "tick", "event_type"]).collect()
E = first_key(pl.scan_parquet(parts_tome("round_end")).select(["match_key", "round", "tick"]).collect(), "E")
CPR = first_key(rst.filter(pl.col("event_type") == "cs_pre_restart").select(["match_key", "round", "tick"]), "CPR")
RE = first_key(rst.filter(pl.col("event_type") == "round_end").select(["match_key", "round", "tick"]), "RE")

jb = E.join(CPR, on=["match_key", "round"]).join(RE, on=["match_key", "round"], how="left")
classB = jb.filter((pl.col("E") == pl.col("CPR")) & pl.col("RE").is_null())
classA = jb.filter(pl.col("E") == pl.col("RE"))
print(f"2024 class B (E==cs_pre_restart, no state round_end on key): {classB.height:,} keys")
print(f"2024 class A (E==state round_end): {classA.height:,} keys")

# choose 2 matches of each kind, lowest match_key for determinism, prefer matches with >= 3 rounds
def pick(df, n):
    mks = (df.group_by("match_key").len().filter(pl.col("len") >= 3).sort("match_key")["match_key"].to_list())
    return mks[:n]


picks = [("B", m) for m in pick(classB, 2)] + [("A", m) for m in pick(classA, 2)]
print("picked:", [(c, m) for c, m in picks])


def raw_dir(match_key):
    # match_key looks like csds/2024/02/DD/<uuid>
    parts = match_key.split("/")
    return RAW24 / parts[3] / parts[4]


def find_match_dir(match_key):
    parts_ = match_key.split("/")
    d = RAW24 / parts_[3]
    cand = d / parts_[4]
    return cand if cand.exists() else None


for kind, mk in picks:
    print(f"\n==== class {kind}: {mk}")
    md = find_match_dir(mk)
    if md is None:
        print("  raw match folder not found under", RAW24)
        continue
    print("  raw folder:", md)
    chans = sorted(os.listdir(md))
    print("  channels present (first 60):", [c for c in chans][:60])
    out = {}
    for ch in ["round_end", "round_start", "round_state"]:
        f = md / ch  # raw channels are single parquet files named by channel
        if not f.is_file():
            print(f"  raw {ch}: not found")
            continue
        raw_schema = pl.read_parquet_schema(str(f))
        print(f"  raw {ch} schema: {dict(raw_schema)}")
        cols = ["round", "tick"] + (["event_type"] if ch == "round_state" else [])
        df = pl.read_parquet(str(f), columns=[c for c in cols if c in raw_schema])
        out[ch] = df
        print(f"  raw {ch}: rows={df.height:,}")
    if "round_end" in out:
        raw_end = first_key(out["round_end"].with_columns(pl.lit(mk).alias("match_key")), "raw_E")
    if "round_state" in out:
        st = out["round_state"].with_columns(pl.lit(mk).alias("match_key"))
        raw_cpr = first_key(st.filter(pl.col("event_type") == "cs_pre_restart").select(["match_key", "round", "tick"]),
                            "raw_CPR")
        raw_re = first_key(st.filter(pl.col("event_type") == "round_end").select(["match_key", "round", "tick"]),
                           "raw_RE")
        raw_fe = first_key(st.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"]),
                           "raw_fe")
        comp = raw_end.join(raw_cpr, on=["match_key", "round"], how="left").join(
            raw_re, on=["match_key", "round"], how="left").join(raw_fe, on=["match_key", "round"], how="left")
        S_tome = first_key(pl.scan_parquet(parts_tome("round_start")).select(
            ["match_key", "round", "tick"]).collect().filter(pl.col("match_key") == mk), "tome_S")
        tome_sub = E.filter(pl.col("match_key") == mk).join(S_tome, on=["match_key", "round"], how="left")
        print("  per round (rounds 1..5): raw round_end, raw cs_pre_restart, raw round_state round_end, raw freeze_end, tome E, tome S")
        j = comp.join(tome_sub, on=["match_key", "round"], how="left").filter(pl.col("round") <= 5).sort("round")
        print(j.select(["round", "raw_E", "raw_CPR", "raw_RE", "raw_fe", "E", "tome_S"]))
        # tome vs raw: basic round_end equality with raw round_end channel
        rawE_all = out["round_end"]
        print(f"  raw basic round_end rows={rawE_all.height}; first tick rounds 1..3 (raw) vs tome:")
        print(rawE_all.sort("round").head(3))
