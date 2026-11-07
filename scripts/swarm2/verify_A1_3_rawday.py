# verify_A1_3_rawday.py
# Verifier for A1-3, per-day breakdown of the RAW per-match schema signature (read-only footers on E:),
# and whether the raw schema class goes together with the day-08 row-count effect.
# For each keyset match of item_equip / item_pickup / player_hurt (2026 tome):
#   raw signature label (column count, def_index presence, id column types), raw num_rows,
#   rounds in that match (from the tome basic_round_end channel, distinct round values).
# Output: per day x label counts, mean raw rows per match, mean raw rows per round.
# ASSUMPTION: raw match file = E:/csds_final/csds/2026/10/<DD>/<uuid>/<channel> (plain parquet; footer read only, plus def_index column when present).
import os, glob, re, collections
import polars as pl
import pyarrow.parquet as pq

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
RAW = "E:/csds_final/csds/2026/10"
CHANS = ["item_equip", "item_pickup", "player_hurt"]
ID_COLS = {"item_equip": ["player_id_fixed"], "item_pickup": ["player_id_fixed"],
           "player_hurt": ["player_id_fixed", "attacker_id_fixed"]}
UUID_PAT = r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
DAY_RE = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")


def P(*a):
    print(*a, flush=True)


def folder_of(ch):
    return os.path.join(TOME, f"basic_{ch}.2026-09-18,2026-10-08")


def keys_of(ch):
    out = []
    for k in sorted(glob.glob(os.path.join(folder_of(ch), "keyset_*"))):
        out += pq.read_table(k).column(0).to_pylist()
    return out


def rounds_map():
    pages = sorted(glob.glob(os.path.join(folder_of("round_end"), "dataframe_*")))
    df = pl.concat([pl.read_parquet(p, columns=["match_key", "round"]) for p in pages])
    df = df.with_columns(pl.col("match_key").str.extract(UUID_PAT, 1).alias("uuid"))
    per = df.group_by("uuid").agg(pl.col("round").n_unique().alias("rounds"))
    return dict(zip(per["uuid"].to_list(), per["rounds"].to_list()))


def label_of(sch, ch):
    types = dict(zip(sch.names, [str(t) for t in sch.types]))
    bits = [f"ncols={len(sch.names)}", "def_index" if "def_index" in types else "NO_def_index"]
    bits += [f"{c}={types.get(c)}" for c in ID_COLS[ch]]
    return " ".join(bits)


if __name__ == "__main__":
    P("verify_A1_3_rawday: raw schema class per day vs rows per match and per round")
    rmap = rounds_map()
    for ch in CHANS:
        keys = keys_of(ch)
        recs = []
        missing = 0
        for k in keys:
            parts = k.split("/")
            day, uuid = parts[3], parts[4]
            path = f"{RAW}/{day}/{uuid}/{ch}"
            if not os.path.exists(path):
                missing += 1
                continue
            sch = pq.read_schema(path)
            nr = pq.read_metadata(path).num_rows
            recs.append(dict(day=day, uuid=uuid, label=label_of(sch, ch), nrows=nr,
                             rounds=rmap.get(uuid, 0)))
        df = pl.DataFrame(recs)
        P(f"\n== {ch}: raw matches read={df.height} missing={missing} rounds_known={int((df['rounds'] > 0).sum())}")
        labels = sorted(df["label"].unique().to_list())
        for lab in labels:
            P(f"   label [{lab}]")
        agg = (df.group_by("day", "label")
                 .agg(pl.len().alias("matches"), pl.col("nrows").mean().alias("rows_per_match"),
                      (pl.col("nrows") / pl.col("rounds").clip(lower_bound=1)).mean().alias("rows_per_round"),
                      pl.col("rounds").mean().alias("rounds_per_match"))
                 .sort("day", "label"))
        for r in agg.iter_rows(named=True):
            P(f"     day {r['day']} | {r['label']:<55} | matches={r['matches']:>5} rows/match={r['rows_per_match']:9.1f} "
              f"rows/round={r['rows_per_round']:8.1f} rounds/match={r['rounds_per_match']:5.2f}")
        # which days carry which labels (counts only)
        P("   label counts by day:")
        cnt = df.group_by("day", "label").agg(pl.len().alias("n")).sort("day", "label")
        by_day = collections.defaultdict(dict)
        for r in cnt.iter_rows(named=True):
            by_day[r["day"]][r["label"]] = r["n"]
        for d in sorted(by_day):
            P(f"     day {d}: {by_day[d]}")
    P("\nDONE rawday")
