# Verifier for A1-3: which read strategies over the drifting channels fail or succeed under the
# project polars (2.0.0), and whether the proposed workarounds (ScanCastOptions / missing_columns /
# per-page cast) produce correct, lossless results. Collects only small projections (low memory).
import os, glob, traceback
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CH = {
    "basic_item_equip": "player_id_fixed",
    "basic_item_pickup": "player_id_fixed",
    "basic_player_hurt": "attacker_id_fixed",
}


def pages_of(ch, only=None):
    folder = os.path.join(T, ch + ".2026-09-18,2026-10-08")
    ps = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    if only is not None:
        ps = [ps[i] for i in only]
    return ps


def attempt(label, fn):
    try:
        out = fn()
        print(f"OK    | {label} | {out}")
    except Exception as e:
        first = str(e).splitlines()[0][:200] if str(e) else ""
        print(f"FAIL  | {label} | {type(e).__name__}: {first}")


for ch, idc in CH.items():
    print("=" * 110)
    print(ch, "id column of interest:", idc)
    allp = pages_of(ch)
    attempt(f"{ch} scan_parquet(all pages).collect() full",
            lambda: pl.scan_parquet(allp).collect().height)
    attempt(f"{ch} scan_parquet(all pages).select(pl.len())  [row count only]",
            lambda: pl.scan_parquet(allp).select(pl.len()).collect().item())
    attempt(f"{ch} scan_parquet(all pages).select(['round','tick']) [non-id projection]",
            lambda: pl.scan_parquet(allp).select(["round", "tick"]).collect().height)
    attempt(f"{ch} scan_parquet(all pages).select(['round', idc])",
            lambda: pl.scan_parquet(allp).select(["round", idc]).collect().height)
    attempt(f"{ch} read_parquet(all pages) full",
            lambda: pl.read_parquet(allp).height)

# Page-subset tests for the item_equip drift (the channel with 3 schemas)
print("=" * 110)
eq = pages_of("basic_item_equip")
attempt("item_equip pages 0-1 only (double, 24 cols, def_index pos differs)",
        lambda: pl.scan_parquet(eq[:2]).select(["round", "player_id_fixed"]).collect().height)
attempt("item_equip pages 2-7 only (int64, 23 cols, no def_index)",
        lambda: pl.scan_parquet(eq[2:]).select(["round", "player_id_fixed"]).collect().height)
attempt("item_equip pages 0 and 2 only (double vs int64, mixed)",
        lambda: pl.scan_parquet([eq[0], eq[2]]).select(["round", "player_id_fixed"]).collect().height)
attempt("item_equip pages 0 and 1 (both double; differ only in column ORDER)",
        lambda: pl.scan_parquet(eq[:2]).select(["round", "player_id_fixed", "def_index"]).collect().height)
attempt("item_equip pages 2-7 with def_index selected (expect missing column)",
        lambda: pl.scan_parquet(eq[2:]).select(["round", "def_index"]).collect().height)

# Proposed workaround 1: cast_options allow int->float, missing columns inserted as null
print("=" * 110)
print("WORKAROUND tests (per channel; collect only small projection)")
for ch, idc in CH.items():
    allp = pages_of(ch)
    cols = ["round", "tick", idc]
    if ch == "basic_item_equip":
        cols = ["round", "tick", "player_id_fixed", "def_index"]
    cast_fn = lambda: pl.scan_parquet(
        allp,
        cast_options=pl.ScanCastOptions(integer_cast="allow-float", missing_struct_fields="insert"),
        missing_columns="insert",
        extra_columns="ignore",
    ).select(cols).collect()
    attempt(f"{ch} WA1 cast_options(integer_cast=allow-float)+missing_columns=insert",
            lambda: (lambda d: f"rows={d.height} dtype[{idc}]={d.schema[idc]} nulls[{idc}]={d.get_column(idc).null_count()}")(cast_fn()))
    attempt(f"{ch} WA1b scan row count with same options",
            lambda: pl.scan_parquet(allp, cast_options=pl.ScanCastOptions(integer_cast="allow-float"),
                                    missing_columns="insert", extra_columns="ignore").select(pl.len()).collect().item())

# Proposed workaround 2 (explicit per-page cast to Int64): exactness vs. the source ints
print("=" * 110)
print("PER-PAGE explicit Int64 cast, exactness check (sum/min/max and nulls per page)")
for ch, idc in CH.items():
    allp = pages_of(ch)
    total_rows = 0
    tot_sum = 0
    frac_bad = 0
    per = []
    for p in allp:
        d = pl.read_parquet(p, columns=[idc])
        s = d.schema[idc]
        if s == pl.Float64:
            nonint = d.filter(pl.col(idc).is_not_null() & (pl.col(idc) != pl.col(idc).floor())).height
            frac_bad += nonint
        c = d.with_columns(pl.col(idc).cast(pl.Int64, strict=True).alias("x"))
        total_rows += c.height
        tot_sum += int(c.get_column("x").sum() or 0)
        per.append(f"{os.path.basename(p)}:{s}/null={d.get_column(idc).null_count()}")
    print(f"{ch}: per-page dtype/nulls {per}")
    print(f"{ch}: cast-to-Int64 rows={total_rows} sum={tot_sum} non-integral double values={frac_bad}")

print("=" * 110)
print("done")
