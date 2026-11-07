"""Y2: audit of the data owner's "Known Flaws" list against our store (CPU, G: only).

  A  hand uploads / FACEIT: platform x server_name class, how often match_date is the processing day, and what the chronological split
     is made of
  B  duplicates: matches that share the header fields the docs suggest (map, server, build, final scores) versus the duplicate groups our
     event-signature rule found (prepare.py); any header-duplicate pair our rule did not link is listed
  C  model quality by source (FACEIT server vs Valve) and by the pipeline versions that built the match (rushb_version, ppp_version)
  D  rows used for the model whose status snapshot disagrees with the event-derived alive counts (missing events / roster defects)
Writes data/viz/Y2_known_flaws_audit.json
"""
import glob
import json
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, os.path.join(r"G:\pureskill\csds-cs2", "scripts", "model"))
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
D = os.path.join(ROOT, "data", "combined")
out = {}
H = pl.scan_parquet(os.path.join(ROOT, "tomes", "combined", "tome", "csds", "header", "dataframe_*")).collect()
M = pl.read_parquet(os.path.join(D, "matches.parquet"))
print("matches columns:", [c for c in M.columns][:40])
H = H.rename({"key": "match_key"}).with_columns(
    pl.when(pl.col("server_name").str.starts_with("FACEIT")).then(pl.lit("faceit_server")).when(pl.col("server_name").str.starts_with("Valve")).then(pl.lit("valve_server")).otherwise(pl.lit("other")).alias("source"),
    pl.col("match_key").str.slice(5, 4).alias("era"))

# ------------------------------------------------------------------ A
a = H.group_by(["era", "platform", "source"]).agg(pl.len().alias("matches")).sort(["era", "platform", "source"])
print("\n== A. platform x server class (all matches in the combined store)")
for r in a.iter_rows(named=True):
    print(f"  {r['era']} {str(r['platform']):<8} {r['source']:<14} {r['matches']:>6}")
out["A_platform_source"] = a.to_dicts()
day = H.with_columns(pl.col("match_key").str.extract(r"csds/(\d{4}/\d{2}/\d{2})/", 1).alias("upload_day"), pl.col("match_date").cast(pl.String).str.slice(0, 10).str.replace_all("-", "/").alias("md"))
same = day.group_by("platform").agg((pl.col("md") == pl.col("upload_day")).mean().alias("match_date_is_upload_day"), pl.len().alias("matches")).sort("platform")
print("\n== match_date equals the upload (processing) day, by platform")
for r in same.iter_rows(named=True):
    print(f"  {str(r['platform']):<8} {100 * r['match_date_is_upload_day']:5.1f}%  of {r['matches']}")
out["A_match_date_is_upload_day"] = same.to_dicts()
G = pl.read_parquet(os.path.join(D, "grid.parquet"), columns=["match_key", "era"]).unique("match_key")
Gm = G.join(H.select("match_key", "platform", "source", "match_date"), on="match_key", how="left")
import wp_common as W
Gc = W.load_grid("combined", cols=["match_key", "era"]).unique("match_key")
sp = W.make_split("chrono", Gc, seed=0)
comp = []
for name, keys in sp.items():
    s = Gm.filter(pl.col("match_key").is_in(sorted(keys)))
    comp.append({"split": name, "matches": s.height, "platform_unknown": float((s["platform"] == "unknown").mean()), "faceit_server": float((s["source"] == "faceit_server").mean()),
                 "era_2026": float((s["era"] == "2026").mean())})
    print(f"  chrono {name:<5} matches {s.height:>5}  platform unknown {100 * comp[-1]['platform_unknown']:.0f}%  FACEIT server {100 * comp[-1]['faceit_server']:.0f}%  2026 {100 * comp[-1]['era_2026']:.0f}%")
out["A_chrono_split_composition"] = comp

# ------------------------------------------------------------------ B
print("\n== B. duplicates: header fields vs our event-signature groups")
cols = [c for c in ("map_name", "server_name", "build_num", "t_starters_score_final", "ct_starters_score_final", "number_of_points") if c in H.columns]
hk = H.select("match_key", *cols).with_columns(pl.concat_str([pl.col(c).cast(pl.String).fill_null("") for c in cols[:5]], separator="|").alias("hkey"))
ours = M.select("match_key", "sig_hash", "dup_group_size", "dup_keeper") if "sig_hash" in M.columns else None
J = hk.join(ours, on="match_key", how="left") if ours is not None else hk
grp = J.group_by("hkey").agg(pl.len().alias("n"), pl.col("sig_hash").n_unique().alias("sig_groups"), pl.col("match_key").first().alias("example"), pl.col("number_of_points").n_unique().alias("npoints_variants"))
multi = grp.filter(pl.col("n") >= 2)
print(f"  header-key groups with 2+ matches: {multi.height}  (matches in them: {int(multi['n'].sum())})")
unlinked = multi.filter(pl.col("sig_groups") > 1)
print(f"  of which our event signatures put the members in different groups (not linked as duplicates): {unlinked.height}")
unl_same_pts = unlinked.filter(pl.col("npoints_variants") == 1)
print(f"  ... and with identical number_of_points as well (strong candidates for missed duplicates): {unl_same_pts.height}")
out["B_header_dup_groups"] = int(multi.height)
out["B_not_linked_by_us"] = int(unlinked.height)
out["B_not_linked_same_number_of_points"] = int(unl_same_pts.height)
if unl_same_pts.height:
    print("  examples:", unl_same_pts.head(3)["example"].to_list())
if "dup_group_size" in M.columns:
    out["B_our_dup_groups"] = int(M.filter(pl.col("dup_group_size") > 1).select("sig_hash").n_unique())
    print(f"  our own duplicate groups (event signature): {out['B_our_dup_groups']}")

# ------------------------------------------------------------------ C
print("\n== C. model quality by source and by pipeline version (snapshot model, held-out sample)")
S = pl.read_parquet(os.path.join(D, "export_winprob_v1_test_sample.parquet")).select("match_key", "p_pipeline", "label_ct_win")
S = S.join(H.select("match_key", "source", "platform", "rushb_version", "ppp_version"), on="match_key", how="left")
p = np.clip(S["p_pipeline"].to_numpy().astype(float), 1e-6, 1 - 1e-6)
y = S["label_ct_win"].to_numpy().astype(float)
S = S.with_columns(pl.Series("ll", -(y * np.log(p) + (1 - y) * np.log(1 - p))))
rng = np.random.default_rng(0)


def table(col, label):
    rows = []
    for k, d in S.group_by(col):
        m = d["match_key"].n_unique()
        if d.height < 600 or m < 12:
            continue
        g = d.group_by("match_key").agg(pl.col("ll").sum().alias("s"), pl.len().alias("n"))
        s, n = g["s"].to_numpy(), g["n"].to_numpy().astype(float)
        bs = [s[i].sum() / n[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(300))]
        rows.append({col: str(k[0] if isinstance(k, tuple) else k), "rows": d.height, "matches": m, "logloss": float(s.sum() / n.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]})
    rows.sort(key=lambda r: r[col])
    print(f"  by {label}:")
    for r in rows:
        print(f"    {r[col]:<22} matches {r['matches']:>4}  rows {r['rows']:>6}  LL {r['logloss']:.4f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
    return rows


out["C_by_source"] = table("source", "server (FACEIT vs Valve)")
out["C_by_platform"] = table("platform", "platform")
out["C_by_rushb_version"] = table("rushb_version", "rushb_version")
out["C_by_ppp_version"] = table("ppp_version", "ppp_version")

# ------------------------------------------------------------------ D
print("\n== D. snapshot vs event-derived alive counts, per round (rows used by the model)")
try:
    import features_status as FS
    Gr = pl.read_parquet(os.path.join(D, "grid.parquet"), columns=["match_key", "round", "tick", "era", "alive_ct", "alive_t"]).filter(pl.col("era") == "2026")
    R = pl.read_parquet(os.path.join(D, "rounds.parquet"), columns=["match_key", "round", "missing_side"])
    Gr = Gr.join(R, on=["match_key", "round"], how="left").filter(pl.col("missing_side").is_null())
    A, have = FS.loadout_aggregates(None, Gr["match_key"].unique().to_list())
    n = A.group_by(["match_key", "round", "tick"]).agg((pl.col("side") == 3).sum().alias("n_ct"), (pl.col("side") == 2).sum().alias("n_t"))
    # side_aggregates has no count column; count players per side from the raw table instead
    raw = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "status_1hz", "*.parquet"))):
        raw.append(pl.read_parquet(f, columns=["match_key", "round", "tick", "side"]).group_by(["match_key", "round", "tick"]).agg((pl.col("side") == 3).sum().alias("n_ct"), (pl.col("side") == 2).sum().alias("n_t")))
    N = pl.concat(raw)
    J2 = Gr.join(N, on=["match_key", "round", "tick"], how="left").with_columns(pl.col("n_ct").fill_null(0), pl.col("n_t").fill_null(0))
    J2 = J2.with_columns(((pl.col("n_ct") != pl.col("alive_ct")) | (pl.col("n_t") != pl.col("alive_t"))).alias("bad"))
    rows_bad = int(J2["bad"].sum())
    rounds_bad = J2.filter(pl.col("bad")).select("match_key", "round").unique().height
    rounds_all = J2.select("match_key", "round").unique().height
    print(f"  rows {J2.height:,}; rows where the table disagrees with the events: {rows_bad:,} ({rows_bad / J2.height:.3%}); rounds affected {rounds_bad} of {rounds_all} ({rounds_bad / rounds_all:.2%})")
    out["D_inconsistent_rows"] = rows_bad
    out["D_inconsistent_rounds"] = rounds_bad
    out["D_rounds"] = rounds_all
except Exception as exc:  # the audit must not stop on this part
    print("  (skipped:", type(exc).__name__, exc, ")")
json.dump(out, open(os.path.join(ROOT, "data", "viz", "Y2_known_flaws_audit.json"), "w", encoding="utf-8"), indent=1, default=str)
print("\n-> data/viz/Y2_known_flaws_audit.json")
