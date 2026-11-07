"""fix332: per-match raw pass (data/csds per-match player_info and player_status) for matches not in the tome chunks.

Same definitions as fix332_scan.py (alive = health > 0 at tick == export freeze; dup_roster; unro_alive).
Control: tome-covered laptop matches in the same S3 day range are re-run from per-match files and compared
round by round with the tome result (round_flags.parquet).
Writes only under fixes/fix332/.
"""
import csv
import json
import os
import time

import numpy as np
import polars as pl

T0 = time.time()
FIN = "/home/sagemaker-user/adam/builder/final/"
EXP = FIN + "export_clean_v4/"
W = "/home/sagemaker-user/adam/builder/fixes/fix332/"
RAW = W + "raw/"


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, flush=True)


keys = json.load(open(EXP + "match_keys.json"))
kidx = {k: i for i, k in enumerate(keys)}
rm = np.load(EXP + "round_match.npy").astype(np.int64)
rn = np.load(EXP + "round_num.npy").astype(np.int64)
rf = np.load(EXP + "round_freeze.npy").astype(np.int64)
ER = pl.DataFrame({"ridx": np.arange(len(rm), dtype=np.int64), "mid": rm, "round": rn, "freeze": rf})
cm = pl.read_parquet(FIN + "clean_matches.parquet", columns=["match_key", "src"]).unique(subset=["match_key"], keep="first")
KEYIDX = pl.DataFrame({"match_key": keys, "mid": np.arange(len(keys), dtype=np.int64)})

lines = [l.rstrip("\n").split("\t") for l in open(W + "dl_list.tsv") if l.strip()]
log("download list", len(lines))
info_parts, st_parts, missing_files, processed = [], [], [], []
ER_KEY = ER.select(["mid", "round", "freeze"]).unique(subset=["mid", "round"], keep="first")
for n_done, (folder, name, tag) in enumerate(lines):
    mk = folder + "/csds"
    d = RAW + name
    pi, ps = d + "/player_info", d + "/player_status"
    if not (os.path.exists(pi) and os.path.getsize(pi) > 0 and os.path.exists(ps) and os.path.getsize(ps) > 0):
        missing_files.append({"match_key": mk, "tag": tag, "name": name})
        continue
    mid = kidx[mk]
    processed.append(mid)
    # export rounds of this match only (round -> freeze), so the per-tick status rows are cut down before storing
    ERm = ER_KEY.filter(pl.col("mid") == mid).select(["round", "freeze"])
    f = pl.read_parquet(pi, columns=["round", "player_id", "team_code"]).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("team_code").cast(pl.Int64))
    f = f.join(ERm, on="round", how="inner").select(["round", "player_id", "team_code"]).with_columns(pl.lit(mid).cast(pl.Int64).alias("mid"))
    info_parts.append(f)
    g = pl.read_parquet(ps, columns=["tick", "round", "player_id", "health"]).with_columns(
        pl.col("tick").cast(pl.Int64), pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("health").cast(pl.Int64))
    g = g.join(ERm, on="round", how="inner").filter(pl.col("tick") == pl.col("freeze"))
    g = g.select(["round", "player_id", "health"]).with_columns(pl.lit(mid).cast(pl.Int64).alias("mid"))
    st_parts.append(g)
    if n_done % 500 == 0:
        log("progress", n_done, "of", len(lines))
log("matches read", len(info_parts), "missing or empty files", len(missing_files))
mf = pl.DataFrame(missing_files) if missing_files else None
if mf is not None:
    mf.write_csv(W + "per_match_missing_files.csv")

info = pl.concat(info_parts).unique(subset=["mid", "round", "player_id", "team_code"], keep="first")
st = pl.concat(st_parts)
ER_SUB = ER_KEY.join(pl.DataFrame({"mid": processed}, schema={"mid": pl.Int64}).unique(), on="mid", how="inner")
log("roster pairs (export rounds)", info.height, "freeze status rows", st.height, "export rounds in processed matches", ER_SUB.height)

codes = info.group_by(["mid", "round", "player_id"]).agg(
    pl.col("team_code").n_unique().alias("n_codes"), pl.col("team_code").min().alias("code_min"))
statpid = st.unique(subset=["mid", "round", "player_id"], keep="first").group_by(["mid", "round"]).agg(
    pl.len().alias("n_status_pids"))
alive = st.filter(pl.col("health") > 0).select(["mid", "round", "player_id"]).unique()
A = alive.join(codes, on=["mid", "round", "player_id"], how="left").with_columns(
    pl.col("n_codes").fill_null(0), pl.col("code_min").fill_null(0))
A = A.with_columns(
    (pl.col("n_codes") == 0).alias("unro"), (pl.col("n_codes") > 1).alias("dup"),
    ((pl.col("n_codes") == 1) & (pl.col("code_min") == 3)).alias("sct"),
    ((pl.col("n_codes") == 1) & (pl.col("code_min") == 2)).alias("st"))
AR = A.group_by(["mid", "round"]).agg(pl.len().alias("alive_n"), pl.col("unro").sum().alias("unro_n"),
                                      pl.col("dup").sum().alias("dup_alive_n"), pl.col("sct").sum().alias("alive_ct"),
                                      pl.col("st").sum().alias("alive_t"))
DR = codes.filter(pl.col("n_codes") > 1)
DRN = DR.group_by(["mid", "round"]).agg(pl.len().alias("dup_roster_n"))
RN = info.group_by(["mid", "round"]).agg(pl.col("player_id").n_unique().alias("n_info_pids"))
R = ER_SUB.join(RN, on=["mid", "round"], how="left").join(statpid, on=["mid", "round"], how="left").join(
    AR, on=["mid", "round"], how="left").join(DRN, on=["mid", "round"], how="left")
fills = ["n_info_pids", "n_status_pids", "alive_n", "unro_n", "dup_alive_n", "alive_ct", "alive_t", "dup_roster_n"]
R = R.with_columns([pl.col(c).fill_null(0) for c in fills]).join(KEYIDX, on="mid", how="left").join(cm, on="match_key", how="left")
R = R.with_columns(
    pl.when((pl.col("dup_roster_n") > 0) & (pl.col("unro_n") > 0)).then(pl.lit("dup+unro"))
    .when(pl.col("dup_roster_n") > 0).then(pl.lit("dup"))
    .when(pl.col("unro_n") > 0).then(pl.lit("unro"))
    .otherwise(pl.lit("")).alias("kind"),
    (pl.col("n_status_pids") > 0).alias("has_freeze_status"),
    (pl.col("n_info_pids") > 0).alias("has_roster"))
R = R.with_columns(pl.lit("per_match").alias("method"))
R.write_parquet(W + "per_match_round_flags.parquet")
log("per-match rounds", R.height, "with freeze status", R.filter(pl.col("has_freeze_status")).height,
    "with roster", R.filter(pl.col("has_roster")).height)

# control comparison against the tome pass
tome = pl.read_parquet(W + "round_flags.parquet", columns=["match_key", "round", "freeze", "n_info_pids", "n_status_pids",
                                                           "alive_n", "unro_n", "dup_roster_n", "dup_alive_n", "alive_ct", "alive_t", "kind"])
ctrl_keys = [l[0] + "/csds" for l in lines if l[2] == "control"]
C = R.filter(pl.col("match_key").is_in(ctrl_keys)).join(
    tome.rename({c: c + "_tome" for c in tome.columns if c not in ("match_key", "round")}), on=["match_key", "round"], how="left")
cmp_cols = ["freeze", "n_info_pids", "n_status_pids", "alive_n", "unro_n", "dup_roster_n", "dup_alive_n", "alive_ct", "alive_t", "kind"]
res = {"control_matches": len(ctrl_keys), "control_rounds": C.height, "control_rounds_tome_missing": C.filter(pl.col("kind_tome").is_null()).height}
for c in cmp_cols:
    a = C[c]
    b = C[c + "_tome"]
    res["mismatch_" + c] = int((a.cast(pl.String) != b.cast(pl.String)).sum())
res["control_kind_counts_per_match"] = C.group_by("kind").len().rows()
json.dump(res, open(W + "per_match_control_compare.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str), flush=True)

# candidates from per-match pass (recovered matches only, not controls)
rec_keys = [l[0] + "/csds" for l in lines if l[2] == "recover"]
Cand = R.filter((pl.col("kind") != "") & pl.col("match_key").is_in(rec_keys)).sort(["match_key", "round"])
log("per-match candidate rounds (recovered matches)", Cand.height, "| by kind", Cand.group_by("kind").len().rows())
dup_ids, unro_ids = {}, {}
for mid, rd, pid in DR.select(["mid", "round", "player_id"]).rows():
    dup_ids.setdefault((mid, rd), []).append(pid)
for mid, rd, pid in A.filter(pl.col("unro")).select(["mid", "round", "player_id"]).rows():
    unro_ids.setdefault((mid, rd), []).append(pid)
rows = []
for r in Cand.select(["mid", "round", "freeze", "match_key", "src", "kind", "alive_ct", "alive_t", "alive_n",
                      "n_info_pids", "n_status_pids", "dup_alive_n", "unro_n", "dup_roster_n"]).iter_rows(named=True):
    k = (r["mid"], r["round"])
    rows.append({"match_key": r["match_key"], "round": r["round"], "freeze_tick": r["freeze"], "kind": r["kind"],
                 "dup_ids": ";".join(str(x) for x in sorted(dup_ids.get(k, []))),
                 "unro_ids": ";".join(str(x) for x in sorted(unro_ids.get(k, []))),
                 "alive_ct": r["alive_ct"], "alive_t": r["alive_t"], "src": r["src"], "alive_n": r["alive_n"]})
with open(W + "per_match_candidates.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["match_key"])
    w.writeheader()
    w.writerows(rows)
log("DONE")
