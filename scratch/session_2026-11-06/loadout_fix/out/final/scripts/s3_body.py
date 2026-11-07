
# ---- S3 channel stage (runs on the space, reads the channel with boto3, writes into the output arrays) ----
import io, sys, json, time
import numpy as np
import polars as pl
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, file=sys.stderr, flush=True)


BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
EXP = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
OUTD = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
TARGET = __TARGET__
P3 = __P3__
P4 = __P4__
# chunks c063 onward hold matches dated 2026-10-15 and later (their windows reach Oct 21+): not read
CHUNKS = ["c%03d" % i for i in range(0, 63)]

s3 = boto3.client("s3", config=Config(max_pool_connections=32, retries={"max_attempts": 8, "mode": "standard"}))
POOL = ThreadPoolExecutor(12)

keys = json.load(open(EXP + "match_keys.json"))
off = np.load(EXP + "offsets.npy").astype(np.int64)
rm = np.load(EXP + "round_match.npy").astype(np.int64)
rn = np.load(EXP + "round_num.npy").astype(np.int64)
rf = np.load(EXP + "round_freeze.npy").astype(np.int64)
nsteps = np.diff(off)
NS = int(off[-1])
key_m = {k: i for i, k in enumerate(keys)}
miss = [k for k in TARGET + P3 + P4 if k not in key_m]
log("export keys", len(keys), "target", len(TARGET), "p3", len(P3), "p4", len(P4), "keys not in export", len(miss))
assert not miss
set_target, set_p3, set_p4 = set(TARGET), set(P3), set(P4)
assert not (set_target & (set_p3 | set_p4)) and not (set_p3 & set_p4)
wanted = sorted(set_target | set_p3 | set_p4)
wanted_m = [key_m[k] for k in wanted]
label = {k: ("target" if k in set_target else ("parity_v3" if k in set_p3 else "parity_v4")) for k in wanted}
mk_df = pl.DataFrame({"match_key": wanted, "m": wanted_m, "set": [label[k] for k in wanted]},
                     schema={"match_key": pl.String, "m": pl.Int64, "set": pl.String})
rt = pl.DataFrame({"m": rm, "round": rn, "r": np.arange(len(rm), dtype=np.int64), "freeze": rf,
                   "off_r": off[:-1], "nsteps": nsteps}).filter(pl.col("m").is_in(wanted_m))
log("rounds for wanted matches", rt.height)

arr16 = np.load(OUTD + "loadout_step_i16.npy", mmap_mode="r+")
arr_eq = np.load(OUTD + "loadout_equip_i32.npy", mmap_mode="r+")
assert arr16.shape == (NS, 2, len(FIELDS) - 1) and arr_eq.shape == (NS, 2), (arr16.shape, arr_eq.shape)
log("output arrays opened", arr16.shape, arr_eq.shape)
wanted_list = wanted


def list_pages(c, kind):
    prefix = f"{PREFIX}/{c}/tome/csds/{kind}."
    out = []
    folders = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            d, n = o["Key"].rsplit("/", 1)
            # the table folder itself (not its nested header/ folder)
            if d.rsplit("/", 1)[-1].startswith(kind + ".") and n.startswith("dataframe_"):
                folders.add(d)
                out.append(o["Key"])
    assert len(folders) == 1, (c, kind, folders)
    return sorted(out)


def get_bytes(key):
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def read_pages(pkeys, columns):
    bs = list(POOL.map(get_bytes, pkeys))
    frames = []
    for b in bs:
        f = pl.read_parquet(io.BytesIO(b))
        missing = [c for c in columns if c not in f.columns]
        assert not missing, ("missing columns", missing)
        frames.append(f.select(columns))
    return frames


ST_COLS = ["match_key", "round", "tick", "player_id"] + INPUT_COLS
INFO_COLS = ["match_key", "round", "player_id", "team_code"]
cov_parts = []
found_m = set()
SET_CODE = np.zeros(len(keys), dtype=np.int8)  # 0 none, 1 target, 2 parity_v3, 3 parity_v4
for k in TARGET:
    SET_CODE[key_m[k]] = 1
for k in P3:
    SET_CODE[key_m[k]] = 2
for k in P4:
    SET_CODE[key_m[k]] = 3
chunk_stats = []
parity_rows = {"parity_v3": [], "parity_v4": []}

for c in CHUNKS:
    t1 = time.time()
    st_keys = list_pages(c, "basic_player_status")
    info_keys = list_pages(c, "basic_player_info")
    if not st_keys:
        log(c, "no status pages")
        continue
    info_frames = read_pages(info_keys, INFO_COLS)
    info = pl.concat(info_frames).filter(pl.col("match_key").is_in(wanted_list)).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("team_code").cast(pl.Int64))
    info_u = info.unique(subset=["match_key", "round", "player_id", "team_code"], keep="first")
    roster = info_u.group_by(["match_key", "round", "player_id"]).agg([
        pl.col("team_code").first().alias("team_code"), pl.len().alias("n_codes")])
    roster_conflicts = roster.filter(pl.col("n_codes") > 1).height
    roster = roster.select(["match_key", "round", "player_id", "team_code"])
    st_frames = read_pages(st_keys, ST_COLS)
    st_raw_rows = sum(f.height for f in st_frames)
    st = pl.concat(st_frames).filter(pl.col("match_key").is_in(wanted_list))
    del st_frames
    st = st.join(mk_df, on="match_key", how="inner")
    st = st.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    n_wanted_rows = st.height
    st = st.join(rt.select("m", "round", "r", "freeze", "off_r", "nsteps"), on=["m", "round"], how="left")
    n_no_round = st.filter(pl.col("r").is_null()).height
    st = st.filter(pl.col("r").is_not_null())
    st = st.join(roster, on=["match_key", "round", "player_id"], how="left")
    n_unrostered = st.filter(pl.col("team_code").is_null()).height
    st = st.filter(pl.col("team_code").is_not_null())
    side_idx_expr = (pl.when(pl.col("team_code") == 3).then(0).when(pl.col("team_code") == 2).then(1).otherwise(-1)).cast(pl.Int64)
    st = st.with_columns(side_idx_expr.alias("side_idx"))
    n_bad_side = st.filter(pl.col("side_idx") < 0).height
    st = st.filter(pl.col("side_idx") >= 0)
    st = st.with_columns((pl.col("tick") - pl.col("freeze")).alias("dd"))
    n_pre_freeze = st.filter(pl.col("dd") < 0).height
    n_off_grid = st.filter((pl.col("dd") >= 0) & (pl.col("dd") % TICK != 0)).height
    st = st.filter((pl.col("dd") >= 0) & (pl.col("dd") % TICK == 0))
    st = st.with_columns((pl.col("dd") // TICK).alias("k"))
    n_beyond = st.filter(pl.col("k") >= pl.col("nsteps")).height
    st = st.filter(pl.col("k") < pl.col("nsteps"))
    st = st.with_columns((pl.col("off_r") + pl.col("k")).alias("s"))
    n_pre_dedupe = st.height
    st = st.unique(subset=["r", "tick", "player_id"], keep="first", maintain_order=False)
    n_dups = n_pre_dedupe - st.height
    cov = st.group_by(["match_key", "set"]).agg([pl.len().alias("rows"), pl.col("r").n_unique().alias("rounds_with_rows"),
                                                  pl.col("s").n_unique().alias("steps_with_rows"),
                                                  (pl.col("k") == 0).sum().alias("freeze_rows")])
    cov_parts.append(cov)
    found_m.update(cov["match_key"].to_list())
    agg = aggregate(st).sort(["s", "side_idx"])
    s_idx = agg["s"].to_numpy()
    side_idx = agg["side_idx"].to_numpy()
    vals = np.stack([agg[cn].to_numpy() for cn in FIELDS], axis=1)
    v16 = vals[:, :-1]
    eq = vals[:, -1]
    if v16.size:
        assert v16.max() <= 32767 and v16.min() >= -32768 and eq.max() <= 2147483647
    # agg rows carry the set label through the step's match
    r_of = np.searchsorted(off, s_idx, side="right") - 1
    m_of = rm[r_of]
    code_of = SET_CODE[m_of]
    tgt = code_of == 1
    if tgt.any():
        arr16[s_idx[tgt], side_idx[tgt], :] = v16[tgt].astype(np.int16)
        arr_eq[s_idx[tgt], side_idx[tgt]] = eq[tgt].astype(np.int32)
    for setname, code in (("parity_v3", 2), ("parity_v4", 3)):
        sel = code_of == code
        if sel.any():
            pdf = pl.DataFrame({"s": s_idx[sel], "side_idx": side_idx[sel], "m": m_of[sel]})
            pdf = pdf.with_columns([pl.Series(FIELDS[j], vals[sel, j]) for j in range(len(FIELDS))])
            parity_rows[setname].append(pdf)
    stt = {"chunk": c, "status_pages": len(st_keys), "info_pages": len(info_keys), "status_rows_raw": st_raw_rows,
           "rows_wanted_matches": n_wanted_rows, "rows_no_export_round": n_no_round, "rows_unrostered": n_unrostered,
           "rows_bad_side": n_bad_side, "rows_pre_freeze": n_pre_freeze, "rows_off_grid": n_off_grid,
           "rows_beyond_round_steps": n_beyond, "rows_kept_before_dedupe": n_pre_dedupe, "duplicates_dropped": n_dups,
           "info_roster_conflicts": roster_conflicts, "agg_rows": int(len(s_idx)), "target_agg_rows_written": int(tgt.sum()),
           "matches_found": int(cov.height), "sec": round(time.time() - t1, 1)}
    chunk_stats.append(stt)
    log(json.dumps(stt))
    del st, agg, info, info_u, roster, cov

arr16.flush()
arr_eq.flush()
cov_all = pl.concat(cov_parts)
cov_all = cov_all.with_columns(pl.col("match_key").cast(pl.String))
out_csv = cov_all.write_csv(None)
sys.stdout.write(out_csv)
sys.stdout.flush()
target_found = len(set(TARGET) & found_m)
log("TARGET keys", len(TARGET), "found in channel chunks c000-c062", target_found, "not found", len(TARGET) - target_found)

# ---- parity sample: keep the S3 rows for the parity keys (compared later against the local build) ----
PARITY_OUT = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout_stage/parity_s3_rows.parquet"
frames = []
for setname in ("parity_v3", "parity_v4"):
    for pdf in parity_rows[setname]:
        frames.append(pdf.with_columns(pl.lit(setname).alias("set")))
if frames:
    pall = pl.concat(frames)
    pall = pall.with_columns(pl.Series("match_key", [keys[m] for m in pall["m"].to_list()]))
    pall.write_parquet(PARITY_OUT)
    log("parity S3 rows written", pall.height, "keys", pall["match_key"].n_unique())
else:
    log("no parity rows")
