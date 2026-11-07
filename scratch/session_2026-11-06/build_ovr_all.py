"""Build the side-override table for every round to be fixed (283 already fixed + 296 open rounds that pass the place check).

Side = place_name of each alive player at the freeze tick: CTSpawn -> 3 (CT), TSpawn -> 2 (T).
Sanity: place at freeze must reproduce the existing override (fixes/freeze_fix/override_rows.parquet) for the 283 fixed rounds.
Writes only under fixes/fix332/ (new files). Never touches the export folders.
"""
import glob
import json
import os
import polars as pl

RAW = "/home/sagemaker-user/adam/builder/raw_cache_bucket/"
FX = "/home/sagemaker-user/adam/builder/fixes/"
OUT = FX + "fix332/v5_build/"
os.makedirs(OUT, exist_ok=True)

cand = pl.read_csv(FX + "fix332/candidates.csv", infer_schema_length=0).with_columns(
    pl.col("round").cast(pl.Int64), pl.col("freeze_tick").cast(pl.Int64))
chk = pl.read_csv(FX + "fix332/side_rule_332_check.csv", infer_schema_length=0).with_columns(
    pl.col("round").cast(pl.Int64), pl.col("gate_5v5").str.to_lowercase().eq("true").alias("pass"))
ovr = pl.read_parquet(FX + "freeze_fix/override_rows.parquet").select(
    "match_key", pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("side").cast(pl.Int64))

fixed_keys = set(zip(ovr["match_key"].to_list(), ovr["round"].to_list()))
open_c = cand.join(chk.select("match_key", "round", "pass"), on=["match_key", "round"], how="left")
open_c = open_c.filter(~pl.struct(["match_key", "round"]).map_elements(
    lambda s: (s["match_key"], s["round"]) in fixed_keys, return_dtype=pl.Boolean))
pass_rounds = open_c.filter(pl.col("pass") == True)  # noqa: E712
flag_rounds = open_c.filter(pl.col("pass") != True)  # noqa: E712
print("open rounds", open_c.height, "pass", pass_rounds.height, "flagged", flag_rounds.height, flush=True)

targets = pl.concat([
    cand.filter(pl.struct(["match_key", "round"]).map_elements(
        lambda s: (s["match_key"], s["round"]) in fixed_keys, return_dtype=pl.Boolean)).select("match_key", "round", "freeze_tick").with_columns(pl.lit("fixed283").alias("source")),
    pass_rounds.select("match_key", "round", "freeze_tick").with_columns(pl.lit("open_pass").alias("source")),
])
print("target rounds", targets.height, flush=True)


def place_map(mk, rnd, freeze):
    folder = RAW + os.path.dirname(mk) + "/"
    st = pl.read_parquet(folder + "player_status", columns=["tick", "round", "player_id", "health", "place_name"])
    st = st.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    a = st.filter((pl.col("round") == rnd) & (pl.col("tick") == freeze) & (pl.col("health") > 0))
    a = a.unique(["player_id"], keep="first")
    rows = []
    for pid, place in a.select("player_id", "place_name").iter_rows():
        side = 3 if place == "CTSpawn" else 2 if place == "TSpawn" else None
        rows.append((pid, side, place))
    return rows


out_rows = []
problems = []
per_round = []
for mk, rnd, frz, src in targets.iter_rows():
    rows = place_map(mk, rnd, frz)
    sides = [s for _, s, _ in rows]
    n_ct, n_t, n_unk = sides.count(3), sides.count(2), sides.count(None)
    ok = (n_ct == 5 and n_t == 5 and n_unk == 0 and len(rows) == 10)
    per_round.append({"match_key": mk, "round": rnd, "source": src, "alive": len(rows), "ct": n_ct, "t": n_t,
                      "unknown": n_unk, "gate_5v5": ok})
    if not ok:
        problems.append({"match_key": mk, "round": rnd, "source": src, "alive": len(rows), "ct": n_ct, "t": n_t, "unknown": n_unk})
        continue
    for pid, side, place in rows:
        out_rows.append({"match_key": mk, "round": rnd, "player_id": pid, "side": side, "source": src})

ov_all = pl.DataFrame(out_rows, schema={"match_key": pl.String, "round": pl.Int64, "player_id": pl.Int64,
                                        "side": pl.Int64, "source": pl.String})
ov_all.write_parquet(OUT + "ovr_all.parquet")
pr = pl.DataFrame(per_round)
pr.write_csv(OUT + "per_round_gate.csv")
flag_rounds.select("match_key", "round", "freeze_tick").write_csv(OUT + "flagged_36.csv")

# sanity: place at freeze must reproduce the existing override on the 283 fixed rounds
fx = ov_all.filter(pl.col("source") == "fixed283").select("match_key", "round", "player_id", pl.col("side").alias("place_side"))
cmp = fx.join(ovr, on=["match_key", "round", "player_id"], how="full", coalesce=True)
mismatch = cmp.filter(pl.col("place_side") != pl.col("side")).height
summary = {
    "open_rounds_checked": open_c.height,
    "pass_rounds": pass_rounds.height,
    "flagged_rounds": flag_rounds.height,
    "target_rounds_written": int(pr.filter(pl.col("gate_5v5")).height),
    "target_rounds_failing_gate": len(problems),
    "player_rows_written": ov_all.height,
    "fixed283_rows_compared": int(cmp.height),
    "fixed283_place_vs_existing_override_mismatches": int(mismatch),
    "problems_sample": problems[:10],
}
json.dump(summary, open(OUT + "build_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
