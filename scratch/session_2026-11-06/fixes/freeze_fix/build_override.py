"""Authoritative side table for the confirmed freeze-tick roster defect (runs on the space; reads local raw copies only).

Scope: the 23 investigated matches (fixes/f1_mismatch_rounds.csv), rounds that the scope scan (freeze_scope/scope_rounds.parquet)
flags as affected. Only rounds with the confirmed pattern are fixed: exactly one alive player at freeze whose player_info rows carry
two team codes (the stray duplicate) AND exactly one alive player with no player_info row. Every other round is left as it is and
reported.

Side per alive player at freeze, from the raw player_status place_name at the freeze tick (CTSpawn -> 3, TSpawn -> 2):
  single-code player : the code (checked against place_name)
  duplicate player   : the code equal to its place_name side (must be one of its two codes)
  unrostered player  : the code left over from the duplicate (checked against its own place_name side)
Round gate: 10 alive players at freeze, 5 CT and 5 T, every place_name known, no conflict. Rounds failing the gate are not written.

Writes: freeze_fix/override_rows.parquet (match_key, round, player_id, side, rule, phys) for gated rounds only,
        freeze_fix/override_rounds.csv (every affected round: status and reason).
"""
import csv
import collections
import json
import os
import polars as pl

FIX = "/home/sagemaker-user/adam/builder/fixes"
OUT = FIX + "/freeze_fix"
os.makedirs(OUT, exist_ok=True)
inv = sorted(set(r["match_key"] for r in csv.DictReader(open(FIX + "/f1_mismatch_rounds.csv"))))
scope = pl.read_parquet(FIX + "/freeze_scope/scope_rounds.parquet")
A = scope.filter(pl.col("affected") & pl.col("match_key").is_in(inv))
print("investigated matches", len(inv), "| affected rounds in them", A.height, flush=True)

rows_out, round_out = [], []
missing_raw = []
for mk in inv:
    d = FIX + "/raw/" + mk.rsplit("/", 1)[0].replace("/", "_")
    if not os.path.exists(d + "/player_status") or not os.path.exists(d + "/player_info"):
        missing_raw.append(mk)
        continue
    ps = pl.read_parquet(d + "/player_status", columns=["tick", "round", "player_id", "health", "place_name"]).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    pi = pl.read_parquet(d + "/player_info", columns=["round", "player_id", "team_code"]).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("team_code").cast(pl.Int64))
    for rd, fz in A.filter(pl.col("match_key") == mk).select("round", "freeze").rows():
        st = ps.filter((pl.col("round") == rd) & (pl.col("tick") == fz) & (pl.col("health") > 0))
        codes = collections.defaultdict(set)
        for pid, tc in pi.filter(pl.col("round") == rd).select("player_id", "team_code").rows():
            codes[pid].add(tc)
        phys = {pid: (3 if pl_ == "CTSpawn" else (2 if pl_ == "TSpawn" else 0)) for pid, pl_ in st.select("player_id", "place_name").rows()}
        alive = sorted(phys)
        dups = [p for p in alive if len(codes.get(p, ())) > 1]
        unro = [p for p in alive if len(codes.get(p, ())) == 0]
        reason = None
        assign = {}
        if len(dups) != 1 or len(unro) != 1:
            reason = f"pattern dups={len(dups)} unrostered={len(unro)} (not the confirmed pattern)"
        else:
            p0, u = dups[0], unro[0]
            ph0 = phys[p0]
            if ph0 not in codes[p0]:
                reason = f"duplicate pid {p0} place side {ph0} not in codes {sorted(codes[p0])}"
            else:
                left = sorted(codes[p0] - {ph0})
                side_u = left[0] if len(left) == 1 else None
                if side_u is None:
                    reason = "no complementary code"
                elif phys[u] in (2, 3) and phys[u] != side_u:
                    reason = f"unrostered pid {u} place side {phys[u]} != complementary code {side_u}"
                else:
                    assign[p0] = ph0
                    assign[u] = side_u
                    for p in alive:
                        if p in assign:
                            continue
                        cs = codes.get(p, set())
                        if len(cs) != 1:
                            reason = f"unexpected alive pid {p} codes {sorted(cs)}"
                            break
                        c = next(iter(cs))
                        if phys[p] in (2, 3) and phys[p] != c:
                            reason = f"single-code pid {p} code {c} != place side {phys[p]}"
                            break
                        assign[p] = c
        if reason is None:
            n_ct = sum(1 for p in alive if assign.get(p) == 3)
            n_t = sum(1 for p in alive if assign.get(p) == 2)
            if len(alive) != 10 or n_ct != 5 or n_t != 5 or any(p not in assign for p in alive):
                reason = f"gate failed: alive {len(alive)} CT {n_ct} T {n_t}"
        status = "fixed" if reason is None else "not_fixed"
        round_out.append({"match_key": mk, "round": rd, "freeze": fz, "alive": len(alive), "dups": str(dups), "unrostered": str(unro),
                          "status": status, "reason": reason or "", "override": json.dumps({str(k): v for k, v in assign.items()}) if reason is None else ""})
        if reason is None:
            for p in alive:
                rows_out.append({"match_key": mk, "round": rd, "player_id": p, "side": assign[p], "rule": "dup_phys" if p == dups[0] else ("unro_comp" if p == unro[0] else "single_code"), "phys": phys[p]})

print("raw missing for", len(missing_raw), missing_raw[:3], flush=True)
R = pl.DataFrame(round_out)
R.write_csv(OUT + "/override_rounds.csv")
print("affected rounds in investigated matches", R.height, "| fixed", R.filter(pl.col("status") == "fixed").height,
      "| not fixed", R.filter(pl.col("status") == "not_fixed").height, flush=True)
print(R.group_by("reason").len().sort("len", descending=True).head(12), flush=True)
if rows_out:
    O = pl.DataFrame(rows_out).with_columns(pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64),
                                             pl.col("side").cast(pl.Int64), pl.col("phys").cast(pl.Int64))
    O.write_parquet(OUT + "/override_rows.parquet")
    print("override rows", O.height, "| rounds", O.select("match_key", "round").unique().height, "| rules", O.group_by("rule").len().to_dicts(), flush=True)
