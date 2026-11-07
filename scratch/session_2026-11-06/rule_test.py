"""fix332: side-rule test for the unrostered alive player and the duplicate-id player at freeze.

Scope: the 23 matches in freeze_fix/inv_matches.json, fixed rounds of freeze_fix/override_rounds.csv.
Reads raw_cache_bucket match folders only. Writes only under fixes/fix332/.
Sides: 2 = T, 3 = CT (as in build_override.py).

Candidates for a subject player s (the unrostered player u, or the duplicate-id player p0):
  A_player_vector_mode : mode of player_vector.team_code (values 2/3) for s in the round
  B_override_place     : override side from override_rounds.csv (place_name at freeze)
  C_player_spawn       : mode of player_spawn.player_team_code (values 2/3) for s in the round, when present

Kill scoring: every player kill (attacker not null, attacker != victim) that involves s.
  counterpart sides, two references:
    ref_override       : override side of the counterpart (alive at freeze); unknown if not alive at freeze
    ref_player_vector  : player_vector mode of the counterpart (non-circular sensitivity)
  consistent = sides differ, same = sides equal, unknown = a side is unknown.
  World kills (attacker null) and suicides are counted separately and not scored.

5 CT / 5 T check at freeze:
  others_from_override : the 9 other alive players take their override side; s takes the candidate side
  pure_source          : all 10 alive players take the candidate's own source (B = override)
"""
import collections
import csv
import json
import os
import time

import polars as pl

T0 = time.time()
RAW = "/home/sagemaker-user/adam/builder/raw_cache_bucket/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"
OUT = "/home/sagemaker-user/adam/builder/fixes/fix332/"
os.makedirs(OUT, exist_ok=True)
SIDES = (2, 3)
CANDS = ["A_player_vector_mode", "B_override_place", "C_player_spawn"]
REFS = ["ref_override", "ref_player_vector"]
SUBJECTS = ["unrostered", "duplicate_id"]


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def mode_map(df, val):
    df = df.filter(pl.col(val).is_in(list(SIDES)))
    acc = collections.defaultdict(list)
    for rd, pid, v, n in df.group_by(["round", "player_id", val]).len().rows():
        acc[(int(rd), int(pid))].append((int(n), int(v)))
    out = {}
    for k, lst in acc.items():
        lst.sort(key=lambda t: (-t[0], t[1]))
        out[k] = {"side": lst[0][1], "tie": len(lst) > 1 and lst[0][0] == lst[1][0],
                  "codes_seen": sorted(v for _, v in lst)}
    return out


def sd(m, k):
    v = m.get(k)
    return None if v is None else v["side"]


def cls(cs, cv):
    if cs is None or cv is None:
        return "unknown"
    return "consistent" if cs != cv else "same"


inv = json.load(open(FX + "inv_matches.json"))
ov_all = pl.read_csv(FX + "override_rounds.csv", infer_schema_length=0)
log("override rows", ov_all.height, "status", ov_all.group_by("status").len().to_dicts())
fixed = ov_all.filter(pl.col("status") == "fixed")
log("fixed rounds", fixed.height, "matches with fixed", fixed["match_key"].n_unique(), "of inv", len(inv))
assert set(fixed["match_key"].unique().to_list()) <= set(inv)

ACC = {s: {c: {r: collections.Counter() for r in REFS} for c in CANDS} for s in SUBJECTS}
FIVE = {s: {c: collections.Counter() for c in CANDS} for s in SUBJECTS}
PURE = {s: {c: collections.Counter() for c in CANDS} for s in SUBJECTS}
KILLS = collections.Counter()
WORLD = collections.Counter()
SUICIDE = collections.Counter()
NOT_ALIVE_CP = collections.Counter()
CHECK = collections.Counter()
PLACE = collections.Counter()
AGREE = collections.Counter()
ROWS = []
rounds_per_match = {}

for key in inv:
    fx = fixed.filter(pl.col("match_key") == key)
    if fx.height == 0:
        continue
    d = RAW + os.path.dirname(key) + "/"
    rl = sorted(set(int(x) for x in fx["round"].to_list()))
    rounds_per_match[key] = len(rl)
    pv_df = pl.read_parquet(d + "player_vector", columns=["round", "player_id", "team_code"]).filter(pl.col("round").is_in(rl))
    sp_df = pl.read_parquet(d + "player_spawn", columns=["round", "player_id", "player_team_code"]).filter(pl.col("round").is_in(rl))
    dt_df = pl.read_parquet(d + "player_death", columns=["round", "player_id", "attacker_id"]).filter(pl.col("round").is_in(rl))
    in_df = pl.read_parquet(d + "player_info", columns=["round", "player_id", "team_code"]).filter(pl.col("round").is_in(rl))
    ps_df = pl.read_parquet(d + "player_status", columns=["tick", "round", "player_id", "player_controller_id", "health", "place_name"]).filter(pl.col("round").is_in(rl))
    pvm = mode_map(pv_df, "team_code")
    spm = mode_map(sp_df, "player_team_code")
    infos = collections.defaultdict(set)
    for rd, pid, tc in in_df.filter(pl.col("team_code").is_not_null()).select(["round", "player_id", "team_code"]).rows():
        infos[(int(rd), int(pid))].add(int(tc))
    deaths = collections.defaultdict(list)
    for rd, vic, att in dt_df.select(["round", "player_id", "attacker_id"]).rows():
        deaths[int(rd)].append((None if vic is None else int(vic), None if att is None else int(att)))
    log("loaded", key.split("/")[3][:8], "rounds", len(rl), "pv modes", len(pvm), "spawn modes", len(spm))

    for row in fx.iter_rows(named=True):
        rd = int(row["round"])
        fz = int(row["freeze"])
        ovmap = {int(k): int(v) for k, v in json.loads(row["override"]).items()}
        dups = [int(x) for x in json.loads(row["dups"])]
        unro = [int(x) for x in json.loads(row["unrostered"])]
        if len(dups) != 1 or len(unro) != 1:
            CHECK["fixed_row_without_one_dup_and_one_unrostered"] += 1
            continue
        p0, u = dups[0], unro[0]
        CHECK["fixed_rounds_scored"] += 1

        st = ps_df.filter((pl.col("round") == rd) & (pl.col("tick") == fz) & (pl.col("health") > 0)).select(
            ["player_id", "player_controller_id", "place_name"]).rows()
        pids = [int(p) for p, _, _ in st]
        CHECK["freeze_alive_rows_eq_10"] += int(len(st) == 10)
        CHECK["freeze_alive_unique_pid_eq_10"] += int(len(set(pids)) == 10)
        CHECK["freeze_alive_rows_gt_unique_pid"] += int(len(st) > len(set(pids)))
        CHECK["p0_alive_rows_at_freeze_eq_1"] += int(pids.count(p0) == 1)
        CHECK["u_alive_rows_at_freeze_eq_1"] += int(pids.count(u) == 1)
        p0_ctrl_round = set(int(c) for (c,) in ps_df.filter((pl.col("round") == rd) & (pl.col("player_id") == p0)).select(["player_controller_id"]).rows())
        CHECK["p0_distinct_controllers_in_round_gt1"] += int(len(p0_ctrl_round) > 1)
        CHECK["u_has_player_info_row"] += int(bool(infos.get((rd, u))))
        CHECK["p0_info_codes_are_2_and_3"] += int(infos.get((rd, p0)) == {2, 3})

        place_side = {}
        for p, c, pl_ in st:
            name = str(pl_)
            ph = 3 if name == "CTSpawn" else (2 if name == "TSpawn" else 0)
            PLACE[name] += 1
            place_side[int(p)] = ph
        pl_u = place_side.get(u, 0)
        pl_p0 = place_side.get(p0, 0)
        CHECK["u_place_unknown_at_freeze"] += int(pl_u not in SIDES)
        CHECK["p0_place_unknown_at_freeze"] += int(pl_p0 not in SIDES)
        CHECK["p0_place_side_eq_override"] += int(pl_p0 == ovmap[p0])
        CHECK["u_place_known_and_eq_override"] += int(pl_u in SIDES and pl_u == ovmap[u])
        CHECK["u_place_known_and_ne_override"] += int(pl_u in SIDES and pl_u != ovmap[u])

        pu = {"A": sd(pvm, (rd, u)), "C": sd(spm, (rd, u))}
        pp = {"A": sd(pvm, (rd, p0)), "C": sd(spm, (rd, p0))}
        AGREE["u_pv_missing"] += int(pu["A"] is None)
        AGREE["u_spawn_missing"] += int(pu["C"] is None)
        AGREE["u_pv_tie"] += int(bool(pvm.get((rd, u), {}).get("tie")))
        AGREE["u_pv_eq_override"] += int(pu["A"] == ovmap[u])
        AGREE["u_spawn_present_and_eq_override"] += int(pu["C"] is not None and pu["C"] == ovmap[u])
        AGREE["u_spawn_present_and_ne_override"] += int(pu["C"] is not None and pu["C"] != ovmap[u])
        AGREE["p0_pv_missing"] += int(pp["A"] is None)
        AGREE["p0_spawn_missing"] += int(pp["C"] is None)
        AGREE["p0_pv_tie"] += int(bool(pvm.get((rd, p0), {}).get("tie")))
        AGREE["p0_pv_eq_place"] += int(pp["A"] == ovmap[p0])
        AGREE["p0_spawn_present_and_eq_place"] += int(pp["C"] is not None and pp["C"] == ovmap[p0])
        AGREE["p0_spawn_present_and_ne_place"] += int(pp["C"] is not None and pp["C"] != ovmap[p0])

        rec = {"match_key": key, "round": rd, "freeze": fz, "unrostered": u, "dup_id": p0}
        for subj, s in (("unrostered", u), ("duplicate_id", p0)):
            cands = {"A_player_vector_mode": sd(pvm, (rd, s)), "B_override_place": ovmap[s],
                     "C_player_spawn": sd(spm, (rd, s))}
            kl = []
            for vic, att in deaths.get(rd, []):
                if vic is None:
                    continue
                if att is None:
                    if vic == s:
                        WORLD[subj] += 1
                    continue
                if att == vic:
                    if vic == s:
                        SUICIDE[subj] += 1
                    continue
                if vic == s:
                    kl.append(att)
                elif att == s:
                    kl.append(vic)
            KILLS[subj] += len(kl)
            for v in kl:
                if v not in ovmap:
                    NOT_ALIVE_CP[subj] += 1
                for c, cs in cands.items():
                    ACC[subj][c]["ref_override"][cls(cs, ovmap.get(v))] += 1
                    ACC[subj][c]["ref_player_vector"][cls(cs, sd(pvm, (rd, v)))] += 1
            src_map = {"A_player_vector_mode": pvm, "C_player_spawn": spm}
            for c, cs in cands.items():
                if cs is None:
                    FIVE[subj][c]["candidate_missing"] += 1
                else:
                    sides = [cs if v == s else ovmap[v] for v in ovmap]
                    ct, t = sides.count(3), sides.count(2)
                    FIVE[subj][c]["rounds_with_candidate"] += 1
                    FIVE[subj][c]["ct5_t5"] += int(ct == 5 and t == 5)
                if c in src_map:
                    pure = [sd(src_map[c], (rd, v)) for v in ovmap]
                else:
                    pure = [ovmap[v] for v in ovmap]
                if all(x is not None for x in pure):
                    PURE[subj][c]["all10_available"] += 1
                    PURE[subj][c]["ct5_t5"] += int(pure.count(3) == 5 and pure.count(2) == 5)
                else:
                    PURE[subj][c]["not_all10_available"] += 1
            rec[subj + "_kills"] = len(kl)
            for c, cs in cands.items():
                rec[f"{subj}_{c}"] = "" if cs is None else cs
        rec["override"] = row["override"]
        ROWS.append(rec)

log("scored rounds", CHECK["fixed_rounds_scored"])


def score_block(subj):
    out = {}
    for c in CANDS:
        out[c] = {}
        for r in REFS:
            a = ACC[subj][c][r]
            known = a["consistent"] + a["same"]
            tot = known + a["unknown"]
            out[c][r] = {
                "consistent": a["consistent"], "same_side": a["same"], "unknown": a["unknown"], "total_kills": tot,
                "consistency_rate_known": round(a["consistent"] / known, 6) if known else None,
                "same_side_rate_known": round(a["same"] / known, 6) if known else None,
                "unknown_rate": round(a["unknown"] / tot, 6) if tot else None,
            }
        fv = FIVE[subj][c]
        pu = PURE[subj][c]
        out[c]["five_five_others_from_override"] = {
            "rounds_with_candidate": fv["rounds_with_candidate"], "candidate_missing": fv["candidate_missing"],
            "ct5_t5": fv["ct5_t5"],
            "rate_of_available": round(fv["ct5_t5"] / fv["rounds_with_candidate"], 6) if fv["rounds_with_candidate"] else None}
        out[c]["five_five_pure_source"] = {
            "all10_available": pu["all10_available"], "not_all10_available": pu["not_all10_available"],
            "ct5_t5": pu["ct5_t5"],
            "rate_of_available": round(pu["ct5_t5"] / pu["all10_available"], 6) if pu["all10_available"] else None}
    return out


result = {
    "task": "side rule for the unrostered alive player and place rule for duplicate-id players, kill consistency, 23 matches",
    "scope": {"matches_in_inv": len(inv), "matches_with_fixed_rounds": len(rounds_per_match),
              "fixed_rounds_scored": CHECK["fixed_rounds_scored"],
              "fixed_rounds_by_match": rounds_per_match,
              "override_source": "fixes/freeze_fix/override_rounds.csv (status=fixed)",
              "raw_root": RAW, "sides": {"2": "T", "3": "CT"},
              "not_written": ["export_clean_v2", "export_clean_v3_loadout", "export_clean_v4"]},
    "subjects": {s: {"kills_involving_subject": KILLS[s], "world_kills_excluded": WORLD[s],
                     "suicides_excluded": SUICIDE[s], "counterpart_not_alive_at_freeze": NOT_ALIVE_CP[s],
                     "candidates": score_block(s)} for s in SUBJECTS},
    "freeze_and_roster_checks": dict(CHECK),
    "side_agreement": dict(AGREE),
    "place_name_counts_alive_at_freeze": dict(PLACE),
}
json.dump(result, open(OUT + "rule_test.json", "w"), indent=2, sort_keys=True)
with open(OUT + "side_rule_rounds.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys()))
    w.writeheader()
    w.writerows(ROWS)
log("wrote", OUT + "rule_test.json", "rows", len(ROWS))
for s in SUBJECTS:
    log("==", s, "kills", KILLS[s], "world", WORLD[s], "suicide", SUICIDE[s], "not_alive_cp", NOT_ALIVE_CP[s])
    for c in CANDS:
        for r in REFS:
            x = result["subjects"][s]["candidates"][c][r]
            log(f"  {c:22s} {r:18s} cons={x['consistent']} same={x['same_side']} unk={x['unknown']} rate={x['consistency_rate_known']}")
        log(f"  {c:22s} 5v5 others: {result['subjects'][s]['candidates'][c]['five_five_others_from_override']}")
        log(f"  {c:22s} 5v5 pure:   {result['subjects'][s]['candidates'][c]['five_five_pure_source']}")
log("checks", dict(CHECK))
log("agree", dict(AGREE))
log("place", dict(PLACE))
log("done")
