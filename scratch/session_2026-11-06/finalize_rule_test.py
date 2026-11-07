import json
import os

import polars as pl

RAW = "/home/sagemaker-user/adam/builder/raw_cache_bucket/"
FX = "/home/sagemaker-user/adam/builder/freeze_fix/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"
OUT = "/home/sagemaker-user/adam/builder/fixes/fix332/"
p = OUT + "rule_test.json"
r = json.load(open(p))
inv = json.load(open(FX + "inv_matches.json"))
ovall = pl.read_csv(FX + "override_rounds.csv", infer_schema_length=0)
fixed = ovall.filter(pl.col("status") == "fixed")
nf = ovall.filter(pl.col("status") != "fixed")

same = []
for key in inv:
    rows = list(fixed.filter(pl.col("match_key") == key).iter_rows(named=True))
    if not rows:
        continue
    d = RAW + os.path.dirname(key) + "/"
    rl = sorted(set(int(x["round"]) for x in rows))
    dt = pl.read_parquet(d + "player_death", columns=["round", "tick", "player_id", "attacker_id"]).filter(pl.col("round").is_in(rl))
    for x in rows:
        rd = int(x["round"])
        ovm = {int(k): int(v) for k, v in json.loads(x["override"]).items()}
        u = int(json.loads(x["unrostered"])[0])
        p0 = int(json.loads(x["dups"])[0])
        for _, tick, vic, att in dt.filter(pl.col("round") == rd).rows():
            if vic is None or att is None or int(att) == int(vic):
                continue
            vic, att = int(vic), int(att)
            if vic in ovm and att in ovm and ovm[vic] == ovm[att] and (u in (vic, att) or p0 in (vic, att)):
                same.append({"match_key": key, "round": rd, "tick": int(tick), "attacker": att,
                             "attacker_side": ovm[att], "victim": vic, "victim_side": ovm[vic],
                             "subject": "unrostered" if u in (vic, att) else "duplicate_id"})

C = r["subjects"]["unrostered"]["candidates"]
D = r["subjects"]["duplicate_id"]["candidates"]
Bu = C["B_override_place"]["ref_override"]
Au = C["A_player_vector_mode"]["ref_override"]
Cu = C["C_player_spawn"]["ref_override"]
Bp = D["B_override_place"]["ref_override"]
Ap = D["A_player_vector_mode"]["ref_override"]
Cp = D["C_player_spawn"]["ref_override"]
Bp5 = D["B_override_place"]["five_five_others_from_override"]
Ap5 = D["A_player_vector_mode"]["five_five_others_from_override"]
Cp5 = D["C_player_spawn"]["five_five_others_from_override"]
Bu_pv = C["B_override_place"]["ref_player_vector"]
n_fixed = r["scope"]["fixed_rounds_scored"]
agree = r["side_agreement"]
checks = r["freeze_and_roster_checks"]

r["extra_checks"] = {
    "same_side_kills_under_override_reference": same,
    "not_fixed_rows": nf.select(["match_key", "round", "freeze", "alive", "dups", "unrostered", "reason"]).to_dicts(),
    "unrostered_team_code_source": {
        "player_vector_rows_for_unrostered_id": 1281409,
        "player_vector_team_code_null_rows": 1281409,
        "player_spawn_rows_for_unrostered_id": 283,
        "player_spawn_team_code_null_rows": 283,
        "player_info_rows_for_unrostered_id": 0,
        "probe": "fixes/fix332/probe_u.py (values from its run)",
        "note": "team codes for the unrostered id are empty in both sources in all 283 rounds; this is consistent with a roster join, which prepare.py uses for the death side; not verified against the export code."},
    "duplicate_id_roster_code_vs_place": {
        "rounds_roster_code_eq_place": agree["p0_pv_eq_place"],
        "rounds_roster_code_ne_place": n_fixed - agree["p0_pv_eq_place"]},
    "override_gate_note": "build_override.py docstring says every place_name must be known, but the unrostered branch only checks the place side when it is 2 or 3; an unknown place (0) would pass. No effect on the 283 fixed rounds: all 2830 alive rows had CTSpawn or TSpawn.",
}

r["decision"] = {
    "unrostered_side_rule": "B_override_place: side = place_name at freeze (CTSpawn -> 3, TSpawn -> 2)",
    "unrostered_side_rule_status": "chosen by elimination: A and C have no side for the unrostered id; B validated in absolute kill consistency only",
    "unrostered_side_rule_basis": (f"A and C are null for the unrostered id in all {n_fixed} rounds. B: {Bu['consistent']} consistent, "
                                   f"{Bu['same_side']} same-side, {Bu['unknown']} unknown of {Bu['total_kills']} kills involving the id "
                                   f"(consistency {Bu['consistency_rate_known']})."),
    "duplicate_id_place_rule": "confirmed: side = place_name at freeze for the duplicate id",
    "duplicate_id_place_rule_basis": (f"B: {Bp['consistent']} consistent, {Bp['same_side']} same-side, {Bp['unknown']} unknown of {Bp['total_kills']} kills "
                                      f"(consistency {Bp['consistency_rate_known']}). A and C (roster-derived code): {Ap['consistent']} consistent, "
                                      f"{Ap['same_side']} same-side (consistency {Ap['consistency_rate_known']}). 5 CT / 5 T at freeze: B "
                                      f"{Bp5['ct5_t5']}/{Bp5['rounds_with_candidate']}, A {Ap5['ct5_t5']}/{Ap5['rounds_with_candidate']}, "
                                      f"C {Cp5['ct5_t5']}/{Cp5['rounds_with_candidate']}."),
    "duplicate_id_status_facts": (f"freeze alive rows = 10 in {checks['freeze_alive_rows_eq_10']}/{n_fixed} rounds; duplicate id has one alive row and "
                                  f"one controller per round in {checks['p0_alive_rows_at_freeze_eq_1']}/{n_fixed} and "
                                  f"{n_fixed - checks['p0_distinct_controllers_in_round_gt1']}/{n_fixed} rounds, so place_name is unambiguous."),
}

r["blockers"] = [
    "Rules A and C cannot be tested for the unrostered player: its team codes are null in player_vector (every row) and player_spawn (every row) in all 283 rounds, and it has no player_info row. A and C therefore give no side for it and cannot be ranked against B.",
    f"B for the unrostered player has no comparator. Its validation is kill consistency only ({Bu['consistent']}/{Bu['total_kills']} consistent, {Bu['same_side']} same-side). The 5 CT / 5 T result ({Bu['total_kills'] and C['B_override_place']['five_five_others_from_override']['ct5_t5']}/283) holds by construction of the override gate and is not independent evidence.",
    f"Two same-side kills involve the unrostered id as victim: round 7 of csds/2026/07/07 and csds/2026/07/09 (see extra_checks). They may be real team kills or a side error; the data here cannot separate the two. They are about 0.5% of the kills scored.",
    f"The counterpart reference matters. With player_vector-mode counterparts the unrostered id has {Bu_pv['same_side']} same-side kills, because the duplicate id's roster-derived code is the wrong side in {n_fixed - agree['p0_pv_eq_place']} of {n_fixed} rounds. The place side is the reference that holds, so the test is not fully independent of the place rule.",
    "The 332 round count in the request does not appear in these inputs. The 23 matches have 284 affected rows in override_rounds.csv (283 fixed, 1 not fixed). freeze_scope/scope_summary.json reports 615 affected rounds in 45 matches. Not reconciled.",
    "One not-fixed round (csds/2026/06/28/m_partial, round 18, alive 9, unrostered id 6, no duplicate id) is outside the confirmed pattern and is not covered by this test.",
    "build_override.py does not enforce the every-place-known gate for the unrostered player (see extra_checks.override_gate_note). No effect on the 283 fixed rounds.",
]

r["next_steps"] = [
    "Keep B as the side for the unrostered alive player and the place rule for the duplicate id; both are confirmed on kill consistency.",
    "Open the two same-side kills on the unrostered id (extra_checks.same_side_kills_under_override_reference) and decide whether they are team kills.",
    "Add an explicit known-place check for the unrostered branch of build_override.py before any rebuild; do not rebuild export_clean_v2, v3_loadout or v4 from this test alone.",
    "Reconcile the 332 round count against the source that produced it before reporting the fix totals.",
]

with open(p, "w") as fh:
    json.dump(r, fh, indent=2, sort_keys=True)
print("rewrote", p, "same_side_kills", len(same), "decision keys", sorted(r["decision"].keys()))
print("B u:", Bu)
print("B p0:", Bp, "A p0:", Ap, "C p0:", Cp)
print("5v5 p0 B/A/C:", Bp5, Ap5, Cp5)
print("same:", same)
print("done")
