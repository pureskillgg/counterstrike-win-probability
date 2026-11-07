"""Feasibility (TASK B): smoke, HE and molotov event lists and impact counts on the 150-match sample (read-only).

Inputs: tomes\\2026-10-01_to_08\\tome\\csds (grenade_state, molotov_state, molotov_fire, player_hurt,
player_blind, bomb_state, bomb_action, bomb_defuse) restricted to the clean rounds in
data\\state_sample\\rounds.parquet. Counts only; NOT effects on win probability.
Outputs: printed counts; CSV lists under logs\\feas\\.
Run: pixi run python -I -X utf8 scripts\\eda\\feas_grenade_events.py
"""
import glob
import math
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
OUT = ROOT / "logs" / "feas"
TICK_RATE = 64              # from the task (freeze_end + 64k)
T, CT = 2, 3                # side codes
HE_WINDOW = 1 * TICK_RATE           # ASSUMPTION: 1 s after HE detonation tick
FIRE_WINDOW = 3 * TICK_RATE         # ASSUMPTION: 3 s after molotov fire start tick
SMOKE_LIFE_FALLBACK = 18 * TICK_RATE  # ASSUMPTION (task spec): 18 s smoke life when no expiry row
SMOKE_RADIUS = 150.0                # ASSUMPTION: game units, 2D (x, y) from smoke centre
HE_WEAPONS = ["hegrenade"]
FIRE_WEAPONS = ["inferno", "molotov", "incendiary"]
K = ["match_key", "round", "entity_id_fixed"]


def read(ch, cols=None):
    folder = TOME / f"basic_{ch}{SUFFIX}"
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def dist_table(label, values):
    counts = {"0": 0, "1": 0, "2": 0, ">2": 0}
    for v in values:
        if v is None:
            continue
        key = str(v) if v in (0, 1, 2) else ">2"
        counts[key] += 1
    m = sum(counts.values())
    parts = [f"{k}={counts[k]} ({counts[k] / m:.4f})" if m else f"{k}=0" for k in counts]
    print(f"{label}: n={m} | " + "; ".join(parts))
    return counts


def pctl(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(math.ceil(q * len(sorted_vals))) - 1))
    return sorted_vals[i]


def unmatched_diag(hurt_df, matched_ids, events_df, ecol, label):
    """Why are hurt rows unmatched? Distance from each unmatched hurt row to the nearest event in its round."""
    un = hurt_df.join(matched_ids, on="hid", how="anti").select(["hid", "match_key", "round", "hurt_tick"])
    n_un = un.height
    j = un.join(events_df.select(["match_key", "round", pl.col(ecol).alias("_e")]),
                on=["match_key", "round"], how="inner")
    j = j.with_columns((pl.col("hurt_tick") - pl.col("_e")).alias("diff"))
    n_round = j.select("hid").unique().height
    near = j.with_columns(pl.col("diff").abs().alias("ad")).group_by("hid").agg(
        pl.col("ad").min().alias("min_abs"))
    bins = near.select(
        pl.when(pl.col("min_abs") <= 64).then(pl.lit("<=1 s"))
        .when(pl.col("min_abs") <= 192).then(pl.lit("1-3 s"))
        .otherwise(pl.lit(">3 s")).alias("b")).group_by("b").len().sort("b").rows()
    before = j.filter((pl.col("diff") >= -64) & (pl.col("diff") < 0)).select("hid").unique().height
    late = j.filter(pl.col("diff") > 64).select("hid").unique().height
    print(f"[diag] {label}: unmatched hurt rows={n_un}; of these, an event of the same kind exists in the same "
          f"round for {n_round}; nearest-event |gap| bins (rows): {bins}; "
          f"hurt 0-1 s BEFORE an event tick (rows with any): {before}; hurt >1 s after an event (rows with any): {late}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- sample rounds (clean rounds only) ----------
    rounds = pl.read_parquet(ROOT / "data" / "state_sample" / "rounds.parquet")
    rk_all = rounds.select(["match_key", "round", "live_start_tick", "round_end_tick"])
    print(f"[A] rounds.parquet rows={rounds.height} matches={rounds['match_key'].n_unique()} "
          f"unique(match_key,round)={rk_all.select(['match_key', 'round']).unique().height}")
    rk = rk_all.unique(subset=["match_key", "round"], keep="first")
    keys = rk.select(["match_key", "round"])

    def in_sample(df):
        return df.join(keys, on=["match_key", "round"], how="semi")

    def outside_live(df, tick_col):
        d = df.join(rk, on=["match_key", "round"], how="left")
        return d.filter((pl.col(tick_col) < pl.col("live_start_tick"))
                        | (pl.col(tick_col) > pl.col("round_end_tick"))).height

    gs = in_sample(read("grenade_state"))
    ms = in_sample(read("molotov_state"))
    mf = in_sample(read("molotov_fire", cols=["match_key", "round", "tick", "entity_id_fixed"]))
    hurt_raw = in_sample(read("player_hurt"))
    blind = in_sample(read("player_blind"))
    bstate = in_sample(read("bomb_state"))
    bact = in_sample(read("bomb_action"))
    bdef = in_sample(read("bomb_defuse"))
    for name, df in [("grenade_state", gs), ("molotov_state", ms), ("molotov_fire", mf),
                     ("player_hurt", hurt_raw), ("player_blind", blind), ("bomb_state", bstate),
                     ("bomb_action", bact), ("bomb_defuse", bdef)]:
        print(f"[A] {name}: rows in sample clean rounds={df.height} "
              f"matches={df['match_key'].n_unique()} rounds={df.select(['match_key', 'round']).unique().height}")

    # ---------- per-(match, round, player) side map from all channels ----------
    def pid_team(df, pid_col, team_col):
        return (df.select(["match_key", "round",
                           pl.col(pid_col).cast(pl.Int64, strict=False).alias("pid"),
                           pl.col(team_col).cast(pl.Int64, strict=False).alias("team")])
                  .filter(pl.col("pid").is_not_null() & pl.col("team").is_in([T, CT])))

    srcs = [
        pid_team(hurt_raw, "player_id_fixed", "player_team_code"),
        pid_team(hurt_raw, "attacker_id_fixed", "attacker_team_code"),
        pid_team(blind, "player_id_fixed", "player_team_code"),
        pid_team(blind, "attacker_id_fixed", "attacker_team_code"),
        pid_team(gs, "player_id_fixed", "player_team_code"),
        pid_team(ms, "player_id_fixed", "player_team_code"),
        pid_team(bact, "player_id_fixed", "player_team_code"),
        pid_team(bdef, "player_id_fixed", "player_team_code"),
        pid_team(bstate, "player_id_fixed", "player_team_code"),
    ]
    tall = pl.concat(srcs).unique()
    tgrp = tall.group_by(["match_key", "round", "pid"]).agg(
        pl.col("team").n_unique().alias("n_teams"),
        pl.col("team").min().alias("team_map"),
    )
    n_conf = tgrp.filter(pl.col("n_teams") > 1).height
    tmap = tgrp.filter(pl.col("n_teams") == 1).select(["match_key", "round", "pid", "team_map"])
    print(f"[F] side map: player-round keys={tgrp.height} conflicting (>1 side)={n_conf} "
          f"resolved={tmap.height}")

    def attach_team(df, pid_col, own_col, out_col):
        m = tmap.rename({"pid": pid_col, "team_map": "_tm"})
        d = df.join(m, on=["match_key", "round", pid_col], how="left")
        return d.with_columns(pl.coalesce([pl.col(own_col), pl.col("_tm")]).alias(out_col)).drop("_tm")

    # ---------- player_hurt (weapon_name, victim, attacker) ----------
    hurt = hurt_raw.select([
        "match_key", "round",
        pl.col("tick").alias("hurt_tick"),
        pl.col("weapon_name"),
        pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("victim"),
        pl.col("attacker_id_fixed").cast(pl.Int64, strict=False).alias("attacker"),
        pl.col("player_team_code").cast(pl.Int64, strict=False).alias("victim_team_own"),
    ]).with_row_index("hid")
    wn = hurt_raw.group_by("weapon_name").len().sort("len", descending=True)
    print("[D] player_hurt weapon_name counts in sample (all values):")
    for r in wn.iter_rows():
        print(f"    {r[0]!r}: {r[1]}")

    known = pl.col("victim_team").is_not_null() & pl.col("thrower_team").is_not_null()

    # ---------- HE list and impacts ----------
    he_det = gs.filter(pl.col("event_type") == "hegrenade_detonate").select([
        "match_key", "round", pl.col("tick").alias("det_tick"), "entity_id_fixed", "grenade_id",
        "x_pos", "y_pos", "z_pos",
        pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("thrower"),
        pl.col("player_team_code").cast(pl.Int64, strict=False).alias("thrower_team_own"),
    ])
    he_det = attach_team(he_det, "thrower", "thrower_team_own", "thrower_team")
    print(f"[C] HE detonations in sample clean rounds: {he_det.height}; "
          f"null grenade_id={he_det['grenade_id'].null_count()}; "
          f"null thrower={he_det['thrower'].null_count()}; "
          f"thrower side unknown={he_det.filter(pl.col('thrower_team').is_null()).height}; "
          f"outside live window={outside_live(he_det, 'det_tick')}")

    he_hurt = attach_team(
        hurt.filter(pl.col("weapon_name").is_in(HE_WEAPONS) & pl.col("victim").is_not_null()),
        "victim", "victim_team_own", "victim_team")
    print(f"[C] hegrenade hurt rows in sample: {he_hurt.height}; victim side unknown="
          f"{he_hurt.filter(pl.col('victim_team').is_null()).height}")
    he_pairs = (he_det.select(["match_key", "round", "entity_id_fixed", "det_tick", "thrower", "thrower_team"])
                .join(he_hurt, on=["match_key", "round"], how="inner")
                .filter((pl.col("hurt_tick") >= pl.col("det_tick"))
                        & (pl.col("hurt_tick") <= pl.col("det_tick") + HE_WINDOW)))
    matched_he = he_pairs.select("hid").unique()
    print(f"[C] hegrenade hurt rows matched to an HE detonation within 1 s (ASSUMPTION window): "
          f"{matched_he.height} of {he_hurt.height}; unmatched={he_hurt.height - matched_he.height}; "
          f"of matched, attacker == thrower: "
          f"{he_pairs.filter(pl.col('attacker') == pl.col('thrower')).select('hid').unique().height}")
    unmatched_diag(he_hurt, matched_he, he_det, "det_tick", "HE hurt")
    print(f"[C] HE pairs with unknown side (victim or thrower): {he_pairs.filter(~known).height}")
    enemy_he = he_pairs.filter(known & (pl.col("victim_team") != pl.col("thrower_team")))
    per_he = enemy_he.group_by(K).agg(pl.col("victim").n_unique().alias("n_enemy_1s"))
    per_he_att = (enemy_he.filter(pl.col("attacker") == pl.col("thrower"))
                  .group_by(K).agg(pl.col("victim").n_unique().alias("n_enemy_1s_att_thrower")))
    he_out = (he_det.join(per_he, on=K, how="left").join(per_he_att, on=K, how="left")
              .with_columns(
                  pl.when(pl.col("thrower_team").is_null()).then(None)
                  .otherwise(pl.col("n_enemy_1s").fill_null(0)).alias("n_enemy_1s"),
                  pl.when(pl.col("thrower_team").is_null()).then(None)
                  .otherwise(pl.col("n_enemy_1s_att_thrower").fill_null(0)).alias("n_enemy_1s_att_thrower")))
    dist_table("[C] HE detonations by enemies hurt by hegrenade within 1 s (ASSUMPTION)",
               he_out["n_enemy_1s"].to_list())
    dist_table("[C] sensitivity: attacker == thrower only",
               he_out["n_enemy_1s_att_thrower"].to_list())
    he_out.select(["match_key", "round", "det_tick", "entity_id_fixed", "grenade_id", "x_pos", "y_pos",
                   "z_pos", "thrower", "thrower_team", "n_enemy_1s", "n_enemy_1s_att_thrower"]) \
          .write_csv(OUT / "feas_he_sample.csv")

    # ---------- molotov / incendiary fire starts and impacts ----------
    mol_st = ms.filter(pl.col("event_type") == "inferno_startburn").select([
        "match_key", "round", pl.col("tick").alias("start_tick"), "entity_id_fixed",
        "x_pos", "y_pos", "z_pos",
        pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("thrower"),
        pl.col("player_team_code").cast(pl.Int64, strict=False).alias("thrower_team_own"),
    ])
    mol_st = attach_team(mol_st, "thrower", "thrower_team_own", "thrower_team")
    print(f"[C] molotov fire starts (molotov_state inferno_startburn): {mol_st.height}; "
          f"duplicate (match,round,entity) keys={mol_st.height - mol_st.select(K).unique().height}; "
          f"thrower null={mol_st['thrower'].null_count()}; thrower id known but side unknown="
          f"{mol_st.filter(pl.col('thrower').is_not_null() & pl.col('thrower_team').is_null()).height}; "
          f"outside live window={outside_live(mol_st, 'start_tick')}")
    mf_keys = mf.select(K).unique()
    print(f"[C] molotov_fire distinct (match,round,entity) keys in sample: {mf_keys.height}; "
          f"starts with a molotov_fire row={mol_st.join(mf_keys, on=K, how='semi').height}; "
          f"molotov_fire keys without a startburn={mf_keys.join(mol_st.select(K), on=K, how='anti').height}")

    fire_hurt = attach_team(
        hurt.filter(pl.col("weapon_name").is_in(FIRE_WEAPONS) & pl.col("victim").is_not_null()),
        "victim", "victim_team_own", "victim_team")
    print(f"[C] fire-weapon hurt rows in sample ({FIRE_WEAPONS}): {fire_hurt.height}; victim side unknown="
          f"{fire_hurt.filter(pl.col('victim_team').is_null()).height}")
    f_pairs = (mol_st.select(["match_key", "round", "entity_id_fixed", "start_tick", "thrower", "thrower_team"])
               .join(fire_hurt, on=["match_key", "round"], how="inner")
               .filter((pl.col("hurt_tick") >= pl.col("start_tick"))
                       & (pl.col("hurt_tick") <= pl.col("start_tick") + FIRE_WINDOW)))
    matched_f = f_pairs.select("hid").unique()
    print(f"[C] fire hurt rows matched to a fire start within 3 s (ASSUMPTION window): "
          f"{matched_f.height} of {fire_hurt.height}; unmatched={fire_hurt.height - matched_f.height}; "
          f"of matched, attacker == thrower: "
          f"{f_pairs.filter(pl.col('attacker') == pl.col('thrower')).select('hid').unique().height}")
    unmatched_diag(fire_hurt, matched_f, mol_st, "start_tick", "fire hurt")
    enemy_f = f_pairs.filter(known & (pl.col("victim_team") != pl.col("thrower_team")))
    per_f = enemy_f.group_by(K).agg(pl.col("victim").n_unique().alias("n_enemy_3s"))
    per_f_att = (enemy_f.filter(pl.col("attacker") == pl.col("thrower"))
                 .group_by(K).agg(pl.col("victim").n_unique().alias("n_enemy_3s_att_thrower")))
    mol_out = (mol_st.join(per_f, on=K, how="left").join(per_f_att, on=K, how="left")
               .with_columns(
                   pl.when(pl.col("thrower_team").is_null()).then(None)
                   .otherwise(pl.col("n_enemy_3s").fill_null(0)).alias("n_enemy_3s"),
                   pl.when(pl.col("thrower_team").is_null()).then(None)
                   .otherwise(pl.col("n_enemy_3s_att_thrower").fill_null(0)).alias("n_enemy_3s_att_thrower")))
    dist_table("[C] molotov starts with known thrower side, enemies hurt by fire within 3 s (ASSUMPTION)",
               mol_out["n_enemy_3s"].to_list())
    dist_table("[C] sensitivity: attacker == thrower only",
               mol_out["n_enemy_3s_att_thrower"].to_list())
    mol_out.select(["match_key", "round", "start_tick", "entity_id_fixed", "x_pos", "y_pos", "z_pos",
                    "thrower", "thrower_team", "n_enemy_3s", "n_enemy_3s_att_thrower"]) \
           .write_csv(OUT / "feas_molotov_starts_sample.csv")

    # ---------- sensitivity checks on window placement (not the primary counts) ----------
    LEAD = 16                     # ASSUMPTION: 0.25 s lead, hurt rows logged just before the detonation tick
    BURN_WIN = 7 * TICK_RATE      # ASSUMPTION: approximate molotov burn length (see burn_duration check below)
    bd_vals = sorted([v for v in ms["burn_duration"].to_list() if v is not None])
    print(f"[sens] molotov_state burn_duration non-null rows={len(bd_vals)}: "
          f"p05={pctl(bd_vals, 0.05):.3f} p50={pctl(bd_vals, 0.5):.3f} p95={pctl(bd_vals, 0.95):.3f} s")
    he_lead = (he_det.select(["match_key", "round", "entity_id_fixed", "det_tick", "thrower", "thrower_team"])
               .join(he_hurt, on=["match_key", "round"], how="inner")
               .filter((pl.col("hurt_tick") >= pl.col("det_tick") - LEAD)
                       & (pl.col("hurt_tick") <= pl.col("det_tick") + HE_WINDOW)))
    per_he_lead = (he_lead.filter(known & (pl.col("victim_team") != pl.col("thrower_team")))
                   .group_by(K).agg(pl.col("victim").n_unique().alias("n")))
    he_lead_out = he_det.join(per_he_lead, on=K, how="left").with_columns(pl.col("n").fill_null(0))
    dist_table(f"[sens] HE window [det-{LEAD}, det+64] ticks (lead ASSUMPTION)", he_lead_out["n"].to_list())
    f_long = (mol_st.select(["match_key", "round", "entity_id_fixed", "start_tick", "thrower", "thrower_team"])
              .join(fire_hurt, on=["match_key", "round"], how="inner")
              .filter((pl.col("hurt_tick") >= pl.col("start_tick"))
                      & (pl.col("hurt_tick") <= pl.col("start_tick") + BURN_WIN)))
    per_f_long = (f_long.filter(known & (pl.col("victim_team") != pl.col("thrower_team")))
                  .group_by(K).agg(pl.col("victim").n_unique().alias("n")))
    mol_long = mol_st.join(per_f_long, on=K, how="left").with_columns(
        pl.when(pl.col("thrower_team").is_null()).then(None).otherwise(pl.col("n").fill_null(0)).alias("n"))
    dist_table("[sens] molotov window 0-7 s (ASSUMPTION)", mol_long["n"].to_list())

    # ---------- smoke list (detonate + expired, joined by entity, then grenade_id) ----------
    sm_det = gs.filter(pl.col("event_type") == "smokegrenade_detonate").select([
        "match_key", "round", pl.col("tick").alias("det_tick"), "entity_id_fixed", "grenade_id",
        "x_pos", "y_pos", "z_pos",
        pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("thrower"),
    ])
    sm_exp = gs.filter(pl.col("event_type") == "smokegrenade_expired").select([
        "match_key", "round", pl.col("tick").alias("exp_tick_ent"), "entity_id_fixed", "grenade_id"])
    print(f"[B] smoke detonations in sample: {sm_det.height}; null grenade_id={sm_det['grenade_id'].null_count()}; "
          f"smokegrenade_expired rows: {sm_exp.height}; expired duplicate (match,round,entity) keys="
          f"{sm_exp.height - sm_exp.select(K).unique().height}; detonate duplicate keys="
          f"{sm_det.height - sm_det.select(K).unique().height}; outside live window="
          f"{outside_live(sm_det, 'det_tick')}")
    sm = sm_det.join(sm_exp.select(K + ["exp_tick_ent"]).unique(subset=K, keep="first"), on=K, how="left")
    sm_exp_g = (sm_exp.filter(pl.col("grenade_id").is_not_null())
                .select(["match_key", "round", "grenade_id", pl.col("exp_tick_ent").alias("exp_tick_gid")]))
    print(f"[B] expired rows with grenade_id: {sm_exp_g.height}; duplicate (match,round,grenade_id) keys="
          f"{sm_exp_g.height - sm_exp_g.select(['match_key', 'round', 'grenade_id']).unique().height}")
    sm_exp_g = sm_exp_g.unique(subset=["match_key", "round", "grenade_id"], keep="first")
    sm = sm.join(sm_exp_g, on=["match_key", "round", "grenade_id"], how="left")
    both = sm.filter(pl.col("exp_tick_ent").is_not_null() & pl.col("exp_tick_gid").is_not_null())
    agree = both.filter(pl.col("exp_tick_ent") == pl.col("exp_tick_gid")).height
    print(f"[B] expiry link check where both entity and grenade_id links exist: {both.height}; "
          f"agree={agree}; disagree={both.height - agree}")
    sm = sm.with_columns(
        pl.coalesce([pl.col("exp_tick_ent"), pl.col("exp_tick_gid")]).alias("exp_obs")
    ).with_columns(
        pl.when(pl.col("exp_obs").is_not_null()).then(pl.col("exp_obs"))
        .otherwise(pl.col("det_tick") + SMOKE_LIFE_FALLBACK).alias("exp_final"),
        pl.when(pl.col("exp_tick_ent").is_not_null()).then(pl.lit("expired_row_entity"))
        .when(pl.col("exp_tick_gid").is_not_null()).then(pl.lit("expired_row_grenade_id"))
        .otherwise(pl.lit("fallback_18s")).alias("exp_source"),
    )
    src_counts = sm.group_by("exp_source").len().sort("exp_source").rows()
    print(f"[B] smoke expiry source counts: {src_counts}")
    life = sorted([(r[0] - r[1]) for r in sm.filter(pl.col("exp_obs").is_not_null())
                   .select(["exp_obs", "det_tick"]).iter_rows()])
    neg = sum(1 for v in life if v < 0)
    OBS_LIFE = int(pctl(life, 0.5)) if life else SMOKE_LIFE_FALLBACK
    if life:
        print(f"[B] observed smoke life (exp - det), n={len(life)}: min={life[0]} ticks "
              f"({life[0] / TICK_RATE:.4f} s) p05={pctl(life, 0.05)} p50={pctl(life, 0.5)} "
              f"p95={pctl(life, 0.95)} max={life[-1]} ticks ({life[-1] / TICK_RATE:.4f} s); negative={neg}; "
              f"distinct values={len(set(life))}")
    print(f"[B] smoke detonations (sample): {sm.height}; matches with >=1 smoke: "
          f"{sm['match_key'].n_unique()}; with expiry row: {sm.filter(pl.col('exp_obs').is_not_null()).height}; "
          f"fallback (no expiry row): {sm.filter(pl.col('exp_source') == 'fallback_18s').height}")
    fb = sm.filter(pl.col("exp_source") == "fallback_18s").join(rk, on=["match_key", "round"], how="left")
    print(f"[B] fallback smokes whose 18 s end is after round_end_tick: "
          f"{fb.filter(pl.col('det_tick') + SMOKE_LIFE_FALLBACK > pl.col('round_end_tick')).height} of {fb.height}; "
          f"whose observed-median end (ticks={OBS_LIFE}) is after round_end_tick: "
          f"{fb.filter(pl.col('det_tick') + OBS_LIFE > pl.col('round_end_tick')).height} of {fb.height}")
    sm.select(["match_key", "round", "det_tick", "exp_final", "exp_source", "grenade_id", "entity_id_fixed",
               "x_pos", "y_pos", "z_pos", "thrower"]).write_csv(OUT / "feas_smokes_sample.csv")

    # ---------- bomb events vs active smoke ----------
    def ev_frame(df, etype, label):
        return df.filter(pl.col("event_type") == etype).select([
            "match_key", "round", pl.col("tick").alias("ev_tick"),
            pl.col("player_x_pos").alias("ex"), pl.col("player_y_pos").alias("ey"),
            pl.col("player_z_pos").alias("ez"), pl.lit(label).alias("ev_type")])

    ev = pl.concat([ev_frame(bdef, "bomb_begin_defuse", "defuse"),
                    ev_frame(bact, "bomb_begin_plant", "plant")]).with_row_index("ev_id")
    print("[E] bomb events (sample): per type rows, null player position (player_x_pos):")
    for r in ev.group_by("ev_type").agg(pl.len().alias("n"), pl.col("ex").null_count().alias("null_pos")).iter_rows():
        print(f"    {r[0]}: n={r[1]} null_pos={r[2]} ({r[2] / r[1]:.4f})")

    ev_rk = ev.join(rk, on=["match_key", "round"], how="left")
    for r in ev_rk.group_by("ev_type").agg(
            (pl.col("ev_tick") > pl.col("round_end_tick")).sum().alias("after_end"),
            (pl.col("ev_tick") < pl.col("live_start_tick")).sum().alias("before_live")).iter_rows():
        print(f"    {r[0]}: events after round_end_tick={r[1]}; before live_start_tick={r[2]}")

    def bomb_check(fallback_expr, label):
        s = sm.join(rk.select(["match_key", "round", "round_end_tick"]), on=["match_key", "round"], how="left")
        s = s.with_columns(
            pl.coalesce([pl.col("exp_obs"), fallback_expr]).alias("exp_f"),
            (pl.col("exp_source") == "fallback_18s").alias("is_fb"))
        evs = ev.join(s.select(["match_key", "round", "det_tick", "exp_f", "is_fb",
                                "x_pos", "y_pos", "z_pos"]), on=["match_key", "round"], how="left")
        evs = evs.with_columns(
            ((pl.col("det_tick") <= pl.col("ev_tick")) & (pl.col("ev_tick") <= pl.col("exp_f")))
            .fill_null(False).alias("active"),
            ((pl.col("ex") - pl.col("x_pos")) ** 2 + (pl.col("ey") - pl.col("y_pos")) ** 2).sqrt().alias("d2"),
            ((pl.col("ex") - pl.col("x_pos")) ** 2 + (pl.col("ey") - pl.col("y_pos")) ** 2
             + (pl.col("ez") - pl.col("z_pos")) ** 2).sqrt().alias("d3"),
        ).with_columns(
            (pl.col("active") & (pl.col("d2") <= SMOKE_RADIUS)).fill_null(False).alias("near2d"),
            (pl.col("active") & (pl.col("d3") <= SMOKE_RADIUS)).fill_null(False).alias("near3d"),
            (pl.col("active") & (pl.col("d2") <= SMOKE_RADIUS) & (~pl.col("is_fb")))
            .fill_null(False).alias("near2d_exact"),
        )
        per_ev = evs.group_by("ev_id").agg(
            pl.col("active").any().alias("any_active"),
            pl.col("near2d").any().alias("any_near2d"),
            pl.col("near3d").any().alias("any_near3d"),
            pl.col("near2d_exact").any().alias("any_near2d_exact"),
        )
        out = ev.join(per_ev, on="ev_id", how="left").with_columns(
            pl.col("any_active").fill_null(False), pl.col("any_near2d").fill_null(False),
            pl.col("any_near3d").fill_null(False), pl.col("any_near2d_exact").fill_null(False))
        print(f"[E] --- {label}")
        for et in ["defuse", "plant"]:
            sub = out.filter(pl.col("ev_type") == et)
            print(f"[E] {et}: n={sub.height}; null position={sub.filter(pl.col('ex').is_null()).height}; "
                  f"active smoke (time only)={sub.filter(pl.col('any_active')).height}; "
                  f"active and null position={sub.filter(pl.col('any_active') & pl.col('ex').is_null()).height}; "
                  f"active and within {SMOKE_RADIUS:.0f} 2D={sub.filter(pl.col('any_near2d')).height}; "
                  f"within {SMOKE_RADIUS:.0f} 3D={sub.filter(pl.col('any_near3d')).height}; "
                  f"2D with exact-expiry smokes only={sub.filter(pl.col('any_near2d_exact')).height}")
        after_end = out.join(rk, on=["match_key", "round"], how="left").filter(
            pl.col("any_near2d") & (pl.col("ev_tick") > pl.col("round_end_tick"))).height
        print(f"[E] all bomb events: n={out.height}; null position={out.filter(pl.col('ex').is_null()).height}; "
              f"active smoke (time only)={out.filter(pl.col('any_active')).height}; "
              f"active and within {SMOKE_RADIUS:.0f} 2D={out.filter(pl.col('any_near2d')).height} "
              f"(of which event tick after round_end_tick: {after_end})")
        return out

    ev_primary = bomb_check(pl.col("det_tick") + SMOKE_LIFE_FALLBACK,
                            "PRIMARY: no expiry row -> detonation + 18 s (task spec ASSUMPTION)")
    bomb_check(pl.col("det_tick") + OBS_LIFE,
               f"SENSITIVITY: no expiry row -> detonation + observed life ({OBS_LIFE} ticks)")
    bomb_check(pl.col("round_end_tick"),
               "SENSITIVITY: no expiry row -> smoke lasts to round_end_tick (ASSUMPTION)")
    ev_primary.select(["ev_id", "match_key", "round", "ev_type", "ev_tick", "ex", "ey", "ez", "any_active",
                       "any_near2d", "any_near3d", "any_near2d_exact"]) \
              .write_csv(OUT / "feas_bomb_smoke_check.csv")
    print("[done] CSV lists written to", OUT)


if __name__ == "__main__":
    main()
