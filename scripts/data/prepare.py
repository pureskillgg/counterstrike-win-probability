"""Match catalog + round table from the combined tome store.

Reads  tomes/<store>/tome/csds (built by build_combined.py).
Writes data/<store>/matches.parquet, rounds.parquet, prep_report.json

matches.parquet  one row per match: era, upload_day, build_num, regime, map, mode flags, scope, channel completeness,
                 duplicate group + keeper flag, round counts.
rounds.parquet   one row per (match, round): ticks, roster, deaths in the live window, bomb events, four label signals,
                 final label + source, eligibility reason (first failing rule) and the effective end tick.

Label signals (winner team code, T = 2, CT = 3):
  rec    round_end.winner_team_code (as recorded)
  bomb   3 if a bomb_defused event, 2 if a bomb_exploded event, inside the round's live window
  score  the side whose freeze-end score rose between round r and r+1 (side-attached scores; undefined on the swap
         boundary rounds, where A6-8 found a boundary artifact, and on the last round of a match)
  elim   3 if all five T died in the window with no plant before the last T death, 2 if all five CT died
Final label: bomb if present (the bomb side is the truth when present: planter / defuser team codes agree);
otherwise the majority of the available {rec, score, elim}; two signals that disagree, or a single recorded signal on
a round the 2024 build-9961 defect can invert (first round after a side swap), leave the round unlabeled and it is
dropped. Nothing is flipped by round number alone.

  pixi run python -I -X utf8 scripts\\data\\prepare.py [--store combined]
"""
import argparse
import collections
import json
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
T, CT = 2, 3
TICK = 64
IN_SCOPE_MAPS = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]
DENSE = ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state"]
SWAP_PRONE_BUILD = 9961  # 2024 builds whose recorded round_end winner is inverted after a side swap (D9, D5, A6)


def swap_after(r, max_rounds):
    """Sides swap after completed round r: halftime, and after the 3rd round of each 6-round overtime period."""
    return (r == max_rounds // 2) | ((r > max_rounds) & (((r - max_rounds - 3) % 6) == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    args = ap.parse_args()
    store = os.path.join(ROOT, "tomes", args.store, "tome", "csds")
    outdir = os.path.join(ROOT, "data", args.store)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    rep = {"store": args.store}

    def scan(ch, cols=None):
        folder = "header" if ch == "header" else f"basic_{ch}.COMBINED"
        lf = pl.scan_parquet(os.path.join(store, folder, "dataframe_*"))
        return lf.select(cols) if cols else lf

    def keys(ch):
        folder = f"basic_{ch}.COMBINED"
        return set(pl.read_parquet(os.path.join(store, folder, "keyset_00000"))["_"].to_list())

    # ------------------------------------------------------------------ matches
    H = (scan("header", ["key", "build_num", "map_name", "match_date", "match_type", "is_wingman", "max_rounds",
                         "t_starters_avg_rank", "ct_starters_avg_rank", "t_starters_score_final", "ct_starters_score_final"])
         .collect().rename({"key": "match_key"}))
    H = H.with_columns(
        pl.col("match_key").str.extract(r"^csds/(\d{4})/", 1).alias("era"),
        pl.col("match_key").str.extract(r"^csds/(\d{4}/\d{2}/\d{2})/", 1).alias("upload_day"),
    )
    print(f"header matches {H.height}  {dict(H.group_by('era').len().iter_rows())}", flush=True)

    RE = (scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count_norm"])
          .collect().sort(["match_key", "round", "tick"]))
    n_dup_re = RE.height - RE.unique(["match_key", "round"]).height
    RE = RE.unique(["match_key", "round"], keep="first", maintain_order=True)
    rep["round_end_rows"] = RE.height
    rep["round_end_duplicate_rows_dropped"] = n_dup_re

    mode_pc = RE.group_by("match_key").agg(pl.col("player_count_norm").mode().first().alias("modal_pc"),
                                           pl.len().alias("n_round_end"))
    H = H.join(mode_pc, on="match_key", how="left")
    H = H.with_columns(
        pl.when(pl.col("is_wingman").is_not_null()).then(pl.col("is_wingman"))
        .otherwise(pl.col("modal_pc") <= 5).fill_null(False).alias("is_wingman_final"),
        (pl.col("is_wingman").is_not_null() & ((pl.col("modal_pc") <= 5) != pl.col("is_wingman"))).alias("wingman_flag_disagrees"),
    )
    H = H.with_columns(
        pl.when(pl.col("max_rounds").is_not_null()).then(pl.col("max_rounds"))
        .otherwise(pl.when(pl.col("is_wingman_final")).then(16).otherwise(24)).alias("max_rounds_final"),
        pl.col("map_name").fill_null("").str.starts_with("cs_").alias("is_hostage_map"),
    )
    H = H.with_columns(
        pl.when(pl.col("is_wingman_final")).then(pl.lit("wingman"))
        .when(pl.col("is_hostage_map")).then(pl.lit("hostage"))
        .when(pl.col("map_name").is_null() | (pl.col("map_name") == "")).then(pl.lit("map_unknown"))
        .when(~pl.col("map_name").is_in(IN_SCOPE_MAPS)).then(pl.lit("map_out_of_pool"))
        .otherwise(pl.lit("in_scope")).alias("scope_reason"))
    H = H.with_columns((pl.col("scope_reason") == "in_scope").alias("in_scope"))
    # regime A: round_announce_match_start present (2024 build 9961 era; the winner inversion lives here)
    RS_all = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect()
    regime = (RS_all.filter(pl.col("event_type") == "round_announce_match_start").group_by("match_key").len()
              .select("match_key", pl.lit(True).alias("regime_a")))
    H = H.join(regime, on="match_key", how="left").with_columns(pl.col("regime_a").fill_null(False))
    # channel completeness (dense channels: key in keyset)
    ks = {ch: keys(ch) for ch in DENSE}
    H = H.with_columns(*[pl.col("match_key").is_in(list(ks[ch])).alias(f"has_{ch}") for ch in DENSE])
    H = H.with_columns(pl.all_horizontal([pl.col(f"has_{ch}") for ch in DENSE]).alias("channels_complete"))
    print(f"scope reasons: {dict(H.group_by('scope_reason').len().iter_rows())}", flush=True)

    # ------------------------------------------------------------------ rounds
    FZ = (RS_all.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
          .unique(["match_key", "round"], keep="first", maintain_order=True)
          .select("match_key", "round", pl.col("tick").alias("freeze"), pl.col("t_score").alias("t_sc"),
                  pl.col("ct_score").alias("ct_sc")))
    RST = scan("round_start", ["match_key", "round"]).unique().collect().with_columns(pl.lit(True).alias("has_round_start"))
    PI = (scan("player_info", ["match_key", "round", "player_id", "team_code"]).collect()
          .unique(["match_key", "round", "player_id"], keep="last", maintain_order=True))
    roster = PI.group_by(["match_key", "round"]).agg(
        pl.len().alias("n_players"), (pl.col("team_code") == T).sum().alias("n_t"),
        (pl.col("team_code") == CT).sum().alias("n_ct"))
    DISC = pl.concat([scan("player_disconnect", ["match_key", "round"]).collect(),
                      scan("bot_takeover", ["match_key", "round"]).collect()]).unique().with_columns(
        pl.lit(True).alias("disc_or_bot"))
    R = (RE.rename({"tick": "end", "winner_team_code": "rec"})
         .join(FZ, on=["match_key", "round"], how="left")
         .join(RST, on=["match_key", "round"], how="left")
         .join(roster, on=["match_key", "round"], how="left")
         .join(DISC, on=["match_key", "round"], how="left")
         .join(H.select("match_key", "era", "build_num", "regime_a", "max_rounds_final", "in_scope"), on="match_key", how="left"))
    R = R.with_columns(pl.col("has_round_start").fill_null(False), pl.col("disc_or_bot").fill_null(False))
    # 2024 player_info often lacks one player (no team code) although round_end counts 10. With 5 on one side and 4 on
    # the other the missing player must be on the 4-player side (a side cannot have 6): the round is a real 5v5.
    R = R.with_columns(
        pl.when((pl.col("player_count_norm") == 10) & (pl.col("n_players") == 9) & (pl.col("n_t") == 5) & (pl.col("n_ct") == 4)).then(CT)
        .when((pl.col("player_count_norm") == 10) & (pl.col("n_players") == 9) & (pl.col("n_t") == 4) & (pl.col("n_ct") == 5)).then(T)
        .otherwise(None).alias("missing_side"))
    R = R.with_columns(
        (pl.col("n_players").is_not_null() & (((pl.col("n_players") == 10) & (pl.col("n_t") == 5) & (pl.col("n_ct") == 5))
                                              | pl.col("missing_side").is_not_null())).alias("roster_ok"))

    # deaths in the live window; side from the event, else from the round roster
    DE = scan("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
    DE = (DE.join(R.select("match_key", "round", "freeze", "end", "missing_side"), on=["match_key", "round"], how="inner")
          .filter(pl.col("freeze").is_not_null() & (pl.col("tick") >= pl.col("freeze")) & (pl.col("tick") <= pl.col("end")))
          .sort(["match_key", "round", "tick"])
          .unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
          .join(PI.rename({"team_code": "roster_team"}), on=["match_key", "round", "player_id"], how="left")
          # event team code, else the roster, else (a player absent from the roster) the side with 4 known players
          .with_columns(pl.coalesce([pl.col("player_team_code"), pl.col("roster_team")]).alias("side")))
    # The missing-player rule is only valid for ONE player absent from both the event and the roster. If several victims
    # of a round are unresolved, the roster does not describe this round (ids shifted): leave them unresolved.
    DE = DE.with_columns((pl.col("side").is_null() & pl.col("missing_side").is_not_null()).alias("_fallback"))
    DE = DE.with_columns(pl.col("_fallback").sum().over(["match_key", "round"]).alias("_n_fallback"))
    DE = DE.with_columns(pl.when(pl.col("_fallback") & (pl.col("_n_fallback") == 1)).then(pl.col("missing_side")).otherwise(pl.col("side")).alias("side"))
    # team codes are 2 (T) or 3 (CT); cast so is_in compares like with like under Polars 2.x
    DE = DE.with_columns(pl.col("side").cast(pl.Int64))
    DE = DE.with_columns(pl.col("side").is_in([T, CT]).alias("side_ok"))
    dth = DE.group_by(["match_key", "round"]).agg(
        pl.len().alias("n_deaths"), (pl.col("side") == T).sum().alias("deaths_t"), (pl.col("side") == CT).sum().alias("deaths_ct"),
        (~pl.col("side_ok")).sum().alias("deaths_unresolved"), pl.col("tick").max().alias("last_death_tick"))
    # the tick of the 5th death of the wiped side (when a side is wiped)
    w5 = (DE.filter(pl.col("side_ok")).with_columns(pl.col("tick").rank("ordinal").over(["match_key", "round", "side"]).alias("k"))
          .filter(pl.col("k") == 5))
    wipe = (w5.filter(pl.col("side") == CT).select("match_key", "round", pl.col("tick").alias("ct_wipe_tick"))
            .join(w5.filter(pl.col("side") == T).select("match_key", "round", pl.col("tick").alias("t_wipe_tick")),
                  on=["match_key", "round"], how="full", coalesce=True))  # tick of the 5th death of each side
    # bomb events
    BS = scan("bomb_state", ["match_key", "round", "tick", "event_type"]).collect().join(
        R.select("match_key", "round", "freeze", "end"), on=["match_key", "round"], how="inner").filter(
        pl.col("freeze").is_not_null() & (pl.col("tick") >= pl.col("freeze")) & (pl.col("tick") <= pl.col("end") + TICK))
    plant = BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("plant_tick"))
    dfz = BS.filter(pl.col("event_type") == "bomb_defused").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("defuse_tick"))
    exp = BS.filter(pl.col("event_type") == "bomb_exploded").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("explode_tick"))
    R = (R.join(dth, on=["match_key", "round"], how="left").join(wipe, on=["match_key", "round"], how="left")
         .join(plant, on=["match_key", "round"], how="left").join(dfz, on=["match_key", "round"], how="left")
         .join(exp, on=["match_key", "round"], how="left"))
    R = R.with_columns(pl.col("n_deaths").fill_null(0), pl.col("deaths_t").fill_null(0), pl.col("deaths_ct").fill_null(0),
                       pl.col("deaths_unresolved").fill_null(0))

    # ---- label signals
    R = R.with_columns(
        pl.when(pl.col("defuse_tick").is_not_null()).then(CT).when(pl.col("explode_tick").is_not_null()).then(T)
        .otherwise(None).alias("bomb"))
    # elimination: a CT wipe decides for T at once; a T wipe decides for CT only if the bomb was not already planted when the
    # last T died. The earliest decisive wipe counts (both sides can end up dead, e.g. CT killed by the blast).
    R = R.with_columns(pl.when(pl.col("plant_tick").is_null() | (pl.col("plant_tick") > pl.col("t_wipe_tick")))
                       .then(pl.col("t_wipe_tick")).otherwise(None).alias("_t_decisive"))
    R = R.with_columns(pl.min_horizontal(pl.col("ct_wipe_tick"), pl.col("_t_decisive")).alias("wipe_tick"))
    R = R.with_columns(
        pl.when(pl.col("wipe_tick").is_null()).then(None)
        .when(pl.col("ct_wipe_tick") == pl.col("wipe_tick")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("wiped_side"))
    R = R.with_columns(
        pl.when(pl.col("wiped_side") == CT).then(T).when(pl.col("wiped_side") == T).then(CT).otherwise(None).alias("elim"))
    nxt = R.select("match_key", (pl.col("round") - 1).alias("round"), pl.col("t_sc").alias("t_nx"), pl.col("ct_sc").alias("ct_nx"))
    R = R.join(nxt, on=["match_key", "round"], how="left")
    R = R.with_columns(
        swap_after(pl.col("round"), pl.col("max_rounds_final")).alias("swap_boundary"))
    dt_, dc_ = pl.col("t_nx") - pl.col("t_sc"), pl.col("ct_nx") - pl.col("ct_sc")
    R = R.with_columns(
        pl.when(pl.col("swap_boundary") | pl.col("t_nx").is_null() | pl.col("t_sc").is_null()).then(None)
        .when((dt_ == 1) & (dc_ == 0)).then(T).when((dc_ == 1) & (dt_ == 0)).then(CT).otherwise(None).alias("score"))
    # suspect rounds: the first round of each half-period (13; overtime 25, 28, 31, ...). Build 9961 records the winner
    # inverted there (D9/D5/A6). Note the score FIELDS swap only at halftime and mid-overtime (swap_boundary, checked on the
    # clean builds: rounds 12, 27, 33, 39 swap, 24 and 30 do not), so these are two different schedules.
    R = R.with_columns(
        ((pl.col("round") == pl.col("max_rounds_final") // 2 + 1)
         | ((pl.col("round") > pl.col("max_rounds_final")) & (((pl.col("round") - pl.col("max_rounds_final") - 1) % 3) == 0))).alias("after_swap"))
    R = R.with_columns((pl.col("after_swap") & pl.col("regime_a") & (pl.col("build_num") == SWAP_PRONE_BUILD)).alias("rec_suspect"))
    sig = ["rec", "score", "elim"]
    # votes per side among the three (rec counted only if not suspect)
    R = R.with_columns(
        pl.when(pl.col("rec_suspect")).then(None).otherwise(pl.col("rec")).alias("rec_v"))
    v2 = sum([(pl.col(c) == T).cast(pl.Int8).fill_null(0) for c in ("rec_v", "score", "elim")])
    v3 = sum([(pl.col(c) == CT).cast(pl.Int8).fill_null(0) for c in ("rec_v", "score", "elim")])
    R = R.with_columns(v2.alias("votes_t"), v3.alias("votes_ct"))
    R = R.with_columns(
        pl.when(pl.col("bomb").is_not_null()).then(pl.col("bomb"))
        .when((pl.col("votes_t") >= 2) & (pl.col("votes_t") > pl.col("votes_ct"))).then(T)
        .when((pl.col("votes_ct") >= 2) & (pl.col("votes_ct") > pl.col("votes_t"))).then(CT)
        .when((pl.col("votes_t") == 1) & (pl.col("votes_ct") == 0) & (pl.col("rec_v") == T)).then(T)   # recorded only, not suspect
        .when((pl.col("votes_ct") == 1) & (pl.col("votes_t") == 0) & (pl.col("rec_v") == CT)).then(CT)
        .otherwise(None).alias("final"))
    R = R.with_columns(
        pl.when(pl.col("bomb").is_not_null()).then(pl.lit("bomb"))
        .when(pl.col("final").is_null()).then(pl.lit("unresolved"))
        .when(pl.col("votes_t") + pl.col("votes_ct") >= 2).then(pl.lit("majority"))
        .otherwise(pl.lit("recorded_only")).alias("label_source"))
    R = R.with_columns((pl.col("final") != pl.col("rec")).alias("final_differs_from_recorded"))

    # ---- effective end: the last undecided second is before the decisive event (bomb outcome, or the wiping death)
    # earliest moment the outcome was fixed: a defuse, an explosion, or the wipe (CT wiped; T wiped before a plant). A later
    # bomb event does not move it: in 2024 a round can show the CT wipe first and the explosion a few seconds after.
    dec = pl.min_horizontal(
        pl.when(pl.col("final") == CT).then(pl.col("defuse_tick")).otherwise(None),
        pl.when(pl.col("final") == T).then(pl.col("explode_tick")).otherwise(None),
        pl.when(pl.col("elim").is_not_null()).then(pl.col("wipe_tick")).otherwise(None))
    R = R.with_columns(dec.alias("decisive_tick"))
    R = R.with_columns(pl.min_horizontal(pl.col("end"), pl.col("decisive_tick").fill_null(pl.col("end"))).alias("end_eff"))
    R = R.with_columns((pl.col("end") - pl.col("end_eff")).alias("end_gap_ticks"))

    # ---- eligibility: first failing rule, legacy order
    R = R.with_columns(
        pl.when(pl.col("player_count_norm") != 10).then(pl.lit("player_count != 10"))
        .when(~pl.col("roster_ok")).then(pl.lit("roster not 5v5"))
        .when(pl.col("disc_or_bot")).then(pl.lit("disconnect or bot takeover"))
        .when(~pl.col("has_round_start")).then(pl.lit("no round_start"))
        .when(pl.col("freeze").is_null()).then(pl.lit("no freeze_end"))
        .when(pl.col("deaths_unresolved") > 0).then(pl.lit("death side unresolved"))
        .when((pl.col("deaths_t") > 5) | (pl.col("deaths_ct") > 5)).then(pl.lit("more than 5 deaths on a side"))
        .when(pl.col("final").is_null()).then(pl.lit("label unresolved"))
        .when(pl.col("end_eff") <= pl.col("freeze")).then(pl.lit("no undecided second"))
        .otherwise(pl.lit("ok")).alias("round_status"))

    # ---- duplicates: identical round_end + player_death + player_info event signature
    def sig_part(df, cols, name):
        return (df.select("match_key", (pl.struct(cols).hash() % 2147483647).cast(pl.Int64).alias("h"))
                .group_by("match_key").agg(pl.len().alias(f"n_{name}"), pl.col("h").sum().alias(f"s_{name}")))
    DEall = scan("player_death", ["match_key", "round", "tick", "player_id"]).collect()
    sg = (sig_part(RE.select("match_key", "round", "tick", "winner_team_code"), ["round", "tick", "winner_team_code"], "re")
          .join(sig_part(DEall, ["round", "tick", "player_id"], "de"), on="match_key", how="full", coalesce=True)
          .join(sig_part(PI, ["round", "player_id", "team_code"], "pi"), on="match_key", how="full", coalesce=True)
          .fill_null(0))
    H = H.join(sg, on="match_key", how="left")
    sigcols = ["n_re", "s_re", "n_de", "s_de", "n_pi", "s_pi"]
    ok_by_match = R.filter(pl.col("round_status") == "ok").group_by("match_key").agg(pl.len().alias("n_ok_rounds"))
    H = H.join(ok_by_match, on="match_key", how="left").with_columns(pl.col("n_ok_rounds").fill_null(0))
    H = H.with_columns(pl.struct(sigcols).hash().alias("sig_hash"))
    # tolerant key: a re-processed copy keeps the map, the round count and the whole sequence of round winners but may shift
    # ticks or renumber players, so it escapes the exact event signature. Matches under 16 rounds use the exact signature only.
    dcount = DEall.group_by(["match_key", "round"]).agg(pl.len().alias("dc"))
    wseq = (RE.sort(["match_key", "round"]).join(dcount, on=["match_key", "round"], how="left").with_columns(pl.col("dc").fill_null(0))
            .group_by("match_key", maintain_order=True)
            .agg((pl.col("winner_team_code").cast(pl.String) + pl.col("dc").cast(pl.String) + pl.lit(",")).str.join("").alias("wseq"),
                 pl.col("round").min().alias("r0"), pl.len().alias("n_rounds_re")))
    H = H.join(wseq, on="match_key", how="left")
    H = H.with_columns(
        pl.when(pl.col("n_rounds_re").fill_null(0) >= 16)
        .then(pl.concat_str([pl.col("map_name"), pl.col("r0").cast(pl.String), pl.col("n_rounds_re").cast(pl.String),
                             pl.col("t_starters_score_final").cast(pl.String), pl.col("ct_starters_score_final").cast(pl.String),
                             pl.col("wseq")], separator="|").hash())
        .otherwise(pl.col("sig_hash")).alias("dup_key"))
    cnt = H.group_by("dup_key").len().rename({"len": "dup_group_size"})
    H = H.join(cnt, on="dup_key", how="left")
    H = H.with_columns(
        pl.when(pl.col("dup_group_size") > 1).then(pl.col("dup_key")).otherwise(None).alias("dup_group"),
        pl.col("n_round_end").fill_null(0))
    # keeper: most eligible rounds, then lowest key
    H = H.sort(["dup_key", "n_ok_rounds", "match_key"], descending=[False, True, False])
    H = H.with_columns(pl.int_range(pl.len()).over("dup_key").alias("_rank_in_group"))
    H = H.with_columns((pl.col("_rank_in_group") == 0).alias("dup_keeper")).drop("_rank_in_group")
    ndist = H.filter(pl.col("dup_group_size") > 1).group_by("dup_key").agg(pl.col("sig_hash").n_unique().alias("n_sig"))
    rep["dup_groups_tolerant"] = ndist.height
    rep["dup_groups_with_different_exact_signature"] = int((ndist["n_sig"] > 1).sum())

    R = R.join(H.select("match_key", "dup_keeper", "channels_complete", "dup_group_size"), on="match_key", how="left")
    R = R.with_columns(
        (pl.col("round_status").eq("ok") & pl.col("in_scope") & pl.col("dup_keeper") & pl.col("channels_complete")).alias("use"))

    # ------------------------------------------------------------------ outputs + report
    H.write_parquet(os.path.join(outdir, "matches.parquet"))
    R.write_parquet(os.path.join(outdir, "rounds.parquet"))
    DE.select("match_key", "round", "tick", "player_id", "side", "side_ok").write_parquet(os.path.join(outdir, "deaths.parquet"))

    def counts(df, by):
        return {str(k): v for k, v in df.group_by(by).len().sort(by).iter_rows()}

    rep["matches"] = H.height
    rep["matches_by_era"] = counts(H, "era")
    rep["scope_by_era"] = {f"{e}|{r}": n for e, r, n in H.group_by(["era", "scope_reason"]).len().sort(["era", "scope_reason"]).iter_rows()}
    rep["wingman_flag_disagrees"] = int(H["wingman_flag_disagrees"].sum())
    rep["channels_incomplete_matches"] = int((~H["channels_complete"]).sum())
    rep["dup_groups"] = int(H.filter(pl.col("dup_group_size") > 1)["dup_key"].n_unique())
    rep["dup_matches_removed"] = int((~H["dup_keeper"]).sum())
    rep["rounds_total"] = R.height
    rep["round_status"] = {f"{e}|{s}": n for e, s, n in R.group_by(["era", "round_status"]).len().sort(["era", "round_status"]).iter_rows()}
    rep["label_source_ok_rounds"] = {f"{e}|{s}": n for e, s, n in R.filter(pl.col("round_status") == "ok").group_by(["era", "label_source"]).len().sort(["era", "label_source"]).iter_rows()}
    ok = R.filter(pl.col("round_status") == "ok")
    rep["final_differs_from_recorded_ok_rounds"] = {
        f"{e}|{b}": [int(a), int(c)] for e, b, a, c in ok.group_by(["era", "build_num"]).agg(
            pl.col("final_differs_from_recorded").sum(), pl.len()).sort(["era", "build_num"]).iter_rows()}
    rep["use_rounds"] = int(R["use"].sum())
    rep["use_matches"] = int(R.filter(pl.col("use"))["match_key"].n_unique())
    rep["use_by_era"] = {e: [int(m), int(r)] for e, m, r in R.filter(pl.col("use")).group_by("era").agg(pl.col("match_key").n_unique(), pl.len()).sort("era").iter_rows()}
    rep["end_gap_ticks_ok_rounds"] = {
        e: {"median": float(g["end_gap_ticks"].median()), "p90": float(g["end_gap_ticks"].quantile(0.9)),
            "share_gt_64": float((g["end_gap_ticks"] > 64).mean())}
        for (e,), g in ok.group_by("era")}
    rep["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(outdir, "prep_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
