"""A6 label and win-code audit, both eras (2024 tome, 2026 tome).

Checks, per round row of round_end (all rows, no sampling):
  1. ct_win = (winner_team_code == 3); winner domain {2, 3}; duplicate (match_key, round) keys.
  2. win_reason_code -> win_reason_message is a function; message matches the canonical CS2 enum
     (names as written in the tome; same table as feature_validation.WIN_REASON).
  3. win_reason_code -> winning side. ASSUMPTION: canonical enum semantics (EXPECTED_SIDE).
  4. bomb_state flags per round (planted / defused / exploded) vs win code and winner.
  5. Death evidence: codes 8 and 9 (elimination endings) should have all players of the losing
     side dead by round end. Victims joined to the round roster (player_info team_code).
  6. Score evidence: winner from round_state round_freeze_end score deltas between consecutive
     rounds, side-based (no swap) and swapped variant. 2026 validated against score_update.
  7. Fingerprint dedupe of matches (identical round_end sequence), label stats on deduped set.
  8. 2024 legacy_code column: distribution and overlap with the bomb-side mismatch.

Run (from project root):
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A6_label_audit.py
"""
import collections
import glob
import json
import re
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
ERAS = {
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
}
T, CT = 2, 3
# Canonical CS2/CS:GO round-end enum, message strings as written in the tome.
WIN_REASON = {
    1: "#SFUI_Notice_Target_Bombed", 7: "#SFUI_Notice_Bomb_Defused", 8: "#SFUI_Notice_CTs_Win",
    9: "#SFUI_Notice_Terrorists_Win", 11: "#SFUI_Notice_All_Hostages_Rescued",
    12: "#SFUI_Notice_Target_Saved", 13: "#SFUI_Notice_Hostages_Not_Rescued",
    17: "#SFUI_Notice_Terrorists_Surrender", 18: "#SFUI_Notice_CTs_Surrender",
}
# ASSUMPTION: side that wins under each reason, taken from the message name.
EXPECTED_SIDE = {1: T, 7: CT, 8: CT, 9: T, 11: CT, 12: CT, 13: T, 17: CT, 18: T}
BOMB_EVS = ["bomb_planted", "bomb_defused", "bomb_exploded"]


def files(base, ch):
    folder = base / "header" if ch == "header" else sorted(base.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(folder / "dataframe_*")))


def load(base, ch, cols):
    return pl.scan_parquet(files(base, ch)).select(cols).collect()


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def audit_era(era, base):
    S = {}
    section(f"ERA {era}  base={base}")
    re_cols = ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
               "win_reason_message", "player_count"]
    if era == "2024":
        re_cols.append("legacy_code")
    RE = load(base, "round_end", re_cols).with_columns(
        pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day"))
    S["round_end_rows"] = RE.height
    S["round_end_matches"] = RE["match_key"].n_unique()
    print(f"round_end rows={RE.height} matches={S['round_end_matches']}")
    print("null counts:", {c: RE[c].null_count() for c in RE.columns if RE[c].null_count()})

    # ---- 1. ct_win and duplicate keys
    print("\n[1] winner domain and ct_win")
    print(RE.group_by("winner_team_code").len().sort("winner_team_code"))
    S["winner_outside_2_3"] = RE.filter(~pl.col("winner_team_code").is_in([T, CT])).height
    S["ct_win_rows"] = RE.filter(pl.col("winner_team_code") == CT).height
    S["t_win_rows"] = RE.filter(pl.col("winner_team_code") == T).height
    S["ct_share_all_rows"] = round(S["ct_win_rows"] / RE.height, 6)
    print(f"ct_win (winner==3) rows={S['ct_win_rows']} t_win rows={S['t_win_rows']} "
          f"ct_share={S['ct_share_all_rows']} winner_outside_2_3={S['winner_outside_2_3']}")
    dup = RE.group_by(["match_key", "round"]).agg(
        pl.len().alias("n"),
        pl.col("winner_team_code").n_unique().alias("nw"),
        pl.col("win_reason_code").n_unique().alias("nc"))
    dupk = dup.filter(pl.col("n") > 1)
    S["dup_round_keys"] = dupk.height
    S["dup_round_keys_conflicting_winner"] = dupk.filter(pl.col("nw") > 1).height
    S["dup_round_keys_conflicting_code"] = dupk.filter(pl.col("nc") > 1).height
    print(f"duplicate (match_key, round) keys={dupk.height}; conflicting winner="
          f"{S['dup_round_keys_conflicting_winner']}; conflicting code={S['dup_round_keys_conflicting_code']}")
    per_m = RE.group_by("match_key").agg(
        pl.col("round").n_unique().alias("nr"),
        pl.col("round").max().alias("mx"),
        pl.col("round").min().alias("mn"))
    S["matches_rounds_not_contiguous_from_1"] = per_m.filter(
        (pl.col("mn") != 1) | (pl.col("mx") != pl.col("nr"))).height
    print("rounds per match (top 12):")
    print(per_m.group_by("nr").len().sort("len", descending=True).head(12))
    print(f"matches with rounds not 1..n contiguous: {S['matches_rounds_not_contiguous_from_1']}")

    # ---- 2 and 3. code -> message and code -> side
    print("\n[2] win_reason_code vs win_reason_message; [3] code vs winner side")
    print(RE.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len().sort(
        ["win_reason_code", "winner_team_code"]))
    S["distinct_code_message_pairs"] = RE.select(["win_reason_code", "win_reason_message"]).unique().height
    S["codes_seen"] = sorted(RE["win_reason_code"].unique().to_list())
    S["rows_code_not_in_enum"] = RE.filter(~pl.col("win_reason_code").is_in(list(WIN_REASON))).height
    chk = RE.with_columns(
        pl.col("win_reason_code").replace_strict(WIN_REASON, default=None).alias("exp_msg"),
        pl.col("win_reason_code").replace_strict(EXPECTED_SIDE, default=None).alias("exp_side"))
    viol_msg = chk.filter(pl.col("exp_msg") != pl.col("win_reason_message"))
    viol_side = chk.filter(pl.col("exp_side") != pl.col("winner_team_code"))
    S["code_message_disagree_rows"] = viol_msg.height
    S["code_side_disagree_rows"] = viol_side.height
    print(f"code/message disagreements vs canonical enum: {viol_msg.height}")
    print(f"code/side disagreements vs canonical side: {viol_side.height}")
    if viol_side.height:
        print(viol_side.group_by(["win_reason_code", "winner_team_code", "round"]).len().sort(
            "len", descending=True).head(40))
    for c in [1, 7, 8, 9, 11, 12, 13, 17, 18]:
        S[f"count_code_{c}"] = RE.filter(pl.col("win_reason_code") == c).height
    S["share_code_8_or_9"] = round((S["count_code_8"] + S["count_code_9"]) / RE.height, 6)
    print(f"code 8={S['count_code_8']} code 9={S['count_code_9']} share 8+9={S['share_code_8_or_9']}; "
          f"code 1={S['count_code_1']} code 7={S['count_code_7']} code 12={S['count_code_12']}")

    # ---- 4. bomb events vs code
    print("\n[4] bomb_state flags per round vs win_reason_code")
    BS = load(base, "bomb_state", ["match_key", "round", "event_type"]).filter(
        pl.col("event_type").is_in(BOMB_EVS))
    print("bomb_state event counts (planted/defused/exploded):")
    print(BS.group_by("event_type").len().sort("event_type"))
    flags = BS.group_by(["match_key", "round"]).agg(
        (pl.col("event_type") == "bomb_planted").sum().alias("n_plant"),
        (pl.col("event_type") == "bomb_defused").sum().alias("n_defuse"),
        (pl.col("event_type") == "bomb_exploded").sum().alias("n_explode"))
    S["bomb_round_keys"] = flags.height
    S["bomb_rounds_multi_defuse_or_explode"] = flags.filter(
        (pl.col("n_defuse") > 1) | (pl.col("n_explode") > 1)).height
    S["bomb_rounds_defuse_and_explode"] = flags.filter(
        (pl.col("n_defuse") > 0) & (pl.col("n_explode") > 0)).height
    rb = RE.join(flags, on=["match_key", "round"], how="left").with_columns(
        pl.col("n_plant").fill_null(0), pl.col("n_defuse").fill_null(0), pl.col("n_explode").fill_null(0))
    xt = rb.with_columns(
        pl.when(pl.col("n_defuse") > 0).then(pl.lit("defused"))
        .when(pl.col("n_explode") > 0).then(pl.lit("exploded"))
        .when(pl.col("n_plant") > 0).then(pl.lit("planted_no_end"))
        .otherwise(pl.lit("no_bomb")).alias("bomb_outcome"))
    ctab = xt.group_by(["bomb_outcome", "win_reason_code", "winner_team_code"]).len().sort(
        ["bomb_outcome", "win_reason_code", "winner_team_code"])
    print(ctab)
    S["bomb_outcome_x_code"] = [list(r) for r in ctab.rows()]
    bomb_end = xt.filter(pl.col("bomb_outcome").is_in(["defused", "exploded"]))
    S["defused_rounds_total"] = xt.filter(pl.col("bomb_outcome") == "defused").height
    S["exploded_rounds_total"] = xt.filter(pl.col("bomb_outcome") == "exploded").height
    S["defused_rounds_coded_8_or_9"] = xt.filter(
        (pl.col("bomb_outcome") == "defused") & pl.col("win_reason_code").is_in([8, 9])).height
    S["exploded_rounds_coded_8_or_9"] = xt.filter(
        (pl.col("bomb_outcome") == "exploded") & pl.col("win_reason_code").is_in([8, 9])).height
    S["defused_rounds_coded_7"] = xt.filter(
        (pl.col("bomb_outcome") == "defused") & (pl.col("win_reason_code") == 7)).height
    S["exploded_rounds_coded_1"] = xt.filter(
        (pl.col("bomb_outcome") == "exploded") & (pl.col("win_reason_code") == 1)).height
    S["bomb_side_mismatch_rounds"] = bomb_end.filter(
        pl.when(pl.col("bomb_outcome") == "defused").then(pl.lit(CT)).otherwise(pl.lit(T))
        != pl.col("winner_team_code")).height
    print(f"defused rounds={S['defused_rounds_total']} (coded 7: {S['defused_rounds_coded_7']}, "
          f"coded 8/9: {S['defused_rounds_coded_8_or_9']}); exploded rounds={S['exploded_rounds_total']} "
          f"(coded 1: {S['exploded_rounds_coded_1']}, coded 8/9: {S['exploded_rounds_coded_8_or_9']})")
    print(f"bomb side (defused->CT, exploded->T) != winner_team_code: {S['bomb_side_mismatch_rounds']}")
    print(f"rounds with >1 defuse or explode event: {S['bomb_rounds_multi_defuse_or_explode']}; "
          f"rounds with both defuse and explode: {S['bomb_rounds_defuse_and_explode']}")

    # ---- 5. death evidence for codes 8/9
    print("\n[5] death evidence for elimination codes (8 = all T dead; 9 = all CT dead)")
    PI = load(base, "player_info", ["match_key", "round", "player_id", "team_code"]).unique(
        ["match_key", "round", "player_id"])
    roster = (PI.filter(pl.col("team_code").is_in([T, CT]))
              .group_by(["match_key", "round", "team_code"]).agg(pl.col("player_id").n_unique().alias("n_roster")))
    PD = (load(base, "player_death", ["match_key", "round", "player_id"])
          .filter(pl.col("player_id").is_not_null())
          .unique(["match_key", "round", "player_id"]))
    dead = PD.join(PI.select(["match_key", "round", "player_id", "team_code"]),
                   on=["match_key", "round", "player_id"], how="left")
    S["death_victims_distinct"] = dead.height
    S["death_victims_unjoined_to_roster"] = dead.filter(pl.col("team_code").is_null()).height
    dead_side = (dead.filter(pl.col("team_code").is_in([T, CT]))
                 .group_by(["match_key", "round", "team_code"]).agg(pl.col("player_id").n_unique().alias("n_dead")))
    rost_w = roster.pivot(on="team_code", index=["match_key", "round"], values="n_roster").rename(
        {"2": "n_t_roster", "3": "n_ct_roster"})
    dead_w = dead_side.pivot(on="team_code", index=["match_key", "round"], values="n_dead").rename(
        {"2": "n_t_dead", "3": "n_ct_dead"})
    ev = (RE.select(["match_key", "round", "win_reason_code", "winner_team_code"])
          .join(rost_w, on=["match_key", "round"], how="left")
          .join(dead_w, on=["match_key", "round"], how="left"))
    for c in ["n_t_roster", "n_ct_roster", "n_t_dead", "n_ct_dead"]:
        if c not in ev.columns:
            ev = ev.with_columns(pl.lit(None, dtype=pl.Int64).alias(c))
        ev = ev.with_columns(pl.col(c).fill_null(0))
    ev = ev.join(xt.select(["match_key", "round", "bomb_outcome"]), on=["match_key", "round"], how="left")
    e8 = ev.filter(pl.col("win_reason_code") == 8)
    e9 = ev.filter(pl.col("win_reason_code") == 9)
    S["code8_rows"] = e8.height
    S["code8_T_not_all_dead"] = e8.filter(pl.col("n_t_dead") < pl.col("n_t_roster")).height
    S["code8_T_roster_not_5"] = e8.filter(pl.col("n_t_roster") != 5).height
    S["code9_rows"] = e9.height
    S["code9_CT_not_all_dead"] = e9.filter(pl.col("n_ct_dead") < pl.col("n_ct_roster")).height
    S["code9_CT_roster_not_5"] = e9.filter(pl.col("n_ct_roster") != 5).height
    print(f"code 8 rounds={S['code8_rows']}: T not all dead={S['code8_T_not_all_dead']}, "
          f"T roster != 5: {S['code8_T_roster_not_5']}")
    print(f"code 9 rounds={S['code9_rows']}: CT not all dead={S['code9_CT_not_all_dead']}, "
          f"CT roster != 5: {S['code9_CT_roster_not_5']}")
    print("code 8 by bomb outcome:")
    print(e8.group_by("bomb_outcome").len())
    print("code 9 by bomb outcome:")
    print(e9.group_by("bomb_outcome").len())
    print("code 8 with T alive, pattern (roster, dead):")
    print(e8.filter(pl.col("n_t_dead") < pl.col("n_t_roster")).group_by(["n_t_roster", "n_t_dead"]).len().sort(
        "len", descending=True).head(10))
    print("code 9 with CT alive, pattern (roster, dead):")
    print(e9.filter(pl.col("n_ct_dead") < pl.col("n_ct_roster")).group_by(["n_ct_roster", "n_ct_dead"]).len().sort(
        "len", descending=True).head(10))
    S["bomb_coded_8_with_T_all_dead"] = e8.filter(
        pl.col("bomb_outcome").is_in(["defused", "exploded"]) & (pl.col("n_t_dead") >= pl.col("n_t_roster"))).height
    S["bomb_coded_9_with_CT_all_dead"] = e9.filter(
        pl.col("bomb_outcome").is_in(["defused", "exploded"]) & (pl.col("n_ct_dead") >= pl.col("n_ct_roster"))).height

    # ---- 6. score evidence
    print("\n[6] score-based winner from round_state round_freeze_end deltas")
    RS = load(base, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(
        pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first")
    S["freeze_end_rows"] = RS.height
    sc = {}
    for m, r, t, c in RS.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
        sc[(m, r)] = (t, c)
    rr = RE.select(["match_key", "round", "winner_team_code", "win_reason_code"]).unique(
        ["match_key", "round"], keep="first")
    st = collections.Counter()
    rows_score = []
    for m, r, w, code in rr.iter_rows():
        a = sc.get((m, r))
        b = sc.get((m, r + 1))
        if a is None or b is None:
            st["no_next_freeze_end"] += 1
            continue
        dt_, dc_ = b[0] - a[0], b[1] - a[1]
        s_ns = T if (dt_, dc_) == (1, 0) else (CT if (dt_, dc_) == (0, 1) else None)
        s_sw = T if (dt_, dc_) == (0, 1) else (CT if (dt_, dc_) == (1, 0) else None)
        rows_score.append((m, r, w, code, s_ns, s_sw))
        st["scored"] += 1
        st["noswap_agree"] += int(s_ns == w)
        st["noswap_disagree"] += int(s_ns is not None and s_ns != w)
        st["noswap_unresolved"] += int(s_ns is None)
        st["swap_agree"] += int(s_sw == w)
        st["swap_disagree"] += int(s_sw is not None and s_sw != w)
    print(dict(st))
    S["score_check"] = dict(st)
    SC = pl.DataFrame(rows_score, schema=["match_key", "round", "winner", "code", "s_noswap", "s_swap"],
                      orient="row").with_columns(
        pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day"),
        pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
        .otherwise(pl.lit("other")).alias("rbin"))
    if era == "2026":
        su = load(base, "score_update", ["match_key", "round", "tick", "team_code", "old_score", "new_score"])
        su1 = (su.filter(pl.col("new_score") - pl.col("old_score") == 1)
               .group_by(["match_key", "round"])
               .agg(pl.col("team_code").n_unique().alias("n_teams"), pl.col("team_code").first().alias("su_side")))
        j = SC.join(su1, on=["match_key", "round"], how="left")
        withsu = j.filter(pl.col("su_side").is_not_null())
        S["su_vs_winner"] = {"rounds_with_su": withsu.height,
                             "agree": withsu.filter(pl.col("su_side") == pl.col("winner")).height,
                             "multi_side_rounds": j.filter(pl.col("n_teams") > 1).height,
                             "no_su_rounds": j.filter(pl.col("su_side").is_null()).height}
        print("score_update vs round_end winner (2026):", S["su_vs_winner"])
        print("score_update side vs noswap score-derived side agree:",
              j.filter(pl.col("su_side") == pl.col("s_noswap")).height, "of",
              j.filter(pl.col("su_side").is_not_null() & pl.col("s_noswap").is_not_null()).height)
    dn = SC.filter(pl.col("s_noswap").is_not_null()).with_columns((pl.col("s_noswap") != pl.col("winner")).alias("dis"))
    print("noswap score-derived != round_end winner, by round bin:")
    bins_ns = dn.group_by("rbin").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort("rbin")
    print(bins_ns)
    S["score_noswap_dis_by_rbin"] = bins_ns.rows()
    ds = SC.filter(pl.col("s_swap").is_not_null()).with_columns((pl.col("s_swap") != pl.col("winner")).alias("dis"))
    bins_sw = ds.group_by("rbin").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort("rbin")
    print("swap score-derived != round_end winner, by round bin:")
    print(bins_sw)
    S["score_swap_dis_by_rbin"] = bins_sw.rows()
    print("noswap score disagreement by win code:")
    print(dn.group_by(["code", "dis"]).len().sort(["code", "dis"]))
    S["score_noswap_dis_by_code"] = dn.group_by(["code", "dis"]).len().sort(["code", "dis"]).rows()

    # ---- 7. fingerprint dedupe and deduped stats
    print("\n[7] fingerprint dedupe of matches (identical round_end sequence)")
    fp = (RE.sort(["match_key", "round"])
          .with_columns(pl.concat_str([pl.col("round").cast(pl.Utf8), pl.col("tick").cast(pl.Utf8),
                                       pl.col("winner_team_code").cast(pl.Utf8),
                                       pl.col("win_reason_code").cast(pl.Utf8)], separator=":").alias("part"))
          .group_by("match_key").agg(pl.col("part").alias("parts"))
          .with_columns(pl.col("parts").list.join("|").hash().alias("fph")))
    grp = fp.group_by("fph").agg(pl.col("match_key").sort().alias("keys"), pl.len().alias("n"))
    dup_groups = grp.filter(pl.col("n") > 1)
    S["fingerprint_dup_groups"] = dup_groups.height
    S["fingerprint_dup_matches_dropped"] = int((dup_groups["n"] - 1).sum()) if dup_groups.height else 0
    keep_keys = set(grp["keys"].list.first().to_list())
    S["fingerprint_kept_matches"] = len(keep_keys)
    print(f"duplicate groups={dup_groups.height}; non-first copies={S['fingerprint_dup_matches_dropped']}; "
          f"kept matches={len(keep_keys)}")
    RE_k = RE.filter(pl.col("match_key").is_in(list(keep_keys)))
    S["dedup_round_rows"] = RE_k.height
    S["dedup_ct_share"] = round(RE_k.filter(pl.col("winner_team_code") == CT).height / RE_k.height, 6)
    S["dedup_code_8_or_9_share"] = round(RE_k.filter(pl.col("win_reason_code").is_in([8, 9])).height / RE_k.height, 6)
    print(f"deduped rows={RE_k.height}; ct_share={S['dedup_ct_share']}; code 8/9 share={S['dedup_code_8_or_9_share']}")
    xk = xt.filter(pl.col("match_key").is_in(list(keep_keys)))
    print("deduped: bomb outcome x code x winner:")
    dk = xk.group_by(["bomb_outcome", "win_reason_code", "winner_team_code"]).len().sort(
        ["bomb_outcome", "win_reason_code", "winner_team_code"])
    print(dk)
    S["dedup_bomb_outcome_x_code"] = [list(r) for r in dk.rows()]

    # ---- 8. legacy_code (2024 only)
    if era == "2024":
        print("\n[8] legacy_code (2024)")
        lc = RE.filter(pl.col("legacy_code") != 0)
        S["legacy_nonzero_rows"] = lc.height
        print(lc.group_by(["legacy_code", "win_reason_code", "winner_team_code"]).len().sort("legacy_code"))
        print("legacy rows by round (top 10):")
        print(lc.group_by("round").len().sort("len", descending=True).head(10))
        print("legacy rows by day:")
        print(lc.group_by("day").len().sort("day"))
        lcb = xt.filter(pl.col("legacy_code") != 0).group_by("bomb_outcome").len()
        print("legacy rows by bomb outcome:")
        print(lcb)
        S["legacy_by_bomb_outcome"] = lcb.rows()
        mis_rows = xt.filter(
            ((pl.col("bomb_outcome") == "defused") & (pl.col("winner_team_code") != CT))
            | ((pl.col("bomb_outcome") == "exploded") & (pl.col("winner_team_code") != T)))
        S["all_bomb_mismatch_rows"] = mis_rows.height
        S["all_bomb_mismatch_rows_with_legacy"] = mis_rows.filter(pl.col("legacy_code") != 0).height
        r13_mis = mis_rows.filter(pl.col("round") == 13)
        S["r13_bomb_mismatch_rows"] = r13_mis.height
        S["r13_bomb_mismatch_with_legacy"] = r13_mis.filter(pl.col("legacy_code") != 0).height
        print(f"bomb-side mismatch rows (all rounds)={mis_rows.height}, with legacy_code!=0: "
              f"{S['all_bomb_mismatch_rows_with_legacy']}; round 13 mismatch={r13_mis.height}, "
              f"with legacy!=0: {S['r13_bomb_mismatch_with_legacy']}")

    # ---- bomb-side mismatch by round bin and by day
    bm = xt.filter(pl.col("bomb_outcome").is_in(["defused", "exploded"])).with_columns(
        pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rbin"),
        (((pl.col("bomb_outcome") == "defused") & (pl.col("winner_team_code") != CT))
         | ((pl.col("bomb_outcome") == "exploded") & (pl.col("winner_team_code") != T))).alias("mis"))
    bres = bm.group_by("rbin").agg(pl.len().alias("n"), pl.col("mis").sum().alias("mis")).sort("rbin")
    print("\nbomb-side mismatch by round bin:")
    print(bres)
    S["bomb_side_mismatch_by_rbin"] = bres.rows()
    bday = bm.group_by("day").agg(pl.len().alias("n"), pl.col("mis").sum().alias("mis")).sort("day")
    print("bomb-side mismatch by day:")
    print(bday)
    S["bomb_side_mismatch_by_day"] = bday.rows()
    return S


if __name__ == "__main__":
    out = {}
    for era, base in ERAS.items():
        out[era] = audit_era(era, base)
    log = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad") / "A6_label_audit_summary.json"
    log.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("\nSUMMARY written:", log)
