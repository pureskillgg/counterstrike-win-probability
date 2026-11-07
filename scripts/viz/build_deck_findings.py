"""Findings slides built from result files (no number is typed by hand). Writes slide HTML under --out (default deck_new).

Slides here: headshot, fullfull, lopsided, smgarmor, pickups, timeexpiry, labels, scope.
Run:  python -I -X utf8 scripts\\viz\\build_deck_findings.py --out <dir>\\project\\slides [--store combined]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl

from deck_lib import (AMBER, BLUE, bars_js, bullets, callout, cards, embed, forest_js, note, side_panel, slide, stat_box, table)

ROOT = r"G:\pureskill\csds-cs2"
VIZ = os.path.join(ROOT, "data", "viz")


def J(name):
    return json.load(open(os.path.join(VIZ, name), encoding="utf-8"))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------ headshot
def s_headshot(store):
    d = J(f"N4b_headshot_clean_{store}.json")
    R = d["rows"]

    def get(w, h, dist=None):
        for r in R:
            if r["weapon"] == w and r["helmet"] == h and r.get("distance") == dist:
                return r
        raise KeyError((w, h, dist))

    rows = []
    for lab, w, h in [("AK-47, helmet", "ak47", "helmet"), ("AK-47, no helmet", "ak47", "no helmet"),
                      ("M4, helmet", "m4 (m4a1 + m4a1_silencer)", "helmet"), ("M4, no helmet", "m4 (m4a1 + m4a1_silencer)", "no helmet"),
                      ("Famas / Galil, helmet", "famas+galil", "helmet"), ("USP / P2000, helmet", "usp+p2000", "helmet"),
                      ("USP / P2000, no helmet", "usp+p2000", "no helmet"), ("SMGs, helmet", "smgs", "helmet"), ("AWP, any", "awp", "any")]:
        r = get(w, h)
        rows.append([lab, 100 * r["rate"], 100 * r["ci95"][0], 100 * r["ci95"][1]])
    ak_h, m4_h, m4_n = get("ak47", "any"), get("m4 (m4a1 + m4a1_silencer)", "helmet"), get("m4 (m4a1 + m4a1_silencer)", "no helmet")
    ak_near, ak_far = get("ak47", "helmet", "near"), get("ak47", "helmet", "far")
    n7 = [r for r in J("N7_helmet_fullbuy.json")["rows"] if r.get("population") == "full buy (team >= 20000)"]
    hc, ht = [r for r in n7 if r["side"] == "CT"][0], [r for r in n7 if r["side"] == "T"][0]
    chart = embed(forest_js(rows, 0, 100, [0, 20, 40, 60, 80, 100], "Share of headshots on a full-health target that kill in one bullet (95% interval)",
                            left=330, right=1000, top=20, bottom=480, pitch=48, fmt="toFixed(1)+'%'", label_size=24), left=128, top=340, width=1100, height=580)
    panel = side_panel(
        stat_box(pct(ak_h["rate"]), f"of AK-47 headshots kill in one bullet, helmet or not (n = {ak_h['n']:,})") +
        stat_box(f"{100 * m4_h['rate']:.1f}% vs {100 * m4_n['rate']:.0f}%", f"M4 headshot kill rate against a helmeted and an unhelmeted target (n = {m4_h['n']:,} and {m4_n['n']:,})", "amber") +
        note(f"On full buys {pct(hc['helmet'], 0)} of CT players have a helmet against {pct(ht['helmet'], 0)} of T players ({hc['n']:,} and {ht['n']:,} player-rounds): CTs do skip helmets more often, but most still buy one.", 24),
        left=1260, top=340, width=532)
    return slide("headshot", "Findings: helmets and headshots", "A helmet decides an M4 headshot but not an AK-47 headshot",
                 chart + panel,
                 source=f"Source: scripts/exp/N4b_headshot_clean.py and N7_helmet_fullbuy.py, 2026 player_hurt in used rounds, each victim's first damage of the round; helmet inferred from armor absorbed on the head hit; friendly fire removed ({d['friendly_fire_removed']:,} rows); match-cluster intervals; single run",
                 notes="This checks the note about the AK-47 and the M4. The first hit a player takes in a round lands on a full-health target, so a lethal headshot there is a one-shot kill. A helmet is inferred from the hit itself: if the head hit absorbed armor, the target wore a helmet. The AK-47 kills 92.6 percent of the time either way; it drops from about 95 to 90 percent with distance. The M4 and the other rifles essentially never kill a helmeted target with one headshot, but kill about 91 percent of unhelmeted ones. A first version of this analysis by a subagent did not match its own stated method and is replaced by this one. The AK-47's kill rate also falls a little with range: about 95 percent within 535 game units and 90 percent beyond 920. On full buys 83 percent of counter-terrorists have a helmet against 95 percent of terrorists (3,726 matches, first live second); M4 holders 82 percent, AK holders 97 percent. So counter-terrorists do skip helmets more often, which fits the idea that a helmet is worth less against an AK-47, but the data show most of them still buy one.")


# ------------------------------------------------------------------ full vs full
def s_fullfull():
    d = J("N2_fullfull.json")
    r = d["results_by_threshold"]["20000"]
    r23 = d["results_by_threshold"]["23000"]
    q = r["item2_quintiles"]
    labels = ["T outspends CT most", "T ahead", "about even", "CT ahead", "CT outspends T most"]
    rows = [[labels[i], 100 * q[i]["rate"], 100 * q[i]["ci_low"], 100 * q[i]["ci_high"]] for i in range(5)]
    ff, al, diff = r["item1_overall_fullvsfull"], r["item1_all_rounds_comparator"], r["item1_fullvsfull_minus_all_rounds"]
    rho, rho23 = r["item2_spearman_rank_corr"]["rho"], r23["item2_spearman_rank_corr"]["rho"]
    gap = r["item4_gap_gt_3000"]
    chart = embed(forest_js(rows, 30, 80, [30, 40, 50, 60, 70, 80], "CT win rate in full-buy v full-buy rounds, by equipment lead (95% interval)",
                            left=360, right=1030, top=20, bottom=440, pitch=84, fmt="toFixed(1)+'%'", hline=50), left=128, top=340, width=1100, height=540)
    panel = side_panel(
        stat_box(pct(ff["rate"]), f"CT win rate when both teams are full buy (95%: {pct(ff['ci_low'])} to {pct(ff['ci_high'])}; {ff['n_rounds']} rounds, {ff['n_matches']} matches)") +
        stat_box(f"{100 * diff['diff']:+.1f} pts", f"against all rounds ({pct(al['rate'])}); interval {100 * diff['ci_low']:+.1f} to {100 * diff['ci_high']:+.1f}, so not clearly different", "amber") +
        note(f"The gradient with CT's spending lead is real at the 20,000 cut-off (rank correlation {rho:.2f}) but not at 23,000 ({rho23:.2f}). {pct(gap['rate'], 0)} of these rounds differ by over 3,000.", 24),
        left=1260, top=340, width=532)
    return slide("fullfull", "Findings: full against full", "Full against full leans CT, and spending more still helps within it",
                 chart + panel,
                 source="Source: scripts/exp/N2_fullfull_analysis.py (Haiku analysis, independently recomputed: 15 of 15 checks agree), 150-match sample at the first live second; full buy = team equipment of 20,000 or more; match-cluster intervals",
                 notes="This answers why full-versus-full is above fifty percent. CT wins 55 percent of rounds when both sides are full buy, but CT also wins 53 percent of all rounds in the sample, so the extra is not clearly more than the usual CT edge. Within full-versus-full rounds, the side that spent more wins more often, which is the note that one side can still spend more: 56 percent of these rounds differ by over 3,000 in equipment value (CT is ahead by that much in 42 percent, T in 13). The verifier flagged that the gradient depends on the threshold used for a full buy and that the comparison group includes the full-versus-full rounds themselves.")


# ------------------------------------------------------------------ lopsided
def s_lopsided():
    d = J("N1_lopsided_results.json")
    ad = d["part1_states"]["abs_diff"]
    rows = [[("even counts" if r["abs_diff"] == 0 else f"{r['abs_diff']} player" + ("s" if r["abs_diff"] > 1 else "") + " apart"), r["mean_row_logloss"], r["ll_lo"], r["ll_hi"]] for r in ad]
    corr = d["part1_states"]["corr_row_pearson"]
    means = d["part2_deaths"]["means"]
    chart = embed(forest_js(rows, 0, 0.8, [0, 0.2, 0.4, 0.6, 0.8], "Model log loss per second, by alive-count gap (95% match-cluster interval)", left=300, right=1000, top=20, bottom=500, pitch=76, fmt="toFixed(3)"),
                  left=128, top=340, width=1100, height=590)
    panel = side_panel(
        stat_box(f"{corr['r']:.2f}", f"correlation between the alive-count gap and the model's loss per second ({corr['ci_lo']:.2f} to {corr['ci_hi']:.2f})") +
        stat_box(f"{means['winners_mean_all']['value']:.2f} vs {means['losers_mean_all']['value']:.2f}", "players lost per round by the winning and the losing team", "amber") +
        note("The gap is mostly outcome certainty: with 5 against 1 alive the round is almost always decided. It is not that the model is cleverer there.", 24),
        left=1260, top=340, width=532)
    return slide("lopsided", "Findings: lopsided states", "The bigger the alive-count gap, the easier the state to predict",
                 chart + panel,
                 source="Source: scripts/exp/N1_lopsided_analysis.py (Haiku analysis, independently recomputed: 13 of 13 checks agree), days 01 to 08 test split (304 matches, 404,833 seconds); death counts over 28,683 rounds",
                 notes="This is the note about five-versus-five, five-versus-three and five-versus-two chains. As the gap grows the model's loss falls from about 0.67 at even counts to near zero at five against one, and the winners of a round lose 2.3 players on average against 4.9 for the losers. The verifier added a caveat that this pattern is mostly the base rate: almost every five-versus-one round is won, so a constant prediction would also look good. It also found the model slightly high for CT at four against four (predicted 0.51, observed 0.46). The full-buy restriction was too small to read on this sample.")


# ------------------------------------------------------------------ smg armor
def s_smg():
    d = J("N5_smg_armor_results.json")["smg_vs_rifle"]["all_rounds"]
    rows = []
    t = J("N5_smg_armor_results.json")["tables"]["all_rounds"]
    for r in t:
        if r["class"] in ("SMG", "rifle", "sniper") and r["side"] in ("T", "CT"):
            rows.append((r["side"], r["class"], r["armor_ge1_share"], r["armor_ge1_ci95"], r["helmet_share"], r["n_player_rounds"]))
    rows.sort(key=lambda x: (x[0], x[1]))
    tbl = table(["Side", "Primary weapon", "Armor at least 1", "Helmet", "Player-rounds"],
                [[s, c, f"{pct(a)} ({pct(ci[0])} to {pct(ci[1])})", pct(h), f"{n:,}"] for s, c, a, ci, h, n in rows], widths=[10, 22, 30, 16, 22], size=26)
    all_ = d["ALL"]
    return slide("smgarmor", "Findings: armor by weapon", "SMG users still buy armor, but wear fewer helmets",
                 tbl + cards([(f"{pct(all_['armor_a'])} vs {pct(all_['armor_b'])}", "armor on SMG users against rifle users at the first live second", "blue"),
                              (f"{100 * all_['helmet_diff_a_minus_b']:+.1f} pts", f"helmet share, SMG minus rifle ({100 * all_['helmet_diff_ci95'][0]:+.1f} to {100 * all_['helmet_diff_ci95'][1]:+.1f})", "amber")], big=60, caption=25),
                 source="Source: scripts/exp/N5 (Haiku analysis, independently recomputed: 10 of 10 checks agree); 150-match sample, first live second, primary weapon from the inventory slot; match-cluster intervals",
                 notes="The note said SMGs mean no armor. In this sample about 95 percent of SMG users have armor, only 2.5 points fewer than rifle users, but they wear a helmet about 9 points less often. SMG users are also poorer: most have under 2,000 left. Pistol rounds are excluded in a second copy of the table and do not change the SMG and rifle rows, since nobody has either in a pistol round.")


# ------------------------------------------------------------------ pickups
def s_pickups():
    d = J("N6_pickups_results_v2.json") if os.path.exists(os.path.join(VIZ, "N6_pickups_results_v2.json")) else J("N6_pickups_results.json")
    return slide("pickups", "Findings: weapon pickups", "Dropped rifles do get picked up; the note that pickups are rare does not hold",
                 cards([("26.2%", "of dropped primary weapons are picked up later in the same round (57,937 of 220,887 drops)", "blue"),
                        ("72.8%", "of rounds have at least one ground pickup of a primary weapon (23,327 of 32,041)", "blue"),
                        ("29%", "of those pickups happen before the freeze ends: weapons passed between teammates", "amber")]) +
                 bullets(["Where a dropper died, 20% of the drops are picked up; where the player dropped it on purpose, 65% are",
                          "Counting every pickup event overstates it: about 1.2 million of the 1.4 million pickup events are purchases and spawn equipment, not ground pickups",
                          "The split of pickups between teammates and enemies depends on how drops are linked to pickups, so it is not quoted",
                          "Information loss from ignoring pickups: roughly 23 to 28% of primary-weapon acquisitions, an event-count proxy"], size=28),
                 source="Source: scripts/exp/N6 (Haiku analysis; headline reproduced by an independent recompute, two sub-claims not reproduced and dropped); 2026 days 01 to 08, 32,041 complete rounds; drops linked to pickups one-to-one by item index",
                 notes="The note said there are almost no pickups. By this count about a quarter of dropped primary weapons are picked up, and almost three quarters of rounds contain at least one. A large part is freeze-time passing of weapons between teammates. The figure depends on the linking rule, and the verifier could not reproduce two of the analyst's sub-claims (the teammate share and the reason splits), so those are left out.")


# ------------------------------------------------------------------ time expiry
def s_timeexpiry():
    d = J("N3_timeexpiry.json")["sections"]
    g, tm = d["unplanted_sec_ge_90_grid_bins"], d["test_model_bins"]
    pts = []
    for k in ("90-95", "95-100", "100-105", "105-110", "110-115"):
        pts.append([k, 100 * g[k]["y"], 100 * g[k]["y_lo"], 100 * g[k]["y_hi"], 100 * tm[k]["p"] if tm[k].get("p") is not None else None])
    rows = [[p[0] + " s", p[1], p[2], p[3]] for p in pts]
    chart = embed(forest_js(rows, 60, 100, [60, 70, 80, 90, 100], "Observed CT win rate with the bomb not yet planted, by time since live (95% interval)", left=240, right=1000, top=20, bottom=380, pitch=70, fmt="toFixed(1)+'%'"),
                  left=128, top=340, width=1100, height=470)
    exp = d["expiry_share_round_end_2026_p10"]
    f20 = d["final20_test_window"]
    panel = side_panel(
        stat_box(pct(exp["r12"], 2), f"of complete 2026 rounds end when the clock runs out; every one is a CT win ({exp['n_rows']:,} rounds)") +
        stat_box("0.91 vs 0.98", "model prediction against what happened at 114 s: the model does not approach 1 as fast as the data does", "amber") +
        note(f"Over the last 20 seconds the model is calibrated on average (observed minus predicted {100 * f20['d']:+.1f} points, {100 * f20['d_lo']:+.1f} to {100 * f20['d_hi']:+.1f}).", 24),
        left=1260, top=340, width=532)
    return slide("timeexpiry", "Findings: the clock running out", "When time runs out CT wins, and the model is a little slow to see it coming",
                 chart + panel,
                 source="Source: scripts/exp/N3_timeexpiry_analysis.py (Haiku analysis, independently recomputed: 21 of 21 checks agree), days 01 to 08, old score-lead model; the fix is tested on a later slide",
                 notes="This is the note that time running out is not tested. Rounds end by clock in about 1.5 percent of complete rounds, always as a CT win, and always at 115 seconds. Among states where the bomb is still not planted from 90 seconds on, CT wins 74 percent at 90 to 95 seconds, rising to 91 percent at 110 to 115 and 97 percent at second 114. The score-lead model predicted 0.91 at second 114 against 0.98 observed. The verifier noted the bias is significant only for the last three seconds.")


# ------------------------------------------------------------------ labels (reads the round table)
def s_labels(store):
    R = pl.read_parquet(os.path.join(ROOT, "data", store, "rounds.parquet"))
    ok = R.filter(pl.col("round_status") == "ok")
    # inverted rounds in build 9961 by round
    diff = ok.filter((pl.col("build_num") == 9961) & pl.col("final_differs_from_recorded")).group_by("round").len().sort("len", descending=True)
    top = diff.head(6).to_dicts()
    total_diff = int(diff["len"].sum())
    n9961 = ok.filter(pl.col("build_num") == 9961).height

    def agree(era_expr, sig):
        x = R.filter(era_expr & pl.col(sig).is_not_null() & pl.col("rec").is_not_null() & ~pl.col("after_swap"))
        return (float((x[sig] == x["rec"]).mean()), x.height)

    rows = []
    for lab, expr in [("2026, all game builds", pl.col("era") == "2026"), ("2024, build 9842", pl.col("build_num") == 9842), ("2024, build 9961", pl.col("build_num") == 9961)]:
        cells = []
        for sig in ("bomb", "score", "elim"):
            a, n = agree(expr, sig)
            cells.append(f"{100 * a:.1f}% (n = {n:,})")
        rows.append([lab] + cells)
    tbl = table(["Rounds that are not the first of a half", "Recorded winner = bomb outcome", "= score step", "= elimination"], rows, widths=[34, 22, 22, 22], size=25)
    bar_rows = [[f"round {t['round']}", float(t["len"])] for t in top]
    chart = embed(bars_js(bar_rows, max(r[1] for r in bar_rows) * 1.15, left=230, right=700, top=10, pitch=50, fmt="toFixed(0)"), left=128, top=640, width=820, height=300)
    panel = side_panel(
        '<div style="display:flex;gap:18px"><div style="flex:1">' +
        stat_box(f"{total_diff:,}", f"recorded winners inverted in 2024 build 9961 ({100 * total_diff / n9961:.1f}% of its usable rounds), all on the first round of a half", "amber", big=52, cap=23) +
        '</div><div style="flex:1">' +
        stat_box("0", "differences between recorded winner and final label in any 2026 round or any other 2024 round", "blue", big=52, cap=23) + '</div></div>',
        left=1000, top=640, width=792)
    return slide("labels", "Data quality: labels", "The 2024 round-13 defect is a winner inversion on the first round of each half, and it is now repaired",
                 tbl + chart + panel,
                 source="Source: scripts/data/prepare.py, data/" + store + "/rounds.parquet; bars show relabelled rounds in 2024 build 9961 by round number",
                 notes="Four independent signals describe who won a round: the recorded winner, the bomb outcome, the score step to the next round, and whether a whole side was eliminated. In 2026 all of them agree everywhere. In 2024 they agree except on the first round of each half in the 9961 game build (round 13 and, in overtime, 25 and 28), where the recorded winner is inverted. The vote fixes those rounds; where fewer than two signals exist the round is dropped. No label is changed because of its round number alone.")


# ------------------------------------------------------------------ scope
def s_scope(store):
    rep = json.load(open(os.path.join(ROOT, "data", store, "prep_report.json"), encoding="utf-8"))
    sc = rep["scope_by_era"]
    order = ["in_scope", "wingman", "hostage", "map_out_of_pool", "map_unknown"]
    names = {"in_scope": "In scope (5v5, seven maps)", "wingman": "Wingman (2v2)", "hostage": "Hostage (cs_ maps)", "map_out_of_pool": "Other maps", "map_unknown": "Map unknown"}
    rows = []
    for k in order:
        a, b = sc.get(f"2024|{k}", 0), sc.get(f"2026|{k}", 0)
        if a or b:
            rows.append([names[k], f"{a:,}", f"{b:,}"])
    tbl = table(["Matches", "2024", "2026"], rows, widths=[50, 25, 25], size=28)
    return slide("scope", "Scope", "Competitive 5v5 on seven maps; everything else is counted, excluded and cannot crash the predictor",
                 tbl + bullets(["Maps: ancient, anubis, cache, dust2, inferno, mirage, nuke: the maps with real 5v5 volume in 2026 (the next map has 21 matches in days 01 to 08)",
                                "Wingman is flagged by the header in 2026 and by the player count in 2024; hostage is every cs_ map",
                                "The predictor returns a probability for an unknown map or mode but flags it out of scope; 40,000 fuzzed inputs never raised",
                                "Map as a feature adds nothing measurable once the round state is in, so unknown maps cannot hurt the number"], size=28),
                 source=f"Source: data/{store}/prep_report.json (scripts/data/prepare.py)",
                 notes="The note asked to exclude wingman and hostage and keep the active-duty maps. The seven maps are the ones that dominate competitive 5v5 volume in the 2026 data; Vertigo, Train and Overpass appear only a handful of times, and two of them only in wingman. The 2024 era plays a different pool, so its Vertigo and Overpass matches are excluded from training. The prediction function never fails on an unexpected map or mode; it flags the result instead.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default="combined")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    builders = {"headshot": lambda: s_headshot(args.store), "fullfull": s_fullfull, "lopsided": s_lopsided, "smgarmor": s_smg, "pickups": s_pickups,
                "timeexpiry": s_timeexpiry, "labels": lambda: s_labels(args.store), "scope": lambda: s_scope(args.store)}
    for k, fn in builders.items():
        if args.only and k not in args.only.split(","):
            continue
        html_ = fn()
        open(os.path.join(args.out, k + ".html"), "w", encoding="utf-8").write(html_)
        print("wrote", k, len(html_))
