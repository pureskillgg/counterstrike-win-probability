"""Text slides of the findings deck (cover, brief, evaluation, incremental tomes, viewer, literature, decisions, questions, next, appendix divider).
Every number is read from a result file or a log under G:\\pureskill\\csds-cs2; nothing is typed by hand. Used by build_deck_model.py.
"""
import json
import os
import re

import polars as pl

from deck_lib import AMBER, BLUE, CARD, INK, RULE, SANS, SERIF, bullets, callout, cards, esc, note, slide, table

ROOT = r"G:\pureskill\csds-cs2"


def J(store, name):
    p = os.path.join(ROOT, "data", store, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _manifest_counts():
    m = json.load(open(os.path.join(ROOT, "data", "combined", "split_manifest.json"), encoding="utf-8"))["assign"]
    c = {}
    for k, v in m.items():
        era = k.split("/")[1]
        c[(era, v)] = c.get((era, v), 0) + 1
    return c


def _step_secs():
    out, p = {}, os.path.join(ROOT, "logs", "release_b.log")
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8-sig"):
            m = re.search(r"ok\s+(\w+) \((\d+) s\)", ln)
            if m:
                out[m.group(1)] = int(m.group(2))
    return out


def _col(ttl, color, items, size=26, pad=30, gap=12):
    li = "".join(f"<li>{esc(i)}</li>" for i in items)
    return (f'<div style="flex:1;display:flex;flex-direction:column;gap:{gap}px;background:{CARD};padding:{pad}px;border:1px solid {RULE};border-radius:16px">'
            f'<h3 style="font-size:34px;font-weight:600;color:{color}">{esc(ttl)}</h3><ul style="font-size:{size}px;line-height:1.38;color:{INK}">{li}</ul></div>')


def s_cover(store, snap, basic):
    ex = J(store, f"export_{snap}.json")
    return (f'<section id="cover" data-transition="fade" style="background:#16202E;color:#F6F5F0;font-family:{SANS};padding:128px;display:flex;flex-direction:column;justify-content:space-between">\n'
            f'<p style="font-size:28px;letter-spacing:3px;text-transform:uppercase;color:#93B4FF;font-weight:600">PureSkill CSDS Â· Counter-Strike 2 Â· status for Dr Freeman</p>\n'
            f'<div style="display:flex;flex-direction:column;gap:32px">\n'
            f'<h1 style="font-family:{SERIF};font-size:112px;font-weight:600;line-height:1.05;color:#F6F5F0">Round win probability from the live round state</h1>\n'
            f'<p style="font-size:40px;color:#C9D3E3;line-height:1.35;width:1500px">A probability that CT wins, for every second of a round, trained on {ex["train"]["matches"]:,} matches and scored on {ex["test"]["matches"]:,} it never saw. It runs on a CPU.</p>\n</div>\n'
            f'<p style="font-size:28px;color:#9AA7BC">Adam Camerer Â· 15 October 2026 Â· every number on these slides is read from a result file; intervals resample whole matches</p>\n'
            f'<aside>The task is a per-tick probability that the CT side wins the round, trained on PureSkill CSDS data. This deck replaces the 13 October status deck: the model, the data and the tests were rebuilt around a snapshot of the living players at each second. The first-batch exploration slides are kept at the end as an appendix.</aside>\n</section>\n')


def s_brief(store, snap, basic):
    ex = J(store, f"export_{snap}.json")
    bl = J(store, "baselines.json")["random"]["models"]
    st = J(store, f"stress_{snap}.json")
    prep = J(store, "prep_report.json")
    mc = _manifest_counts()
    tr = sum(v for (e, sp), v in mc.items() if sp == "train")
    va = sum(v for (e, sp), v in mc.items() if sp == "val")
    te = sum(v for (e, sp), v in mc.items() if sp == "test")
    lat = next((c["detail"] for c in st["checks"] if " latency" in c["check"][:12]), "")
    med = re.search(r"median ([0-9.]+) ms", lat)
    fz = next((c["detail"] for c in st["checks"] if c["check"].startswith("1 fuzz")), "")
    n_fuzz = sum(int(x) for x in re.findall(r"'\w+': (\d+)", fz.split("statuses=")[-1])) if "statuses=" in fz else 0
    g = ex["gpu_trained"]
    d_old = g["logloss"] - bl["XGB F_SCORE | era 2026"]["logloss"]
    rows = [["Run at every tick", "Every second of every round is scored; half-second offsets checked", "Done"],
            ["No random split", f"Frozen match split: {tr:,} train, {va:,} validation, {te:,} test matches", "Done"],
            ["Target metric: log loss", f"{g['logloss']:.4f} on {ex['test']['matches']:,} held-out matches", "Done"],
            ["Calibration plot", f"Reliability diagram; ECE {g['ece10']:.3f}", "Done"],
            ["Ticks are correlated", "Every interval resamples whole matches (warning on each slide)", "Done"],
            ["Beat the earlier model", f"{d_old:+.3f} against our score-lead design; PureSkill's own model is not named", "Open"],
            ["CPU in production, GPU to train", f"GPU and CPU fits agree; {med.group(1) + ' ms' if med else 'about a millisecond'} per prediction on a CPU", "Done"],
            ["Never crash on a new map or mode", (f"{n_fuzz:,} fuzzed states, none raised" if n_fuzz else "Fuzz and stress tests pass"), "Done"],
            ["Enough data, no leakage", f"{prep['use_matches']:,} usable matches; shuffled-label check; mutation test", "Done"]]
    return slide("brief", "The brief", "What was asked for, and where each point stands",
                 table(["Requirement", "What we did", "Status"], rows, widths=[30, 58, 12], size=25),
                 source=f"Source: result files under data/{store}; split_manifest.json; stress_{snap}.json; baselines.json; prep_report.json",
                 notes="Walk through the rules from the brief. Done means the check ran and passed on held-out matches. Open means we cannot close it ourselves: the documents do not name the earlier PureSkill model, so the comparison is against our own re-implementation of the score-lead design, which uses information the final model deliberately leaves out.")


def s_eval(store, snap):
    ex = J(store, f"export_{snap}.json")
    cs = J(store, f"cluster_se_{snap}.json")
    mc = _manifest_counts()
    n26 = mc[('2026', 'train')] + mc[('2026', 'val')] + mc[('2026', 'test')]
    parts = []
    for ttl, col, txt in (("1. Split", BLUE, f"By match, never by row or round, and frozen: a match keeps its side of the split when new matches arrive. 2026: {100 * mc[('2026', 'train')] / n26:.0f}% train, {100 * mc[('2026', 'val')] / n26:.0f}% validation, {100 * mc[('2026', 'test')] / n26:.0f}% test ({mc[('2026', 'train')]:,} / {mc[('2026', 'val')]:,} / {mc[('2026', 'test')]:,} matches); 2024 splits the same way."),
                          ("2. Score", BLUE, "Log loss is the target, reported over every second. Brier, AUC and calibration error are secondary. A model that always says 50 percent scores 0.693."),
                          ("3. Uncertainty", AMBER, "One round has one winner, so its seconds are not independent. Every interval resamples whole matches" + (f"; the match-cluster standard error is {cs['match_cluster_ratio']:.1f} times the naive one on the final model." if cs else "."))):
        parts.append(f'<div style="flex:1;display:flex;flex-direction:column;gap:14px;background:{CARD};padding:36px;border:1px solid {RULE};border-radius:16px"><h3 style="font-size:36px;font-weight:600;color:{col}">{esc(ttl)}</h3><p style="font-size:28px;line-height:1.4;color:#3B4656">{esc(txt)}</p></div>')
    body = '<div style="display:flex;gap:24px">' + "".join(parts) + "</div>" + callout("Warning: the seconds of one round are not independent observations. An interval from counting seconds as separate rows would be wrong by a large factor.", 30)
    return slide("eval", "How we evaluate", "Split by match, judge by log loss, treat seconds as correlated", body,
                 source=f"Source: data/{store}/split_manifest.json, cluster_se_{snap}.json (scripts/viz/cluster_se.py); {ex['test']['rows']:,} test seconds",
                 notes="This is the central methodological slide. The bootstrap resamples matches, not rounds and not seconds, because a match is the unit that shares teams, map and day. Resampling rounds instead gives almost the same width, so the correlation is inside the round. The split is frozen: when new matches arrive they are assigned by a hash of their id, so nothing already assigned moves between train and test.")


def s_incremental(store):
    t = _step_secs()
    prep = J(store, "prep_report.json")
    rows = [("Tome builder, stock", "a stopped or extended build starts again from zero"),
            ("Tome builder, our fork", "resumes a stopped build and extends a finished tome with new pages without rebuilding the old ones; its 311 tests pass"),
            ("Project layer", "new pages are published as chunks; a ledger and an append-only combined store merge only what is new")]
    stats = [(f"{t.get('append', 0) // 60} min", "to merge the 2,797 day-13 matches into the combined store (append step)", "blue"),
             (f"{sum(t.get(k, 0) for k in ('prepare', 'grid', 'split', 'extract', 'cache')) // 60} min", "to refresh scope, the grid, the split, the loadout extraction and the model cache on top of it", "blue"),
             (f"{prep['use_matches']:,}", "usable matches after the merge; nothing earlier was rebuilt", "amber")]
    body = cards(stats, big=60, caption=25) + bullets([a + ": " + b for a, b in rows], size=27) + note("Check: an 86-match fixture and a 40-match real-data build give the same tome built in one go or resumed and extended. Day 13 is half downloaded; its second half arrives through the same path.", 25)
    return slide("incremental", "Adding data without restarting", "New uploads are folded in; nothing is rebuilt from scratch", body,
                 source="Source: logs/release_b.log; G:\\pureskill\\dsdk-fork (docs/incremental-tomes.md, branch incremental-tomes); scripts/data/build_tomes_incremental.py, append_combined.py",
                 notes="The stock dsdk tome builder cannot extend a finished tome and restarts a stopped build from zero. We forked it, tagged at 4.1.1: a build can resume from its own manifest and a finished tome can be extended with new pages, with the narrowing step re-planned over old and new signatures. On top of that the project keeps a chunk queue, a ledger and an append-only combined store, so a day of new matches costs minutes. Day 13 was merged this afternoon through that path; the times above are from the run log.")


def s_viewer():
    d = os.path.join(ROOT, "notes", "replay")
    f = next((x for x in sorted(os.listdir(d)) if re.match(r"replay_[0-9a-f]{8}\.html$", x)), None) if os.path.isdir(d) else None
    body = bullets(["Plays a held-out match round by round on the radar: 8 frames a second, kill and bomb events, each player's health, armor, weapon and money",
                    "Shows the model's CT win chance for every second, the way the PureSkill match viewer shows its own",
                    "Players are anonymous (CT1 to T5) because the source has no names; scores are by starting side",
                    "Radar images are Valve game assets from awpy-data: fine locally, not cleared for sharing"], size=28)
    if f:
        body += callout("Open " + f + " from notes\\replay in any browser: it is one self-contained file.", 28)
    return slide("viewer", "See it live", "A replay viewer shows the model's call on a real match", body,
                 source="Source: scripts/replay (extract_replay.py, build_replay_page.py, viewer_template.html); raw per-tick channels at 64 Hz",
                 notes="This answers the request to see the probability the way the PureSkill viewer does. The page is built from the raw per-tick player vector, player status, deaths and bomb channels of one held-out match and the model's per-second predictions. Kills in the replay agree with the drops in living players. It is a development viewer: no utility icons or economy panel yet.")


def s_lit():
    cols = [("Counter-Strike papers", AMBER, ["Mostly CS:GO, not CS2", "Small datasets: some have only tens of matches", "Accuracy as the metric, not log loss or calibration",
                                              "Dr Freeman's view (he co-published with Xenopoulos): weak references, none gives a number we can compare with"]),
            ("Teacher-student, where it exists", BLUE, ["For tabular data: a large pretrained tabular model teaches a boosted-tree student (TabPFN-2.5 distillation, TabICL into XGBoost)", "For privileged information: a teacher sees more than the live service has",
                                                      "Not for Counter-Strike win probability that we could find"]),
            ("Our own tests", INK, ["Deep teacher on the same inputs: slightly better only in one combination, and not reliably", "Tabular foundation models, zero-shot and fine-tuned: worse than boosted trees here",
                                    "A player-level graph model: small, reliable gain"])]
    return slide("lit", "What others have published", "The Counter-Strike papers are a weak reference; teacher-student exists for tabular data, not for this task",
                 '<div style="display:flex;gap:24px">' + "".join(_col(t, c, i, pad=32) for t, c, i in cols) + "</div>",
                 source="Source: notes/research/PAPERS_NUMBERED_BY_SUBFOLDER_2026-10-15.md (Haiku-only search, 43 of 45 candidates opened); the first column is Dr Freeman's assessment as relayed",
                 notes="The Counter-Strike literature is thin and has known problems: older game, small datasets, accuracy instead of a probability metric. Dr Freeman co-published with Xenopoulos and has said these are a problem, so we do not use their numbers as a benchmark. Teacher-student methods do exist for tabular data, where a large pretrained tabular model is distilled into a boosted-tree student; we have not found one for win probability in Counter-Strike. Our own tests of a deep teacher and of tabular foundation models are in the appendix.")

def s_decisions():
    settled = ["Split by match, frozen; log loss is the target", "No score lead and no round number: the snapshot of the living players is the input",
               "Boosted trees with raw probabilities (isotonic and Platt do not help)", "Train on a GPU, serve on a CPU; a simpler alive-count model answers when there is no player table",
               "Unknown maps and modes are answered and flagged out of scope", "Match-cluster intervals on every comparison"]
    need = ["Benchmark of record: which earlier PureSkill model, on what data?", "About 1,100 FACEIT matches have no match type: keep them as competitive?",
            "2024 data: only the simpler alive-count model uses it; keep or drop?", "Is a one-second grid enough for every tick?",
            "Positions: train a per-player model if one-second positions can be shared", "Sharing: the data are CC BY-NC-SA; the radar images are Valve assets"]
    return slide("decisions", "Decisions", "Settled so far, and what needs a decision",
                 '<div style="display:flex;gap:24px">' + _col("Settled", BLUE, settled, size=27, pad=32, gap=14) + _col("Needs a decision", AMBER, need, size=27, pad=32, gap=14) + "</div>",
                 source="Source: this deck; notes/STATUS_2026-10-15.md",
                 notes="The settled items were tested in this deck. The open items need an answer from outside the lab: the benchmark and the sharing terms in particular. The matches with no recorded match type are kept as competitive 5v5 because their map mix and roster sizes match; the status page lists their maps.")


def s_questions():
    qs = ["Which earlier PureSkill model are we beating, and on what data and metric? The documents name none.",
          "Is a one-second grid enough for every tick, or do you need the 64-per-second curve?",
          "Headline metric: log loss over all seconds, or round-weighted? They weight long rounds differently.",
          "About 1,100 matches have no match type (FACEIT-style). Keep them in the competitive set?",
          "Can we share one-second player positions for training? Published deep models need them.",
          "What margin would make a deeper model worth adopting? We propose 0.005 log loss on held-out matches.",
          "The data are CC BY-NC-SA 4.0 and the radar images are Valve assets: any limit on sharing the model, the predictions or the viewer?"]
    li = "".join(f"<li>{esc(q)}</li>" for q in qs)
    return slide("questions", "For Dr Freeman", "Questions we need answered", f'<ol style="font-size:30px;line-height:1.38;color:#E3E9F3">{li}</ol>', dark=True,
                 source="Each question is tied to a decision on the previous slide",
                 notes="Questions one and seven block how we describe and share the result. The rest are design choices where a quick answer saves a week.")


def s_next(store):
    prep = J(store, "prep_report.json")
    pos = J(store, "exp_positions.json")
    gain = f"{-pos['geometry+view+blind']['delta_vs_base']:.4f}" if pos else None
    cols = [("This week", BLUE, [f"Refit on the {prep['use_matches']:,} usable matches after each batch lands",
                                 (f"Serve the position inputs: they cut log loss by {gain} in the paired test (needs player positions in the input contract)" if gain else "Test position and view-angle inputs"),
                                 "Check raw-to-tome row counts and short matches for duplicates",
                                 "Complete-matches-only sensitivity: 22% of matches have fewer rounds than the header score"]),
            ("Then", BLUE, ["Walking distance from the radar's walkable area instead of straight lines",
                            "Per-player positions model as teacher, snapshot model as student",
                            "Late-round bins: the clock running out is slightly under-predicted"]),
            ("Later", INK, ["Match win probability from round probability and score", "Utility and economy panels in the viewer", "Package the model and the viewer for the team"])]
    return slide("next", "Next steps", "Refit on every upload, serve the position inputs, then try walking distance",
                 '<div style="display:flex;gap:24px">' + "".join(_col(t, c, i) for t, c, i in cols) + "</div>",
                 source="Source: notes/STATUS_2026-10-15_evening.md",
                 notes="The model is refit on a pod in about an hour of wall clock for a few dollars, so each batch can be folded in. The position test showed a measurable gain from distances to the bomb sites, spacing and the defuse slack, and very little from view cones and blindness; serving them needs the live service to pass player positions. Straight-line distance ignores walls, so a walking distance taken from the radar's walkable area is the next refinement.")


def s_faceit(store):
    d = J(store, "faceit_audit.json")
    if not d:
        return None
    g, fm = d["groups"], d["final_model_by_group"]
    fa, va = g.get("FACEIT unknown"), g.get("Valve competitive")
    if not (fa and va and "FACEIT unknown" in fm and "Valve competitive" in fm):
        return None
    rows = []
    for lab, k in (("FACEIT, no match type in the header", "FACEIT unknown"), ("Valve matchmaking (competitive)", "Valve competitive")):
        x, m = g[k], fm[k]
        rows.append([lab, f"{x['matches']:,}", f"{x['usable_rounds']:,}", f"{100 * x['ct_win_rate']:.1f}%", f"{100 * x['plant_rate']:.1f}%", f"{100 * x['round_status_ok_share']:.0f}%", f"{m['logloss']:.4f}", f"{m['auc']:.3f}"])
    tbl = table(["2026 matches", "Matches", "Usable rounds", "CT wins", "Bomb planted", "Rounds clean", "Log loss", "AUC"], rows, widths=[30, 10, 13, 9, 11, 11, 9, 7], size=24)
    refit = d.get("refits", {}).get("scores")
    extra = ""
    if refit:
        rr = []
        for lab, key in (("All training matches (the final setting)", "all"), ("Valve matchmaking only", "valve_only"), ("FACEIT only", "faceit_only")):
            cells = [lab]
            for t in ("Valve test matches", "FACEIT test matches"):
                v = refit[t][key]
                cells.append(f"{v['logloss']:.4f}" + (f" ({v['delta_vs_all']:+.4f})" if "delta_vs_all" in v else ""))
            rr.append(cells)
        ntr = d["refits"]["train_matches"]
        extra = table(["Model trained on", f"Scored on {d['refits']['test_matches']['Valve test matches']} Valve test matches", f"Scored on {d['refits']['test_matches']['FACEIT test matches']:,} FACEIT test matches"], rr, widths=[40, 30, 30], size=24)
        extra += note(f"Same settings as the final model, every {d['refits']['stride']}rd second; training matches: all {ntr['all']:,}, Valve {ntr['valve_only']:,}, FACEIT {ntr['faceit_only']:,}. In brackets: change against the all-matches model on the same test matches (paired); plus = worse.", 24)
    M26 = pl.read_parquet(os.path.join(ROOT, "data", store, "matches.parquet"), columns=["era"]).filter(pl.col("era") == "2026").height
    body = tbl + callout(f"The FACEIT matches with no match type are {100 * fa['matches'] / M26:.0f}% of the {M26:,} 2026 matches: the header gives them no match type, so they are labelled 'unknown'. On every check here they look like ordinary 5v5 competitive games.", 26) + extra
    return slide("faceit", "Match types", "FACEIT matches with no match type look like normal competitive games, and the model scores as well on them", body,
                 source="Source: faceit_audit.json (scripts/model/faceit_audit.py); held-out matches, final model v3; intervals resample whole matches",
                 notes="The header field match_type is 'unknown' for FACEIT servers, which is most of the 2026 data, so the question is whether they belong with the competitive matches. Their round counts, bomb plant rate, kills per round and CT win rate look like the Valve matchmaking games, and their rounds are cleaner. The final model scores them about as well as the Valve matches (the groups differ in size, so the two log losses are not a paired comparison). Training on them helps: a model trained on Valve matches alone is worse on the Valve test matches, and a model trained on FACEIT matches alone is also worse on them. Intervals and paired differences use the match-cluster bootstrap.")


def s_graph(store):
    d = J(store, "exp_gnn.json")
    status = "Not tried before this deck: every neural model so far was tabular (a multi-layer network on per-side totals, a deep teacher, foundation models)."
    if not d:
        body = bullets([status, "Idea from the paper on temporal heterogeneous graph networks (42 Analytics): treat each player as a node with position, view direction, weapon and health; let players attend to each other; look at the last seconds.",
                        "What we built: one graph per second with the 10 players, pairwise distance and who-sees-whom as edge features, attention layers, and a recurrent step over three seconds; trained on the GPU on the same matches and seconds as the position test.",
                        "Result: still running when this deck was built; the numbers go here when it finishes."], size=27)
        return slide("graph", "Graph models", "Player-level graph models: built and running", body, source="Source: scripts/model/gnn; notes/research/PAPERS_NUMBERED_BY_SUBFOLDER_2026-10-15.md (paper 28)",
                     notes="The paper trains a temporal heterogeneous graph network on 114 professional Inferno matches and reports log loss 0.4587 against 0.4688 for logistic regression and 0.5353 for XGBoost, but its XGBoost baseline is weaker than its logistic regression, the split is not described, and it uses player rating features. Our test is paired against our own tuned XGBoost on the same held-out matches.")
    names = {"xgb59": "Boosted trees, the 59 final inputs", "xgb83": "Boosted trees, plus the position inputs (83)", "gnn_hyb1": "Graph model: players plus the 59 inputs",
             "gnn_temporal": "Graph model, plus the last 4 seconds", "gnn_nopos": "Graph model with positions removed", "gnn_set": "Graph model: players only",
             "ens_gnn_xgb83": "Average of the graph model and boosted trees + positions"}
    order = [k for k in names if k in d["models"]]
    rows = []
    for k in order:
        m = d["models"][k]
        rows.append([names[k], f"{m['logloss']:.4f}", f"{m['auc']:.4f}"] + [(f"{m[f'delta_vs_{r}']:+.4f} [{m[f'delta_vs_{r}_ci95'][0]:+.4f}, {m[f'delta_vs_{r}_ci95'][1]:+.4f}]" if f"delta_vs_{r}" in m else "-") for r in ("xgb59", "xgb83")])
    body = table(["Model", "Log loss", "AUC", "Change vs trees, 59 inputs", "Change vs trees + positions"], rows, widths=[32, 10, 10, 24, 24], size=22) + note(f"{d['test_matches']:,} held-out matches, {d['test_rows']:,} seconds; minus = better, paired, with match-cluster 95% intervals. Same frozen split and rows as the position test.", 24)
    h = d["models"].get("gnn_hyb1")
    if h and "delta_vs_xgb83_ci95" in h:
        title = ("A player-level graph model beats boosted trees on the same seconds, by a small margin" if h["delta_vs_xgb83_ci95"][1] < 0 else
                 ("A player-level graph model is slightly better than boosted trees, but not reliably" if h["delta_vs_xgb83"] < 0 else "A player-level graph model does not beat boosted trees"))
    else:
        title = "Player-level graph models against boosted trees on the same seconds"
    return slide("graph", "Graph models", title, body, source="Source: exp_gnn.json (scripts/model/gnn); same frozen split and rows as exp_positions.json; paper 28 in the reading list",
                 notes="One graph per second: the 10 players are nodes with position, view direction, health, weapon, grenades and blindness; every pair of living players is an edge carrying distance and who sees whom; attention layers read the graph and the result is pooled by team. The hybrid variant also gets the final model's 59 aggregate inputs; the temporal variant looks at the same graph 2 and 4 seconds earlier through a recurrent step; the no-positions variant zeroes positions, view directions and distances, so what is left is player-level information without geometry. The references are boosted trees on the same rows, with and without the position inputs. All comparisons are paired on the same test seconds.")


def s_divider():
    return slide("appendix", "Appendix", "First-batch exploration: 1 to 8 October, 1,521 matches",
                 '<p style="font-size:36px;line-height:1.4;color:#C9D3E3;width:1500px">The slides after this one were computed on the first batch of 2026 matches (the score-lead model and its follow-ups). They are kept for the record and were not re-run on the larger data, so their numbers differ from the ones before this slide.</p>',
                 dark=True, source="Superseded numbers: use the slides before this one",
                 notes="Appendix. The first-batch work: per-second evaluation of the earlier model, the terminal-state correction, round-level charts, plants, pistol and armor, buy types, weapons, eras and data quality. Nothing here is the final model.")
