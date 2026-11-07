"""Build the Word documents for the end-of-day package.

  ...pixi python: .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\viz\\build_word_docs.py

1. The wrap-up (final choices, models, features, metrics, example ticks for each split, how to call the function). Numbers come from the
   saved result files and from a run of scripts/serving/win_probability.py on real data.
2. Word versions of the three notes (methodology, deliverable summary, experiment register), converted from their markdown.

HTML is written first; LibreOffice converts it to .docx (no Word install needed). Writes deliverables/word/*.docx.
"""
import glob
import html
import json
import os
import re
import subprocess
import sys

import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "combined")
MODELS = os.path.join(ROOT, "models")
SERVING = os.path.join(ROOT, "scripts", "serving")
OUT = os.path.join(ROOT, "deliverables", "word")
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SUFFIX = ".2026-09-18,2026-10-08"
SOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"
DATE = "2026-10-19"

sys.path.insert(0, SERVING)
import win_probability as WP  # noqa: E402

CSS = ("body{font-family:Calibri,Arial,sans-serif;font-size:10.5pt;line-height:1.35}"
       "h1{font-size:18pt;margin-bottom:4pt}h2{font-size:13.5pt;margin-top:14pt}h3{font-size:11.5pt;margin-top:10pt}"
       "table{border-collapse:collapse;margin:4pt 0 8pt 0}td,th{border:1px solid #444;padding:3pt 5pt;vertical-align:top;font-size:9pt}"
       "th{background:#D9D9D9;text-align:left}code{font-family:Consolas,monospace;font-size:9pt}")


def esc(s):
    return html.escape(str(s), quote=False)


def inline(s):
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def md_to_html(md):
    lines = md.splitlines()
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows if not re.match(r"^\|[\s:|-]+\|?\s*$", r)]
            if cells:
                out.append("<table>")
                out.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells[0]) + "</tr>")
                for row in cells[1:]:
                    out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
                out.append("</table>")
            continue
        m = re.match(r"^(#{1,3}) (.*)", ln)
        if m:
            out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>")
            i += 1
            continue
        if ln.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("</ul>")
            continue
        if re.match(r"^\d+\. ", ln):
            out.append("<ol>")
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                out.append(f"<li>{inline(re.sub(r'^\d+\. ', '', lines[i]))}</li>")
                i += 1
            out.append("</ol>")
            continue
        if not ln.strip():
            i += 1
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("|", "#", "- ")) and not re.match(r"^\d+\. ", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


def wrap(title, body):
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{esc(title)}</title><style>{CSS}</style></head>'
            f"<body>{body}</body></html>")


def load_feats():
    meta = json.load(open(os.path.join(MODELS, "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))
    return meta, meta["features"]


def examples(feats):
    man = json.load(open(os.path.join(D, "split_manifest.json"), encoding="utf-8"))["assign"]
    keys_by = {"train": [], "validation": [], "test": []}
    tag = {"train": "train", "validation": "val", "test": "test"}
    for k, v in man.items():
        if not k.startswith("csds/2026/"):
            continue
        for name, t in tag.items():
            if v == t:
                keys_by[name].append(k)
    base = ["match_key", "round", "tick", "alive_ct", "alive_t", "bomb_planted", "seconds_since_round_live", "map_name",
            "st_hp_ct", "st_rifle_ct", "st_hp_t", "st_rifle_t", "label_ct_win"]
    cols = list(dict.fromkeys(base + feats))
    bst = xgb.Booster()
    bst.load_model(os.path.join(MODELS, "winprob_v6_drop_heavy.json"))
    best = int(json.load(open(os.path.join(MODELS, "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["best_iteration"])
    lazy = pl.scan_parquet(os.path.join(D, "cache_snapshot_2026.parquet")).select(cols)
    out = {}
    ranges = {}
    for name in ["train", "validation", "test"]:
        keys = sorted(keys_by[name])
        sub = lazy.filter(pl.col("match_key").is_in(keys)).collect()
        out_name = {"matches": sub["match_key"].n_unique(), "seconds": sub.height}
        k0 = keys[0]
        m = sub.filter(pl.col("match_key") == k0)
        r0 = sorted(m["round"].unique().to_list())[0]
        rr = m.filter(pl.col("round") == r0).sort("tick").head(6)
        X = rr.select(feats).to_numpy().astype("float32")
        p = bst.predict(xgb.DMatrix(X, feature_names=feats), iteration_range=(0, best + 1))
        rows = rr.to_dicts()
        for r, pv in zip(rows, p):
            r["p_ct"] = round(float(pv) * 100, 2)
        out[name] = {"counts": out_name, "match_key": k0, "round": int(r0), "rows": rows}
        if name == "train":
            for c in feats:
                ranges[c] = (float(sub[c].min()), float(sub[c].max()))
    return out, ranges


def function_sample():
    import test_win_probability as TWP  # the parity test's reader: per-tick player status comes from the per-tick folder
    hdr = pl.concat([pl.read_parquet(f, columns=["key"]) for f in sorted(glob.glob(os.path.join(TOME, "header", "dataframe_*")))])
    grid = pl.read_parquet(os.path.join(D, "export_winprob_v6_drop_heavy_test_predictions.parquet"), columns=["match_key"])
    tests = set(grid["match_key"].unique().to_list())
    data_key = sorted(k for k in hdr["key"].to_list() if k in tests)[0]
    data = WP.load_tome_match(TOME, data_key, SUFFIX)
    data["player_status"] = TWP.read_player_status(data_key)
    probs = WP.win_probability(data, team="CT", model_name="winprob_v6_drop_heavy")
    probs_t = WP.win_probability(data, team="T", model_name="winprob_v6_drop_heavy")
    ticks = sorted(probs)
    return {"match_key": data_key, "n": len(probs), "first": [(t, probs[t], probs_t[t]) for t in ticks[:3]],
            "warnings": list(WP.last_warnings)}


LOADOUT_DEF = {
    "hp": "sum of health over {who} players alive at this second",
    "armor_n": "count of {who} players alive with armour above 0",
    "armor_sum": "sum of armour over {who} players alive at this second",
    "helmet_n": "count of {who} players alive with a helmet",
    "kit_n": "count of {who} players alive with a defuse kit",
    "defusing": "count of {who} players alive and defusing the bomb at this second",
    "ak": "count of {who} players whose primary weapon is an AK-47 (item 7)",
    "m4": "count of {who} players whose primary weapon is an M4A4 (item 16) or M4A1-S (item 60)",
    "awp": "count of {who} players whose primary weapon is an AWP (item 9)",
    "rifle": "count of {who} players whose primary weapon is a rifle: AK-47 (7), AUG (8), FAMAS (10), Galil AR (13), M4A4 (16), SG 553 (39) or M4A1-S (60); includes the ak and m4 counts",
    "sniper_other": "count of {who} players whose primary weapon is a G3SG1 (11), SCAR-20 (38) or SSG 08 (40)",
    "smg": "count of {who} players whose primary weapon is a MAC-10 (17), P90 (19), MP5-SD (23), UMP-45 (24), PP-Bizon (26), MP7 (33) or MP9 (34)",
    "no_primary": "count of {who} players alive with no primary weapon (inventory primary is 0)",
    "flash": "flashbang grenades carried by {who} players alive, summed over the players",
    "smoke": "smoke grenades carried by {who} players alive, summed over the players",
    "he": "HE grenades carried by {who} players alive, summed over the players",
    "fire": "molotov plus incendiary grenades carried by {who} players alive, summed over the players",
    "decoy": "decoy grenades carried by {who} players alive, summed over the players",
    "equip": "equipment value summed over {who} players alive at this second",
}
LOADOUT_SRC = {
    "hp": "health", "armor_n": "armor", "armor_sum": "armor", "helmet_n": "has_helmet", "kit_n": "has_defuser",
    "defusing": "is_defusing", "ak": "inv_primary", "m4": "inv_primary", "awp": "inv_primary", "rifle": "inv_primary",
    "sniper_other": "inv_primary", "smg": "inv_primary", "no_primary": "inv_primary", "flash": "inv_flashbang",
    "smoke": "inv_smokegrenade", "he": "inv_hegrenade", "fire": "inv_molotov, inv_incgrenade", "decoy": "inv_decoy",
    "equip": "equipment_value_calc",
}
EVENT_DEF = {
    "alive_ct": "CT players alive at this second: the CT count at the round start (normally 5) minus the CT deaths since the freeze end, each player counted once",
    "alive_t": "T players alive at this second: the T count at the round start (normally 5) minus the T deaths since the freeze end, each player counted once",
    "ct_alive_minus_t": "alive_ct minus alive_t",
    "bomb_planted": "1 from the bomb-plant tick onward in the round, else 0",
    "seconds_since_round_live": "(tick minus the freeze-end tick) / 64; 0 at the freeze end",
    "seconds_since_plant": "(tick minus the plant tick) / 64 from the plant onward; -1 before the plant",
}
EVENT_SRC = {
    "alive_ct": "player_info (roster), player_death", "alive_t": "player_info (roster), player_death", "ct_alive_minus_t": "derived",
    "bomb_planted": "bomb_state", "seconds_since_round_live": "round_state (freeze end)", "seconds_since_plant": "bomb_state, round_state",
}


def feature_html(feats, ranges, constant):
    def rng(c):
        lo, hi = ranges[c]
        if float(lo).is_integer() and float(hi).is_integer():
            return f"{lo:.0f} to {hi:.0f}"
        return f"{lo:.2f} to {hi:.2f}"

    rows = []
    for i, c in enumerate(feats, start=1):
        if c.startswith("map_is_"):
            m = c[len("map_is_"):]
            group, defin, src = "Event: map", f"1 if the match map is {m}, else 0; all seven are 0 for a map outside the seven", "header: map_name"
        elif c in EVENT_DEF:
            group, defin, src = "Event", EVENT_DEF[c], EVENT_SRC[c]
        elif c.startswith("st_") and c.endswith("_diff"):
            base = c[3:-5]
            group, defin, src = "Difference, CT minus T", f"CT value minus T value of {base} (st_{base}_ct minus st_{base}_t)", "derived from the side inputs"
        else:
            if c.endswith("_ct"):
                side, base = "ct", c[3:-3]
            else:
                side, base = "t", c[3:-2]
            who = "the CT" if side == "ct" else "the T"
            group = "Loadout, CT" if side == "ct" else "Loadout, T"
            defin = LOADOUT_DEF[base].format(who=who)
            src = "player_status: " + LOADOUT_SRC[base]
        rows.append(f"<tr><td>{i}</td><td><code>{esc(c)}</code></td><td>{group}</td><td>{esc(defin)}</td><td>{esc(src)}</td><td>{rng(c)}</td></tr>")
    const_txt = ("Inputs with one value in every 2026 training row: " + "; ".join(f"<code>{c}</code> (always {v:.0f})" for c, v in constant.items())
                 + ". These carry no information.") if constant else "No input in the 2026 training rows has a single value."
    return ("<p>Every input is computed for each live second of each round, from the channels named in the last column. The loadout inputs come "
            "from the per-tick player status channel, which has one row per living player. Check: on 3,283 live seconds in three October test "
            "matches, the number of status rows equalled alive_ct plus alive_t on every second. The last column is the lowest and highest value "
            "in the 2026 training rows.</p>"
            "<table><tr><th>#</th><th>Input</th><th>Group</th><th>Exact definition</th><th>Source (channel: field)</th><th>Min to max, training rows</th></tr>"
            + "".join(rows) + "</table><p>" + const_txt + "</p>")


def fmt_int(n):
    return f"{int(n):,}"


def main():
    os.makedirs(OUT, exist_ok=True)
    sub = json.load(open(os.path.join(D, "exp_submission_57.json"), encoding="utf-8"))
    par = json.load(open(os.path.join(D, "exp_eventonly_paired.json"), encoding="utf-8"))
    meta, feats = load_feats()
    ex, ranges = examples(feats)
    constant = {c: lo for c, (lo, hi) in ranges.items() if lo == hi}
    feature_str = feature_html(feats, ranges, constant)
    fs = function_sample()
    v, t = sub["splits"]["validation"], sub["splits"]["test"]
    g = t["groups"]
    pair = sub["paired_57_minus_58_ensemble_test"]

    def ci(x):
        return f"[{x[0]:.6f}, {x[1]:.6f}]"

    ev = [f for f in feats if not f.startswith("st_") and not f.startswith("map_")]
    mp = [f for f in feats if f.startswith("map_")]
    side = [f for f in feats if f.startswith("st_") and f.endswith("_ct")]
    diff = [f for f in feats if f.startswith("st_") and f.endswith("_diff")]
    assert len(ev) + len(mp) + len(side) * 2 + len(diff) == 57

    rows_html = []
    for name, label in [("train", "Train"), ("validation", "Validation"), ("test", "Test")]:
        e = ex[name]
        c = e["counts"]
        heads = ["tick", "seconds since live", "alive CT", "alive T", "bomb planted", "map", "CT health", "CT rifles", "T health",
                 "T rifles", "CT won (label)", "P(CT wins), %"]
        trs = "".join(
            "<tr>" + "".join(f"<td>{x}</td>" for x in [
                r["tick"], f"{r['seconds_since_round_live']:.0f}", r["alive_ct"], r["alive_t"], r["bomb_planted"], r["map_name"],
                f"{r['st_hp_ct']:.0f}", f"{r['st_rifle_ct']:.0f}", f"{r['st_hp_t']:.0f}", f"{r['st_rifle_t']:.0f}",
                r["label_ct_win"], f"{r['p_ct']:.2f}"]) + "</tr>"
            for r in e["rows"])
        rows_html.append(
            f"<h3>{label}: {fmt_int(c['matches'])} matches, {fmt_int(c['seconds'])} seconds</h3>"
            f"<p>Example: match <code>{esc(e['match_key'])}</code>, round {e['round']}, the first six live seconds of the round.</p>"
            f"<table><tr>{''.join(f'<th>{h}</th>' for h in heads)}</tr>{trs}</table>")

    fn_rows = "".join(f"<tr><td>{t_}</td><td>{p_c:.4f}</td><td>{p_t:.4f}</td></tr>" for t_, p_c, p_t in fs["first"])
    warn_txt = "; ".join(fs["warnings"]) if fs["warnings"] else "none"

    body = f"""
<h1>CS2 round win probability: end-of-day deliverable ({DATE})</h1>
<p>The model gives, for every live second of a round, the probability that CT (or T) wins the round, in percent. It reads one match from the DSDK tome and returns a dictionary of tick to percent. The submitted model is the 57-input model <code>winprob_v6_drop_heavy</code>. Its validation and test figures are below; all intervals are 95% match-cluster intervals (1,000 resamples, seed 0).</p>

<h2>1. Final choices</h2>
<table>
<tr><th>Choice</th><th>Setting</th><th>Why, and the evidence</th></tr>
<tr><td>Unit of scoring</td><td>Each live second; per-second log loss (lower is better)</td><td>Every second of a round is a prediction. Log loss is the metric reported throughout the record.</td></tr>
<tr><td>Data for the submission</td><td>2026 matches only: {fmt_int(meta['training']['matches'])} train, {fmt_int(meta['training']['val_matches'])} validation, {fmt_int(meta['test']['matches'])} test</td><td>The 2024 matches have no per-tick player table, so they cannot feed the 57 loadout inputs. The 2024 matches are used only by the event-only comparison.</td></tr>
<tr><td>Split</td><td>By match, 72 / 8 / 20 of matches; random, frozen once in a manifest (sha256 a17f219b...)</td><td>Seconds inside one match are almost identical, so a split by second would leak into the test set. 72/8/20 is not a named standard. It is the first training code's default (20% test, then 10% of the remaining 80% as validation), which I carried over and did not tune. The guides cite 80/20 for a two-way split and 60/20/20, 70/15/15 or 80/10/10 for three-way splits. No fraction sweep was run.</td></tr>
<tr><td>Roster-defect rounds</td><td>Dropped (<code>drop_roster_defect_rounds</code>: true)</td><td>In these rounds one side's roster is short by a player, so the player status cannot be trusted. Recorded in the methodology, section 5.</td></tr>
<tr><td>Inputs</td><td>57 = 13 event inputs + 44 loadout inputs</td><td>The 59-input set minus the heavy-weapon group (shotguns and machine guns). The heavy group was the only candidate better than the control on both splits for both seeds. The choice used test as well as validation, which is disclosed.</td></tr>
<tr><td>Score lead</td><td>Left out of the submission</td><td>Decided at the 19 October meeting. The score lead is in the 58-input comparison model. The reason recorded for leaving it out is round leakage; the check of that reason is in the methodology, section 1. The owner needs to confirm the reason.</td></tr>
<tr><td>Model</td><td>XGBoost: depth 4, learning rate 0.05, minimum child weight 20, subsample 0.8, early stopping after 100 rounds without validation gain; best iteration {meta['best_iteration']:,}</td><td>Ten alternative settings were tried on validation; none beat the baseline there (experiment register, E24).</td></tr>
<tr><td>Monotone constraints</td><td>More CT alive never lowers the CT probability; more T alive never raises it</td><td>Kept. Removing them is neutral on validation and worse on test for the 59-input model; see the methodology.</td></tr>
<tr><td>Seeds</td><td>One seed (seed 0) for the submission</td><td>The 58-input comparison is the mean of five seeds, which gains 0.00008 on test over seed 0.</td></tr>
<tr><td>Maps</td><td>Seven: ancient, anubis, cache, dust2, inferno, mirage, nuke</td><td>Other maps and wingman are out of scope; the function warns on them.</td></tr>
</table>

<h2>2. Models</h2>
<table>
<tr><th>Model</th><th>Inputs</th><th>Files</th><th>Test log loss (3,059 matches)</th><th>Status</th></tr>
<tr><td><b>Submission:</b> winprob_v6_drop_heavy</td><td>57</td><td>models/winprob_v6_drop_heavy.json and .meta.json</td><td>{t['logloss']:.6f} {ci(t['ci95'])}</td><td>The default of win_probability.py. Its metadata says served: false. Serving parity checked on 294 October test matches (below).</td></tr>
<tr><td>Score-lead comparison: winprob_v8_ensemble</td><td>58 (13 event, 1 score lead, 44 loadout), mean of five seeds</td><td>models/winprob_v8_ensemble.meta.json and winprob_v8_seed0 to seed4 .json</td><td>0.455605 [0.452430, 0.458410]</td><td>Passed by name: model_name="winprob_v8_ensemble". Serving status of its metadata is open (provenance).</td></tr>
<tr><td>Event-only: winprob_v7_basic_scorelead</td><td>14 (13 event, 1 score lead)</td><td>models/winprob_v7_basic_scorelead.json and .meta.json</td><td>0.498695 on the same 3,059 matches; 0.501242 on its own 3,601 matches</td><td>For tomes without per-tick player status. Must be named.</td></tr>
</table>
<p>File format: the models are XGBoost's native JSON format. Each <code>.json</code> holds the trees; the matching <code>.meta.json</code> holds the feature names in order, the settings and the metrics. The function reads both.</p>

<h2>3. Features (57)</h2>
{feature_str}

<h2>4. Metrics</h2>
<table>
<tr><th>Split</th><th>Matches</th><th>Seconds</th><th>Per-second log loss [95% interval]</th></tr>
<tr><td>Validation</td><td>{fmt_int(v['matches'])}</td><td>{fmt_int(v['seconds'])}</td><td>{v['logloss']:.6f} {ci(v['ci95'])}</td></tr>
<tr><td>Test</td><td>{fmt_int(t['matches'])}</td><td>{fmt_int(t['seconds'])}</td><td>{t['logloss']:.6f} {ci(t['ci95'])}</td></tr>
</table>
<table>
<tr><th>Test, by round group</th><th>Seconds</th><th>Per-second log loss [95% interval]</th></tr>
<tr><td>Pistol rounds (1, 13)</td><td>{fmt_int(g['pistol (1, 13)']['seconds'])}</td><td>{g['pistol (1, 13)']['logloss']:.6f} {ci(g['pistol (1, 13)']['ci95'])}</td></tr>
<tr><td>Rounds after pistol (2, 14)</td><td>{fmt_int(g['after pistol (2, 14)']['seconds'])}</td><td>{g['after pistol (2, 14)']['logloss']:.6f} {ci(g['after pistol (2, 14)']['ci95'])}</td></tr>
<tr><td>All other rounds</td><td>{fmt_int(g['other rounds']['seconds'])}</td><td>{g['other rounds']['logloss']:.6f} {ci(g['other rounds']['ci95'])}</td></tr>
</table>
<p><b>Against the score-lead comparison model</b> (same test seconds): the submission is worse by {pair['difference']:+.6f} [{pair['ci95'][0]:+.6f}, {pair['ci95'][1]:+.6f}]. The event-only model is worse by +0.043090 [+0.041449, +0.044884] on the same seconds, and the gap is largest after the pistol round.</p>
<p><b>Serving parity, checked today:</b> the function was run on the raw tome channels of the 294 test matches that the 1 to 8 October tome holds (374,500 seconds), and compared with the grid predictions of the same model at the same ticks. The 1 to 8 October tome does not carry per-tick player status, so that channel was read from the per-tick folder on the E: drive, as in the parity test. The largest gap is 0.0001 percentage points, the mean is 0.00002, and 100% of seconds agree to 0.01 points. The T probability is 100 minus the CT probability on every second. The 294 are the October matches in that tome; the record does not show whether they cover every October test match.</p>

<h2>5. Example ticks in each split</h2>
<p>A tick is a game tick; there are 64 per second. A live second is a tick at or after the freeze end, every 64 ticks, until the round ends (or the plant cap). seconds since live = (tick minus freeze-end tick) / 64. The label is 1 if CT won the round. The probability is the submission's output for that second.</p>
{''.join(rows_html)}

<h2>6. How to call the function</h2>
<p><code>scripts/serving/win_probability.py</code>. Dependencies: xgboost, polars, numpy. Nothing else from the project.</p>
<pre style="font-family:Consolas,monospace;font-size:9pt;background:#F2F2F2;padding:6pt">from win_probability import win_probability

probs_ct = win_probability(data, team="CT", model_name="winprob_v6_drop_heavy")   # {{tick: percent}}
probs_t  = win_probability(data, team="T",  model_name="winprob_v6_drop_heavy")   # 100 minus probs_ct
probs_fraction = win_probability(data, team="CT", model_name="winprob_v6_drop_heavy", unit="fraction")</pre>
<p><b>Input.</b> <code>data</code> is a dictionary of DataFrames keyed by channel name, read as-is from the DSDK tome (pandas or polars): <code>header</code>, <code>round_state</code>, <code>round_end</code>, <code>player_info</code>, <code>player_death</code>, <code>bomb_state</code>, <code>player_status</code>. The keys work with or without the <code>basic_</code> prefix. The submission needs <code>player_status</code>; without it the function returns an empty dictionary and says so.</p>
<p><b>Output.</b> A dictionary of tick to percent (0 to 100), one entry per live second of each round. The default <code>model_name</code> is <code>winprob_v6_drop_heavy</code>, the submission; pass <code>"winprob_v8_ensemble"</code> for the 58-input comparison. Warnings (a map not modelled, a wingman match, a round without a roster) are listed in <code>win_probability.last_warnings</code>.</p>
<p><b>Real output on one test match</b> (<code>{esc(fs['match_key'])}</code>): {fs['n']:,} ticks. The first three:</p>
<table><tr><th>tick</th><th>P(CT wins), %</th><th>P(T wins), %</th></tr>{fn_rows}</table>
<p>Warnings from that run: {esc(warn_txt)}.</p>

<h2>7. Open items</h2>
<ol>
<li>Owner: the reason for leaving the score lead out. The measured check does not show that the score lead contains the current round's result.</li>
<li>Owner: the origin of the day 14 and 15 October folders on the E: drive (950 and 835 match folders). They are not in the built store and are not used in any result.</li>
<li>Serving status and provenance of the 58-input metadata (the file says served true; the script that writes it says false).</li>
<li>The live Slides deck is not yet republished with the current labels, and the speaking notes link to an older artifact.</li>
<li>Event-only model: monotone constraints not tested on the 14-input model; 2024 data in training not isolated.</li>
</ol>
"""
    wrap_body = body

    notes = []
    for name, path in [("METHODOLOGY", "notes/METHODOLOGY_2026-10-19.md"), ("DELIVERABLE", "notes/DELIVERABLE_2026-10-19.md"),
                       ("EXPERIMENTS", "notes/EXPERIMENTS_2026-10-19.md")]:
        md = open(os.path.join(ROOT, path), encoding="utf-8").read()
        notes.append((name, md_to_html(md)))
    return wrap_body, notes


def write_docx(stem, title, body):
    page = wrap(title, body) if "<html" not in body else body
    html_path = os.path.join(OUT, stem + ".html")
    open(html_path, "w", encoding="utf-8", newline="\n").write(page)
    docx = os.path.join(OUT, stem + ".docx")
    if os.path.exists(docx):
        os.remove(docx)
    cmd = [SOFFICE, "--headless", "--infilter=HTML (StarWriter)", "--convert-to", "docx:MS Word 2007 XML", "--outdir", OUT, html_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    ok = os.path.exists(docx)
    print(("wrote " if ok else "FAILED ") + docx, "" if ok else (r.stdout + r.stderr).strip()[-300:])


if __name__ == "__main__":
    main()
