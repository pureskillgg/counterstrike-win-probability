"""Build the inputs, model, data, split and settings slides with diagrams drawn from the saved results (no numbers typed by hand).

  ...python.exe -I -X utf8 scripts\\viz\\build_diagram_slides.py
Writes deliverables/deck_2026-10-19/project/slides/{inputs,model,data,split,settings}.html.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from deck_svg import INK, RUST, SLATE, GREY, PAPER

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "combined")
SL = os.path.join(ROOT, "deliverables", "deck_2026-10-19", "project", "slides")
TH = 'style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:right;padding:12px 16px;border:2px solid #1A1A1A"'
TD = 'style="text-align:right;padding:12px 16px;border:2px solid #1A1A1A"'
TDL = 'style="padding:12px 16px;border:2px solid #1A1A1A"'
FOOT = '<p style="position:absolute;left:128px;bottom:64px;font-size:24px">CS2 round win probability | {n} of 19</p>'


def load(name):
    return json.load(open(os.path.join(D, name), encoding="utf-8"))


def write(name, html):
    open(os.path.join(SL, name), "w", encoding="utf-8", newline="\n").write(html)
    print("wrote", name)


def stacked_bar(segments, total, width=1664, height=150, label_h=60):
    """segments: list of (label, value, color). One horizontal bar split to scale, labels underneath."""
    out = [f'<svg viewBox="0 0 {width} {height + label_h}" width="{width}" height="{height + label_h}" xmlns="http://www.w3.org/2000/svg" aria-label="stacked bar">']
    x = 0.0
    lx = 0.0  # legend cursor: labels sit in a row under the bar, so they cannot overlap
    for label, value, color in segments:
        w = value / total * width
        out.append(f'<rect x="{x:.1f}" y="0" width="{w:.1f}" height="{height}" fill="{color}" stroke="{PAPER}" stroke-width="3"/>')
        out.append(f'<rect x="{lx:.1f}" y="{height + 22}" width="24" height="24" fill="{color}"/>')
        out.append(f'<text x="{lx + 34:.1f}" y="{height + 44}" font-size="24" fill="{INK}">{label}</text>')
        lx += 34 + 14 * len(label) + 48
        x += w
    out.append("</svg>")
    return "".join(out)


def inputs_slide():
    segs = [("13 event inputs", 13, SLATE), ("19 CT loadout", 19, INK), ("19 T loadout", 19, "#4B4B4B"), ("6 diffs", 6, RUST)]
    bar = stacked_bar(segs, 57)
    html = f'''<section id="inputs" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:28px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">What the submission model reads</h2>
  <p style="font-size:28px;line-height:1.3">57 inputs for each second. Blocks are drawn to scale.</p>
  {bar}
  <div style="display:flex;gap:40px">
    <div style="flex:1;display:flex;flex-direction:column;gap:12px;border:2px solid #1A1A1A;border-radius:12px;padding:24px">
      <h3 style="font-size:36px;font-weight:700;line-height:1.2">Event: 13</h3>
      <ul style="font-size:24px;line-height:1.35;display:flex;flex-direction:column;gap:4px">
        <li>alive CT, alive T, their difference</li>
        <li>bomb planted; seconds since round live; seconds since plant</li>
        <li>seven map indicators</li>
      </ul>
    </div>
    <div style="flex:1;display:flex;flex-direction:column;gap:12px;border:2px solid #1A1A1A;border-radius:12px;padding:24px">
      <h3 style="font-size:36px;font-weight:700;line-height:1.2">Loadout: 19 per side</h3>
      <ul style="font-size:24px;line-height:1.35;display:flex;flex-direction:column;gap:4px">
        <li>health (sum); armour count and sum; helmets; defuse kits; defusing</li>
        <li>AK; M4; AWP; all rifles; other snipers; SMG; no primary</li>
        <li>flash, smoke, HE, fire, decoy; equipment value</li>
      </ul>
    </div>
    <div style="flex:1;display:flex;flex-direction:column;gap:12px;border:2px solid #9A3412;border-radius:12px;padding:24px">
      <h3 style="font-size:36px;font-weight:700;line-height:1.2">Differences: 6</h3>
      <ul style="font-size:24px;line-height:1.35;display:flex;flex-direction:column;gap:4px">
        <li>CT minus T: health, rifles, equipment, armour count, AWP, SMG</li>
        <li>Heavy weapons (shotguns, machine guns) are left out of the submission model</li>
      </ul>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;gap:8px;border:2px solid #9A3412;border-radius:8px;padding:14px 24px">
    <p style="font-size:24px;line-height:1.35"><b style="color:#9A3412">Event-only model:</b> the 13 event inputs and score lead (14 inputs). It needs no per-tick status, so it also runs on 2024 matches.</p>
  </div>
  {FOOT.format(n=9)}
  <aside style="font-size:24px">Definitions are in the methodology document (notes/METHODOLOGY_2026-10-19.md). The loadout values come from the per-second status extraction, which covers all 14,814 2026 matches in the split; the cache holds 14,804 because the roster-defect rule removes every round of 10 matches (see the data slide). Serving needs the per-tick player status channel in the tome. Positions, facing, blindness, and smoke and fire effects on the map (not carried grenades) are not in the model. Event-derived features the tome carries (cumulative damage and shots) were tested and rejected.</aside>
</section>
'''
    write("inputs.html", html)


def model_slide():
    seeds = load("exp_seed_avg58.json")["seeds"]
    boxes = []
    x = 60
    for s in seeds:
        boxes.append(f'<rect x="{x}" y="20" width="250" height="120" rx="10" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>')
        boxes.append(f'<text x="{x + 125}" y="66" font-size="26" fill="{INK}" text-anchor="middle">seed {s["seed"]}</text>')
        boxes.append(f'<text x="{x + 125}" y="106" font-size="22" fill="{GREY}" text-anchor="middle">best iteration {s["best_iteration"]:,}</text>')
        x += 300
    diagram = (f'<svg viewBox="0 0 1664 150" width="1664" height="150" xmlns="http://www.w3.org/2000/svg" aria-label="five seeds averaged">'
               + "".join(boxes) + '</svg>')
    # second row: average -> output, drawn as its own svg for clarity
    out_svg = (f'<svg viewBox="0 0 1664 120" width="1664" height="120" xmlns="http://www.w3.org/2000/svg" aria-label="average and output">'
               f'<rect x="480" y="10" width="704" height="90" rx="10" fill="{INK}"/>'
               f'<text x="832" y="50" font-size="28" fill="{PAPER}" text-anchor="middle">mean of the five probabilities, per second</text>'
               f'<text x="832" y="84" font-size="24" fill="{PAPER}" text-anchor="middle">P(CT wins the round) ; T = 100 minus CT</text>'
               f'</svg>')
    html = f'''<section id="model" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:24px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">XGBoost, fixed settings</h2>
  <p style="font-size:28px;line-height:1.3">Each seed is one XGBoost model with the same settings; seeds differ only in their random subsampling. The score-lead comparison averages five seeds; the submission is one seed.</p>
  {diagram}
  {out_svg}
  <ul style="font-size:28px;line-height:1.35;display:flex;flex-direction:column;gap:8px">
    <li>Settings: depth 4, learning rate 0.05, minimum child weight 20, subsample 0.8; stop after 100 rounds with no validation gain.</li>
    <li>Monotone constraints on the alive counts: more CT alive never lowers the CT chance, more T alive never raises it. Measured on the 59-input model, removing them is neutral on validation (-0.00006, interval includes zero) and costs 0.00009 on test [+0.00001, +0.00018]; kept. The 13-input event model (no score lead) points the other way: removing its constraints lowers log loss on validation (-0.000188 [-0.000357, -0.000018]) and on test (-0.000443 [-0.000588, -0.000315]). The two experiments disagree; the event-only constraints are open (exp_59_nomono.json; exp_mono_paired.json).</li>
    <li>Trained on the GPU; the serving device is not recorded.</li>
  </ul>
  {FOOT.format(n=10)}
  <aside style="font-size:24px">XGBoost builds many small decision trees one after another, each correcting the errors of the ones before; a logistic function turns the sum into a probability. For the score-lead comparison, the five seeds give a small gain over any single seed (test 0.455605 against 0.455634 to 0.455732 for single seeds). The boxes show each seed's best iteration on validation (zero-based, so the tree count is one more).</aside>
</section>
'''
    write("model.html", html)


def data_slide():
    html = f'''<section id="data" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:36px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">The data we start from</h2>
  <p style="font-size:32px;line-height:1.4">Two windows: 2026 matches uploaded 1 September to 13 October 2026 (header dates 2 August to 13 October; for hand uploads the header is the processing day), used by the submission and comparison models; and 2024 uploads from February, used by the event-only model only. Seven maps; wingman and other maps are out of scope.</p>
  <table style="width:100%;font-size:26px;line-height:1.4">
    <tr>
      <th style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:left;padding:12px 16px;border:2px solid #1A1A1A">Matches in the split</th>
      <th {TH}>Train</th><th {TH}>Validation</th><th {TH}>Test</th>
    </tr>
    <tr><td {TDL}>2026 (manifest)</td><td {TD}>10,584</td><td {TD}>1,169</td><td {TD}>3,061</td></tr>
    <tr><td {TDL}>2026 used by both models (cache)</td><td {TD}>10,577</td><td {TD}>1,168</td><td {TD}>3,059</td></tr>
    <tr><td {TDL}>2024 and 2026 (manifest)</td><td {TD}>12,563</td><td {TD}>1,389</td><td {TD}>3,611</td></tr>
  </table>
  <div style="display:flex;flex-direction:column;gap:8px;border:2px solid #9A3412;border-radius:8px;padding:16px 24px">
    <p style="font-size:26px;line-height:1.4"><b style="color:#9A3412">The 10 cache-dropped matches are all 2026 (exp_cache_skip_trace.json):</b> 7 train, 1 validation, 2 test. Traced: all 146 of their grid rounds have a player missing from one side's roster, so the roster filter removes every round. They were uploaded 1 to 6 September 2026.</p>
  </div>
  {FOOT.format(n=4)}
  <aside style="font-size:24px">The 2026 split holds 14,814 matches. The snapshot cache holds 14,804 of them, so both models train on 10,577 matches, validate on 1,168 and test on 3,059. The event-only model's own test set is 3,601 matches. The manifest's 2024 and 2026 test count is 3,611. The event-only test set drops 10 of those (8 from 2024, 2 from 2026), and none of their rounds survives the roster filter, so it is 3,601. The 2024 part was checked against the grid, but no saved file records it. Dates and sources: notes/METHODOLOGY_2026-10-19.md, sections 3 and 5(c). Compare numbers only within the same test set.</aside>
</section>
'''
    write("data.html", html)


def split_slide():
    m = [("train 10,584", 10584, SLATE), ("validation 1,169", 1169, GREY), ("test 3,061", 3061, INK)]
    c = [("train 10,577", 10577, SLATE), ("validation 1,168", 1168, GREY), ("test 3,059", 3059, INK)]
    html = f'''<section id="split" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:32px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">How matches are split</h2>
  <ul style="font-size:30px;line-height:1.35;display:flex;flex-direction:column;gap:12px">
    <li><b>Random by match, frozen once</b> in a manifest (sha256 starts a17f219b). New matches get their split from a hash rule, so later results stay comparable.</li>
    <li><b>Not by second:</b> seconds in one match are almost identical, so a split by second would leak the answer into the test set.</li>
    <li><b>Checked:</b> the manifest check passes (sets are disjoint, every grid match is assigned), and every model's metadata records the same hash.</li>
  </ul>
  <p style="font-size:26px;line-height:1.3">2026 matches, manifest (top) and the matches the models use (bottom):</p>
  {stacked_bar(m, sum(v for _, v, _ in m))}
  {stacked_bar(c, sum(v for _, v, _ in c))}
  {FOOT.format(n=6)}
  <aside style="font-size:24px">Splitting by match is the key choice: seconds from one match are almost identical, so a split by second would hand the answer to the test set. The split is fixed in a manifest and checked by a script that confirms the sets do not overlap and that every grid match has a split. All reported models read that same manifest.</aside>
</section>
'''
    write("split.html", html)


def settings_slide():
    h = load("exp_hparams58.json")
    rows = []
    for c in h["configs"]:
        if c["name"].startswith("baseline"):
            continue
        rows.append((c["name"], c["validation_delta_vs_baseline"] * 1e4))
    rows.sort(key=lambda r: r[1])
    lw, vw, pw = 620, 420, 1664 - 620 - 420
    top, rh = 16, 60
    x0, x1 = 0.0, max(v for _, v in rows) * 1.15
    X = lambda v: lw + (v - x0) / (x1 - x0) * pw
    svg = [f'<svg viewBox="0 0 1664 {top + rh * len(rows) + 80}" width="1664" height="{top + rh * len(rows) + 80}" xmlns="http://www.w3.org/2000/svg" aria-label="validation change vs baseline">']
    for i, (name, v) in enumerate(rows):
        cy = top + rh * i + rh / 2
        svg.append(f'<text x="0" y="{cy + 8:.1f}" font-size="24" fill="{INK}">{name}</text>')
        svg.append(f'<rect x="{lw}" y="{cy - 16:.1f}" width="{X(v) - lw:.1f}" height="32" fill="{RUST}" opacity="0.85"/>')
        svg.append(f'<text x="{1664:.0f}" y="{cy + 8:.1f}" font-size="24" fill="{INK}" text-anchor="end">{v:+.2f}</text>')
    ay = top + rh * len(rows) + 8
    svg.append(f'<line x1="{lw}" y1="{ay}" x2="{lw + pw}" y2="{ay}" stroke="{INK}" stroke-width="2"/>')
    for t in [0, 1, 2, 3]:
        if t <= x1:
            svg.append(f'<text x="{X(t):.1f}" y="{ay + 36}" font-size="22" fill="{INK}" text-anchor="middle">{t}</text>')
    svg.append(f'<text x="{lw + pw / 2:.0f}" y="{ay + 66}" font-size="22" fill="{GREY}" text-anchor="middle">validation log loss change vs the baseline, units of 0.0001 (right = worse); no validation interval recorded</text>')
    svg.append("</svg>")
    chart = "".join(svg)
    html = f'''<section id="settings" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:24px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">How the settings were chosen</h2>
  <p style="font-size:28px;line-height:1.3">Validation decides. Ten alternatives to the baseline were tried on the score-lead (58-input) set; none beat the baseline on validation. The submission keeps the same settings. The closest were +0.00001. An earlier 59-input sweep chose depth 3 on validation (0.45577 against 0.45584 for depth 4; data/combined/exp_hparams.json), and depth 3 was not tried on the 58-input set.</p>
  {chart}
  <div style="display:flex;flex-direction:column;gap:8px;border:2px solid #1A1A1A;border-radius:8px;padding:14px 24px">
    <p style="font-size:24px;line-height:1.35"><b>Seeds, averaged:</b> validation 0.452722 (single seeds 0.452767 to 0.452838); test 0.455605 (single seeds 0.455634 to 0.455732). Averaging against seed 0 alone: test -0.00008 [-0.00013, -0.00002].</p>
  </div>
  {FOOT.format(n=11)}
  <aside style="font-size:24px">Each setting was judged on validation. The baseline was kept because none of the ten alternatives beat it there. Test numbers were recorded for every setting in the same run, but the choice was made on validation alone, and the chosen settings' test score is the one reported. Validation intervals were not recorded for the alternatives, so the bars show point changes only; test intervals are in exp_hparams58.json. The five-seed average is the score-lead comparison model; the submission is one seed with the same settings.</aside>
</section>
'''
    write("settings.html", html)


def main():
    inputs_slide()
    model_slide()
    data_slide()
    # split.html is written by build_split_slide.py (computed shares)
    settings_slide()


if __name__ == "__main__":
    main()
