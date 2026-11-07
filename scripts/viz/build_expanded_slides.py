"""Write the four added slides of the expanded deck: pool, roster exclusion, split rationale, and the submission, score-lead and event-only comparison.

Numbers come from the same sources as the methodology document; the structural numbers (counts, shares) are read from the data here.
Footers are numbered by renumber_footers() once the deck order is fixed (see the deck index).

  ...python.exe -I -X utf8 scripts\\viz\\build_expanded_slides.py
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SL = os.path.join(ROOT, "deliverables", "deck_2026-10-19", "project", "slides")
RUST = "#9A3412"
HEAD = 'style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:26px"'
TH = 'style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:left;padding:12px 16px;border:2px solid #1A1A1A"'
THR = 'style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:right;padding:12px 16px;border:2px solid #1A1A1A"'
TD = 'style="padding:12px 16px;border:2px solid #1A1A1A"'
TDR = 'style="text-align:right;padding:12px 16px;border:2px solid #1A1A1A"'
FOOT = '<p style="position:absolute;left:128px;bottom:64px;font-size:24px">CS2 round win probability | {n}</p>'


def wr(name, html):
    open(os.path.join(SL, name), "w", encoding="utf-8", newline="\n").write(html)
    print("wrote", name)


def pool():
    wr("pool.html", f'''<section id="pool" data-transition="fade" {HEAD}>
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">Scope and the map pool</h2>
  <ul style="font-size:30px;line-height:1.35;display:flex;flex-direction:column;gap:12px">
    <li><b>Seven maps:</b> ancient, anubis, cache, dust2, inferno, mirage, nuke. Chosen from the data: the 5v5 maps with real volume.</li>
    <li><b>Pool check, dated:</b> the seven are the Active Duty pool as of July 2026, per Wikipedia (a secondary source; no exact date is recorded in notes/METHODOLOGY_2026-10-19.md, section 4). October is not confirmed at a primary source.</li>
    <li><b>October:</b> Wikipedia's list as of July 2026 gives the same seven. No Valve notice dated after July 2026 was found.</li>
    <li><b>Guides disagree:</b> some list Overpass or Vertigo. Those lists disagree with Wikipedia's July 2026 list, and their dates are not checked.</li>
    <li><b>Out of scope:</b> 1,032 matches in 2024 and 263 in 2026 are on other maps and are not used.</li>
  </ul>
  {FOOT.format(n="3 of 19")}
  <aside style="font-size:24px">The seven maps were chosen from the data in the first eight days of October (1,561 of 1,603 non-wingman matches). Wikipedia's list of Active Duty maps gives these seven as of July 2026, when Cache replaced Overpass. A community wiki lists Vertigo where Wikipedia has Cache, so the two sources disagree on that one map. No Valve notice dated after July 2026 was found; October 2026 is not confirmed at a primary source.</aside>
</section>
''')


def roster():
    wr("roster.html", f'''<section id="roster" data-transition="fade" {HEAD}>
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">Excluded rounds: a missing player</h2>
  <ul style="font-size:30px;line-height:1.35;display:flex;flex-direction:column;gap:12px">
    <li><b>What it is:</b> in some rounds the roster lists one player fewer on one side. The missing player's side then has to be inferred from the events.</li>
    <li><b>Why one player is enough to drop the round:</b> the player's own status (alive or dead, health, armour, kit, weapons) is missing for every second of the round, and the events do not record it. Only the short side is known, from the counts, and the side inference was wrong in 37 of 90 sampled rows (about 40%).</li>
    <li><b>How much:</b> 0.095% of 2026 rounds (327 of 345,078) and 7.5% of 2024 rounds (6,262 of 83,526), over all rows of rounds.parquet (notes/METHODOLOGY_2026-10-19.md, section 5(a)). Small in size, but it is a correctness problem, so those rounds are dropped.</li>
    <li><b>The 10 cache-dropped matches:</b> all 2026 (exp_cache_skip_trace.json). Every round of each is affected (146 grid rounds), so the whole match drops out: 7 train, 1 validation, 2 test.</li>
  </ul>
  <div style="display:flex;flex-direction:column;gap:8px;border:2px solid {RUST};border-radius:8px;padding:16px 24px">
    <p style="font-size:24px;line-height:1.4"><b style="color:{RUST}">Measured:</b> keeping the 3,841 defect rounds in training (3,841 of the 329,028 used rounds across both eras, data/combined/e2e_winprob_v3.json) makes clean test log loss worse by 0.000042 [0.000020, 0.000062] for the 13-input winprob_v4_basic model (`data/combined/exp_roster_sensitivity.json`), not the event-only model. The event-only test set drops 10 manifest test matches (8 from 2024, 2 from 2026), so it is 3,601, not 3,611; no saved file records the 2024 part (notes/METHODOLOGY_2026-10-19.md, section 5(c)).</p>
  </div>
  {FOOT.format(n="5 of 19")}
  <aside style="font-size:24px">The rule removes a round only when its roster is incomplete. Those rounds are a small share of the data, and the same rule is applied to every model. The 40% figure (37 of 90 sampled rows) is from section 5(a), Roster defect, of notes/METHODOLOGY_2026-10-19.md.</aside>
</section>
''')


def split_rationale():
    wr("split_rationale.html", f'''<section id="split_rationale" data-transition="fade" {HEAD}>
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">Why 72 / 8 / 20</h2>
  <ul style="font-size:28px;line-height:1.35;display:flex;flex-direction:column;gap:10px">
    <li><b>Where it comes from:</b> the first training code's defaults: 20% of matches as test, then 10% of the remaining 80% as validation (8% of all). Train is the rest, 72%.</li>
    <li><b>Not a named standard:</b> 72/8/20 is the first code's default, carried over and not tuned. Blog guides cite 80/20 for a two-way split and 60/20/20, 70/15/15 or 80/10/10 for three-way splits.</li>
    <li><b>What our numbers support:</b> the test set (3,059 matches) gives a 95% interval of about ±0.0029 log loss for the submission (0.453311 to 0.459138; data/combined/exp_submission_57.json). The validation set (1,168 matches) gives paired 95% half-widths of about ±0.00012 (exp_59_nomono.json) to ±0.00029 (exp_families_pistol.json, CatBoost GPU), and about ±0.0005 for the score-lead change (exp_scorelead_combo.json), enough for the setting changes compared.</li>
    <li><b>Not tested:</b> no sweep of the fractions was run. The split is a choice, not a tuned optimum.</li>
  </ul>
  <table style="width:100%;font-size:24px;line-height:1.35">
    <tr><th {TH}>Matches used by both models</th><th {THR}>Train</th><th {THR}>Validation</th><th {THR}>Test</th></tr>
    <tr><td {TD}>Design target</td><td {TDR}>72%</td><td {TDR}>8%</td><td {TDR}>20%</td></tr>
    <tr><td {TD}>Obtained (14,804 matches)</td><td {TDR}>71.45%</td><td {TDR}>7.89%</td><td {TDR}>20.66%</td></tr>
  </table>
  {FOOT.format(n="7 of 19")}
  <aside style="font-size:24px">Sources, all general guidance: Lightly, MetricGate and APXML blog guides on train, validation and test splits; Chelsea Troy's summary of Andrew Ng's Machine Learning Yearning on dev and test sizes; Dietterich (1998) on single-split versus repeated-split tests, which sets no fraction. Our own intervals, not a source, support the sizes shown above. The outside sources are not saved in the project record and were not re-checked; they are unverified (notes/METHODOLOGY_2026-10-19.md, section 7).</aside>
</section>
''')


def compare():
    wr("compare.html", f'''<section id="compare" data-transition="fade" {HEAD}>
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">Submission, score-lead and event-only, side by side</h2>
  <table style="width:100%;font-size:24px;line-height:1.35">
    <tr><th {TH}></th><th {THR}>Submission</th><th {THR}>Score-lead</th><th {THR}>Event-only</th></tr>
    <tr><td {TD}>Inputs</td><td {TDR}>57: 13 event + 44 loadout</td><td {TDR}>58: 13 event + 1 score lead + 44 loadout</td><td {TDR}>14: 13 event + 1 score lead</td></tr>
    <tr><td {TD}>Training data</td><td {TDR}>2026 matches only</td><td {TDR}>2026 matches only</td><td {TDR}>2024 and 2026 matches</td></tr>
    <tr><td {TD}>Model</td><td {TDR}>one seed</td><td {TDR}>mean of five XGBoost seeds</td><td {TDR}>one seed, best iteration 548</td></tr>
    <tr><td {TD}>Needs per-tick player status</td><td {TDR}>yes</td><td {TDR}>yes</td><td {TDR}>no</td></tr>
    <tr><td {TD}>Test log loss, same 3,059 matches</td><td {TDR}>0.456411 [0.453311, 0.459138]</td><td {TDR}>0.455605 [0.45243, 0.45841]*</td><td {TDR}>0.498695</td></tr>
    <tr><td {TD}>Difference from the score-lead model</td><td {TDR}>+0.000806 [+0.000508, +0.001105]</td><td {TDR}></td><td {TDR}>+0.043090 [+0.041449, +0.044884]</td></tr>
  </table>
  <p style="font-size:26px;line-height:1.35">The submission's gap to the score-lead model is the cost of leaving the score lead out. The event-only gap is largest in rounds after the pistol (+0.139070). It mixes the loadout inputs with the 2024 training data, so it does not isolate the loadout effect.</p>
  {FOOT.format(n="17 of 19")}
  <aside style="font-size:24px">Both models are scored on the same 3,956,374 seconds. The event-only metadata figure of 0.501242 is on a different test set (3,601 matches, both years) and is not comparable with the 3,059-match figures. The fallback is not automatic: the event-only model must be requested by name. * The interval is saved in data/combined/exp_rounds_groups.json; a re-check on 2026-10-19 did not reproduce the endpoints exactly (notes/METHODOLOGY_2026-10-19.md, item 15).</aside>
</section>
''')


if __name__ == "__main__":
    pool()
    roster()
    split_rationale()
    compare()
