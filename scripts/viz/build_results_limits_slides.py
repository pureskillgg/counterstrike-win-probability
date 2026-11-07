"""Build the 'Results' and 'What each model can do' slides from saved results (no numbers typed by hand, except the score-lead rows, which
are the earlier saved ensemble figures).

  ...python.exe -I -X utf8 scripts\\viz\\build_results_limits_slides.py
Writes deliverables/deck_2026-10-19/project/slides/results.html and limits.html.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from deck_svg import forest, RUST

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "combined")
SL = os.path.join(ROOT, "deliverables", "deck_2026-10-19", "project", "slides")
U = 1e3  # units of 0.001 log loss


def load(name):
    return json.load(open(os.path.join(D, name), encoding="utf-8"))


def main():
    p = load("exp_eventonly_paired.json")
    g = p["groups"]
    s = load("exp_submission_57.json")
    sub_t, sub_v = s["splits"]["test"], s["splits"]["validation"]
    sub_gap = s["paired_57_minus_58_ensemble_test"]["difference"]
    rows = [
        {"label": f"All rounds ({p['test_matches']:,} matches)", "value": p["event_only_minus_production"] * U,
         "lo": p["ci95"][0] * U, "hi": p["ci95"][1] * U, "color": RUST, "d": 2},
        {"label": "Pistol rounds (1, 13)", "value": g["pistol (1, 13)"]["event_only_minus_production"] * U,
         "lo": g["pistol (1, 13)"]["ci95"][0] * U, "hi": g["pistol (1, 13)"]["ci95"][1] * U, "color": RUST, "d": 2},
        {"label": "Rounds after pistol (2, 14)", "value": g["after pistol (2, 14)"]["event_only_minus_production"] * U,
         "lo": g["after pistol (2, 14)"]["ci95"][0] * U, "hi": g["after pistol (2, 14)"]["ci95"][1] * U, "color": RUST, "d": 2},
        {"label": "Other rounds", "value": g["other rounds"]["event_only_minus_production"] * U,
         "lo": g["other rounds"]["ci95"][0] * U, "hi": g["other rounds"]["ci95"][1] * U, "color": RUST, "d": 2},
    ]
    chart = forest(rows, x0=0, x1=150, ticks=[0, 25, 50, 75, 100, 125, 150], label_w=640, value_w=520,
                   axis_title="event-only minus score-lead model, same test seconds, units of 0.001 (higher = event-only worse)", zero_label="")
    results = f'''<section id="results" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:24px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">Results, with the test sets named</h2>
  <table style="width:100%;font-size:24px;line-height:1.4">
    <tr>
      <th style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:left;padding:10px 16px;border:2px solid #1A1A1A">Model (per-second log loss, lower is better)</th>
      <th style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:right;padding:10px 16px;border:2px solid #1A1A1A">Inputs</th>
      <th style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:right;padding:10px 16px;border:2px solid #1A1A1A">Validation</th>
      <th style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:right;padding:10px 16px;border:2px solid #1A1A1A">Test (matches, interval)</th>
    </tr>
    <tr>
      <td style="padding:10px 16px;border:2px solid #1A1A1A"><b>Submission: one seed, no score lead</b></td>
      <td style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A">57</td>
      <td style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A">{sub_v['logloss']:.6f} ({sub_v['matches']:,})</td>
      <td style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A">{sub_t['logloss']:.6f} [{sub_t['ci95'][0]:.6f}, {sub_t['ci95'][1]:.6f}] ({sub_t['matches']:,}, 2026)</td>
    </tr>
    <tr>
      <td style="padding:10px 16px;border:2px solid #1A1A1A">Score-lead comparison: five-seed ensemble</td>
      <td style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A">58</td>
      <td style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A">0.452722 (1,168)</td>
      <td style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A">0.455605 [0.45243, 0.45841] (3,059, 2026)</td>
    </tr>
    <tr>
      <td style="padding:10px 16px;border:2px solid #1A1A1A">Event-only, its own test set</td>
      <td style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A">14</td>
      <td style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A">0.498608 (1,383)</td>
      <td style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A">0.501242 (3,601, 2024 and 2026)</td>
    </tr>
    <tr>
      <td style="padding:10px 16px;border:2px solid #1A1A1A">Event-only, on the same 3,059 test matches</td>
      <td style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A">14</td>
      <td style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A">not run</td>
      <td style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A">{p['event_only_test_logloss']:.6f} ({p['test_matches']:,}, 2026)</td>
    </tr>
  </table>
  <p style="font-size:26px;line-height:1.3">Same test seconds, event-only minus score-lead model. Dark line: 95% match-cluster interval.</p>
  {chart}
  <p style="position:absolute;left:128px;bottom:64px;font-size:24px">CS2 round win probability | 15 of 19</p>
  <aside style="font-size:24px">The submission (57 inputs, no score lead) reaches {sub_t['logloss']:.6f} on the 3,059 2026 test matches, {sub_gap:.6f} worse than the score-lead comparison model (0.455605). The score-lead comparison model's 95% interval is 0.45243 to 0.45841 from match resampling; it is saved in data/combined/exp_rounds_groups.json, and a re-check on 2026-10-19 did not reproduce the endpoints exactly (notes/METHODOLOGY_2026-10-19.md, item 15). The event-only model scores {p['event_only_test_logloss']:.6f} on those same seconds, {p['event_only_minus_production'] * 1000:.1f} x10^-3 worse than the score-lead model, with a 95% interval of {p['ci95'][0] * 1000:.1f} to {p['ci95'][1] * 1000:.1f}. The gap is widest after the pistol round ({g['after pistol (2, 14)']['event_only_minus_production'] * 1000:.0f} x10^-3). It mixes two things: the loadout inputs the event-only model lacks, and its training on the 2024 matches as well as 2026, so it is not a pure measure of the loadout inputs. The 3,601-match row uses a different test set and is not comparable with the 3,059-match rows.</aside>
</section>
'''
    open(os.path.join(SL, "results.html"), "w", encoding="utf-8", newline="\n").write(results)

    limits = '''<section id="limits" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:40px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">What each model can do</h2>
  <div style="display:flex;gap:40px">
    <div style="flex:1;display:flex;flex-direction:column;gap:16px;border:2px solid #1A1A1A;border-radius:12px;padding:32px">
      <h3 style="font-size:40px;font-weight:700;line-height:1.2">Submission: 57 inputs</h3>
      <ul style="font-size:32px;line-height:1.4;display:flex;flex-direction:column;gap:8px">
        <li><b>57 = 13 event + 44 loadout</b> (loadout: 19 per side, plus 6 CT-minus-T differences); no score lead, no heavy-weapon group</li>
        <li>Needs the per-tick player status channel in the tome</li>
        <li>Training uses the per-second status extraction (all 14,814 2026 matches in the split); the cache holds 14,804 because the roster-defect rule removes every round of 10 matches (see the data slide)</li>
        <li>Serving not yet checked for this model (open item)</li>
      </ul>
    </div>
    <div style="flex:1;display:flex;flex-direction:column;gap:16px;border:2px solid #1A1A1A;border-radius:12px;padding:32px">
      <h3 style="font-size:40px;font-weight:700;line-height:1.2">Event-only: 14 inputs</h3>
      <ul style="font-size:32px;line-height:1.4;display:flex;flex-direction:column;gap:8px">
        <li><b>14 = 13 event + 1 score lead</b>, with no loadout inputs</li>
        <li>Runs on any tome with the event channels, including the 2024 matches</li>
        <li>Matches the grid on the 294 October test matches in its parity run (max gap 0.0001 percentage points)</li>
        <li>On the same 3,059 test matches it is 0.043 worse (interval 0.041 to 0.045)</li>
      </ul>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;gap:8px;border:2px solid #9A3412;border-radius:8px;padding:16px 24px">
    <p style="font-size:24px;line-height:1.4"><b style="color:#9A3412">Limits:</b> two data windows: 2024 uploads from February 2024 (2,749 matches in the split), used only by the event-only model; and 2026 matches uploaded 1 September to 13 October 2026 (14,814 in the split), used by the submission and comparison models. Matches come from FACEIT and Valve servers. The models cover seven maps; 1,295 matches on other maps are excluded and not measured. Positions, facing, blindness, and smoke and fire effects on the map (not carried grenades) are not in the model. Rows per second were tested only up to 8. The test set was computed many times during the work. Two inputs are always zero (st_kit_n_t and st_defusing_t; both are among the submission's 57). The 58-input model was never tested without its monotone constraints.</p>
    <p style="font-size:24px;line-height:1.4"><b>Score-lead comparison (58 = 13 event + 1 score lead + 44 loadout):</b> serving matches the grid on all 294 October test matches (largest gap 0.0001 points).</p>
  </div>
  <p style="position:absolute;left:128px;bottom:64px;font-size:24px">CS2 round win probability | 18 of 19</p>
  <aside style="font-size:24px">The submission needs per-tick player status to serve, and its serving has not been checked yet; the score-lead comparison model's serving was checked on all 294 October test matches. Training uses the per-second status extraction, which covers all 14,814 2026 matches in the split; the cache holds 14,804 because the roster-defect rule removes every round of 10 matches (traced on the data slide). The event-only model is the fallback for tomes without per-tick status, but the fallback is not automatic. The seven maps are the Active Duty pool that Wikipedia lists as of 6 July 2026; October is not confirmed at a primary source. The pool changed during the data windows, and no Valve notice was found. Neither model has been measured on maps outside the seven. An earlier model was run on de_vertigo and de_overpass with the map inputs set to zero, which is only an approximation. Sources for the dates and limits added here: notes/METHODOLOGY_2026-10-19.md, sections 3, 8 and 11, and section 16 (item 14).</aside>
</section>
'''
    open(os.path.join(SL, "limits.html"), "w", encoding="utf-8", newline="\n").write(limits)
    print("wrote results.html and limits.html")
    print(f"event-only minus score-lead {p['event_only_minus_production']:+.6f} [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}]")
    print(f"submission minus score-lead {sub_gap:+.6f}")


if __name__ == "__main__":
    main()
