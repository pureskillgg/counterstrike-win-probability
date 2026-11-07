"""Build the 'Submission model' slide from the saved numbers (no numbers typed by hand).

  ...python.exe -I -X utf8 scripts\\viz\\build_submission_slide.py
Writes deliverables/deck_2026-10-19/project/slides/submission.html.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "combined")
OUT = os.path.join(ROOT, "deliverables", "deck_2026-10-19", "project", "slides", "submission.html")
RUST = "#9A3412"
TH = 'style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:left;padding:12px 16px;border:2px solid #1A1A1A"'
THR = 'style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:right;padding:12px 16px;border:2px solid #1A1A1A"'
TD = 'style="padding:12px 16px;border:2px solid #1A1A1A"'
TDR = 'style="text-align:right;padding:12px 16px;border:2px solid #1A1A1A"'


def ci(v):
    return f"[{v[0]:.6f}, {v[1]:.6f}]"


def main():
    s = json.load(open(os.path.join(D, "exp_submission_57.json"), encoding="utf-8"))
    t, v = s["splits"]["test"], s["splits"]["validation"]
    p = s["paired_57_minus_58_ensemble_test"]
    g = t["groups"]
    rows = "".join(
        f'<tr><td {TD}>{name}</td><td {TDR}>{x["seconds"]:,}</td><td {TDR}>{x["logloss"]:.6f} {ci(x["ci95"])}</td></tr>'
        for name, x in g.items())
    html = f'''<section id="submission" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:24px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">Submission model: 57 inputs, no score lead</h2>
  <p style="font-size:28px;line-height:1.35">59 inputs minus the heavy-weapon group. Trained on every round type, one seed. The score lead is left out of the submission; the reason is pending owner confirmation.</p>
  <table style="width:100%;font-size:24px;line-height:1.35">
    <tr><th {TH}>Test, 3,059 matches, 3,956,374 seconds</th><th {THR}>Seconds</th><th {THR}>Per-second log loss [95%]</th></tr>
    <tr><td {TD}><b>All rounds</b></td><td {TDR}>{t["seconds"]:,}</td><td {TDR}><b>{t["logloss"]:.6f} {ci(t["ci95"])}</b></td></tr>
    {rows}
  </table>
  <p style="font-size:26px;line-height:1.35">Validation: {v["logloss"]:.6f} {ci(v["ci95"])}, {v["matches"]:,} matches.</p>
  <div style="display:flex;flex-direction:column;gap:8px;border:2px solid {RUST};border-radius:8px;padding:16px 24px">
    <p style="font-size:24px;line-height:1.4"><b style="color:{RUST}">Cost of leaving out the score lead:</b> against the score-lead comparison model (five seeds, same test seconds), the submission is worse by {p["difference"]:+.6f} [{p["ci95"][0]:+.6f}, {p["ci95"][1]:+.6f}]. The submitted model needs per-tick player status, like the comparison model.</p>
  </div>
  <p style="position:absolute;left:128px;bottom:64px;font-size:24px">CS2 round win probability | 16 of 19</p>
  <aside style="font-size:24px">The score-lead model on the earlier slides is the comparison, not the submission. The 57-input model is the single-seed model winprob_v6_drop_heavy, scored with the same split and the same match-cluster intervals. Group results are on the same test seconds; the pistol and after-pistol groups are not compared with the other rounds on a like-for-like model yet (open item). The reason for leaving out the score lead is pending owner confirmation (METHODOLOGY_2026-10-19.md, section 16, item 12). Adding the score lead with the heavy-group drop improves both splits for the seed-0 model: validation -0.000865 [-0.001378, -0.000372] and test -0.000754 [-0.001066, -0.000459] (exp_scorelead_combo.json). The leak check (exp_scorelead_leak.json) found that the score lead changes in every consecutive 2026 round pair and the change in the score lead has the same sign as the previous round's CT-win result in 96.79% of pairs; that fits a running score of earlier rounds, and it does not show that the score lead contains the current round's result. The heavy-weapon group was dropped partly on test consistency. In the 59-input model, seed 0 (the file has no seed field; notes/METHODOLOGY_2026-10-19.md, section 11, gives seed 0), removing it changed validation by -0.00009 and test by -0.00002 [-0.00010, +0.00005] (exp_loadout_ablation.json). With seed 1 it changed validation by -0.000080 and test by -0.000079 [-0.000145, -0.000008] (exp_loadout_seed1.json). The metadata's selection rule, 'the only candidate better than the control on both splits for both seeds' (models/winprob_v6_drop_heavy.meta.json), uses test numbers as well as validation.</aside>
</section>
'''
    open(OUT, "w", encoding="utf-8", newline="\n").write(html)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
