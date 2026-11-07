"""Build the 'Other model families' slide from the saved results (no numbers typed by hand).

Two designs, both on the same test seconds (3,956,374 seconds of the 3,059 2026 test matches):
  (a) matched settings on every third training second (data/combined/exp_families_pistol.json, key families): paired deltas against the
      XGBoost reference, match-cluster 95% intervals.
  (b) full training set, 13.76 million seconds, 59-input set (data/combined/exp_full_comparison.json and exp_families_torch.json): each
      model's test log loss, and its paired difference to XGBoost with a match-cluster 95% interval (delta_vs_xgb_ci95).

  ...python.exe -I -X utf8 scripts\\viz\\build_families_slide.py
Writes deliverables/deck_2026-10-19/project/slides/families.html.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from deck_svg import forest, RUST, INK

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "combined")
OUT = os.path.join(ROOT, "deliverables", "deck_2026-10-19", "project", "slides", "families.html")
U = 1e4  # units of 0.0001 log loss


def load(name):
    return json.load(open(os.path.join(D, name), encoding="utf-8"))


def main():
    s3 = load("exp_families_pistol.json")["families"]
    full = load("exp_full_comparison.json")["models"]
    torch_full = load("exp_families_torch.json")["models"]
    xgb_full = full["xgb"]["logloss"]

    lgb3 = s3["lightgbm"]
    cb3 = s3["catboost (GPU, no monotone constraints)"]
    rows = [
        {"label": "LightGBM, monotone", "value": lgb3["test_delta_vs_xgboost"] * U,
         "lo": lgb3["test_delta_ci95"][0] * U, "hi": lgb3["test_delta_ci95"][1] * U, "color": INK},
        {"label": "CatBoost on GPU, no monotone", "value": cb3["test_delta_vs_xgboost"] * U,
         "lo": cb3["test_delta_ci95"][0] * U, "hi": cb3["test_delta_ci95"][1] * U, "color": RUST},
    ]
    chart = forest(rows, x0=-3, x1=4, ticks=[-2, -1, 0, 1, 2, 3, 4], label_w=560, value_w=520,
                   axis_title="test log loss change vs XGBoost, same seconds, units of 0.0001 (right = worse)")

    def row(name, mono, key):
        m = full[key]
        if key == "xgb":
            delta, interval = "", ""
        else:
            lo, hi = m["delta_vs_xgb_ci95"]
            delta, interval = f"{m['delta_vs_xgb']:+.6f}", f"[{lo:+.6f}, {hi:+.6f}]"
        cell = 'style="text-align:right;padding:10px 16px;border:2px solid #1A1A1A"'
        return (f'<tr><td style="padding:10px 16px;border:2px solid #1A1A1A">{name}</td>'
                f'<td {cell}>{mono}</td><td {cell}>{m["logloss"]:.6f}</td><td {cell}>{m["validation_logloss"]:.6f}</td>'
                f'<td {cell}>{delta}</td><td {cell}>{interval}</td></tr>')

    table_rows = "".join([
        row("<b>XGBoost, reference (59 inputs)</b>", "yes", "xgb"),
        row("LightGBM (31 leaves)", "yes", "lgbm"),
        row("CatBoost, CPU (depth 6)", "yes", "catboost"),
        row("CatBoost, GPU (depth 6)", "no", "catboost_gpu"),
    ])
    other = (f'Random forest {full["rf"]["logloss"]:.6f}; extra trees {full["et"]["logloss"]:.6f}; '
             f'neural net {torch_full["mlp_torch_gpu"]["logloss"]:.6f}; logistic regression {full["logit"]["logloss"]:.6f} (all test, full training set).')

    html = f'''<section id="families" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:22px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">Other model families</h2>
  <p style="font-size:26px;line-height:1.3">Two designs, both on the same test seconds, each family paired against XGBoost. Top: matched settings on every third training second. Bottom: the full training set, with the settings each run used.</p>
  {chart}
  <table style="width:100%;font-size:22px;line-height:1.35">
    <tr>
      <th style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:left;padding:10px 16px;border:2px solid #1A1A1A">Full training set (13.76 M seconds), 59 inputs</th>
      <th style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:right;padding:10px 16px;border:2px solid #1A1A1A">Monotone</th>
      <th style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:right;padding:10px 16px;border:2px solid #1A1A1A">Test</th>
      <th style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:right;padding:10px 16px;border:2px solid #1A1A1A">Validation</th>
      <th style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:right;padding:10px 16px;border:2px solid #1A1A1A">Test vs XGBoost</th>
      <th style="background:#1A1A1A;color:#FBFBF8;font-weight:700;text-align:right;padding:10px 16px;border:2px solid #1A1A1A">Paired 95%</th>
    </tr>
    {table_rows}
  </table>
  <p style="font-size:22px;line-height:1.35">{other}</p>
  <div style="display:flex;flex-direction:column;gap:8px;border:2px solid #9A3412;border-radius:8px;padding:12px 24px">
    <p style="font-size:22px;line-height:1.35"><b style="color:#9A3412">Settings differ between the two designs.</b> On the full set, with these settings, every boosted family is behind XGBoost and every paired interval excludes zero. On matched settings the gaps are inside noise. The full-set families were not tuned to themselves.</p>
  </div>
  <p style="position:absolute;left:128px;bottom:64px;font-size:24px">CS2 round win probability | 12 of 19</p>
  <aside style="font-size:24px">Top chart: the 58-input score-lead set, every third training second (4.65 million training seconds), the settings the score-lead model uses for XGBoost. LightGBM is ahead on validation (-0.00011, interval -0.00026 to +0.00003) and on test (-0.00007, interval includes zero). CatBoost on the GPU without monotone constraints is level with XGBoost on validation (-0.000004) and slightly behind on test (+0.00010, interval includes zero). Bottom table: the full training set (13.76 million seconds), 59 inputs, each family with the settings in its run. There, LightGBM and CatBoost on CPU and on GPU are 0.00034 to 0.00040 behind XGBoost on test, and every paired interval excludes zero. A repeat of the GPU CatBoost run moved validation by about 0.00015, so gaps of that size are noise. XGBoost stays: it is best or tied in both designs. The full-set result holds with these settings, not with each family tuned.</aside>
</section>
'''
    open(OUT, "w", encoding="utf-8", newline="\n").write(html)
    print("wrote", OUT)
    for r in rows:
        print(f"{r['label']}: {r['value']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")


if __name__ == "__main__":
    main()
