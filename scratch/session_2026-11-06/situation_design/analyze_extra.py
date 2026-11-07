"""Extra validation-only checks on the per-situation GDT output.

Reads docs/eval_reports/positions_validation_gdt/by_situation.csv, summary.json and the fixed dumb table
evaluation/intern_gdt/table.csv (the metric's reference table; not holdout, not test). Writes nothing into the repo.
"""
import csv
import json

BASE = "C:/Users/13144/Documents/pureskill/counterstrike-win-probability"
SRC_CSV = f"{BASE}/docs/eval_reports/positions_validation_gdt/by_situation.csv"
SRC_JSON = f"{BASE}/docs/eval_reports/positions_validation_gdt/summary.json"
TABLE = f"{BASE}/evaluation/intern_gdt/table.csv"
MODELS = ["p_full_trainctr", "p_base"]
CELL = ["alive_ct", "alive_t", "planted", "tb", "econ_ct", "econ_t"]


def load():
    with open(SRC_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {"key": tuple(r[c] for c in CELL), "n": int(r["datapoints"]), "matches": int(r["matches"]),
             "ct_win_share": float(r["ct_win_share"]), "p_table": float(r["p_table"]), "ll_table": float(r["ll_table"]),
             "mean_p": {}, "ll": {}, "gain": {}}
        for m in MODELS:
            d["mean_p"][m] = float(r[f"mean_{m}"])
            d["ll"][m] = float(r[f"ll_{m}"])
            d["gain"][m] = float(r[f"gain_{m}"])
        out.append(d)
    return out


def load_table():
    with open(TABLE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    t = {}
    for r in rows:
        key = tuple(r[c] for c in CELL)
        t[key] = r
    return t


def main():
    sits = load()
    N = sum(s["n"] for s in sits)
    m = "p_full_trainctr"
    total_loss = sum(s["n"] * s["ll"][m] for s in sits)

    # (a) top 10 by datapoints vs top 10 by total loss: same set?
    top_n = sorted(sits, key=lambda s: -s["n"])[:10]
    top_l = sorted(sits, key=lambda s: -(s["n"] * s["ll"][m]))[:10]
    print("(a) top-10 by datapoints identical to top-10 by total loss:",
          set(s["key"] for s in top_n) == set(s["key"] for s in top_l))
    print(f"    top-10 by datapoints: {sum(s['n'] for s in top_n)} datapoints, "
          f"{sum(s['n']*s['ll'][m] for s in top_n)/total_loss*100:.2f}% of model loss; "
          f"mean model ll in them {sum(s['n']*s['ll'][m] for s in top_n)/sum(s['n'] for s in top_n):.4f}")
    rest = [s for s in sits if s not in top_n]
    print(f"    other 1119 situations: {sum(s['n'] for s in rest)} datapoints, "
          f"{sum(s['n']*s['ll'][m] for s in rest)/total_loss*100:.2f}% of model loss; "
          f"mean model ll {sum(s['n']*s['ll'][m] for s in rest)/sum(s['n'] for s in rest):.4f}")
    print()

    # (b) threshold sensitivity of an equal-weight-per-situation GDT (the GDT_sit idea) at lower thresholds
    print(f"(b) equal-weight-per-situation gain for {m} at different minimum datapoint counts:")
    print(f"    {'min n':>6} {'situations':>10} {'datapoints':>10} {'share dp':>9} {'mean gain (eq wt)':>18} "
          f"{'worse sits':>10} {'worse share of sits':>20}")
    for thr in [500, 250, 100, 50, 20, 10, 1]:
        sel = [s for s in sits if s["n"] >= thr]
        if not sel:
            continue
        dp = sum(s["n"] for s in sel)
        eq = sum(s["gain"][m] for s in sel) / len(sel)
        w = sum(1 for s in sel if s["gain"][m] < 0)
        print(f"    {thr:>6} {len(sel):>10} {dp:>10} {dp/N*100:>8.2f}% {eq:>+18.5f} {w:>10} {w/len(sel)*100:>19.1f}%")
    weak_all = [s for s in sits if s["gain"][m] < 0]
    print(f"    weak situations (gain<0) by size: n<=10: {sum(1 for s in weak_all if s['n']<=10)}, "
          f"11-50: {sum(1 for s in weak_all if 11<=s['n']<=50)}, 51-100: {sum(1 for s in weak_all if 51<=s['n']<=100)}, "
          f"101-499: {sum(1 for s in weak_all if 101<=s['n']<=499)}, >=500: {sum(1 for s in weak_all if s['n']>=500)}")
    print(f"    weak situations with n>=500 (the only ones GDT_sit sees): "
          f"{[ (s['key'], s['n'], round(s['gain'][m],5)) for s in weak_all if s['n']>=500 ]}")
    print()

    # (c) upper bound on what fixing the weak situations could add to overall GDT
    neg = sum(s["n"] * s["gain"][m] for s in weak_all) / N
    gross_pos = sum(s["n"] * s["gain"][m] for s in sits if s["gain"][m] > 0) / N
    gdt = sum(s["n"] * s["gain"][m] for s in sits) / N
    print("(c) upper bound on overall GDT if every weak situation were lifted to gain 0:")
    print(f"    GDT {gdt:.5f}; weak contribution {neg:+.5f} ({abs(neg)/gdt*100:.1f}% of GDT); "
          f"gross positive {gross_pos:+.5f}; weak as % of gross positive {abs(neg)/gross_pos*100:.1f}%")
    print(f"    GDT with weak lifted to 0: {gdt - neg:.5f}")
    print()

    # (d) how much of the validation sample is in table backoff levels, by situation size
    table = load_table()
    levels = {}
    missing = 0
    for s in sits:
        r = table.get(s["key"])
        if r is None:
            missing += 1
            continue
        s["level"] = r["level"]
        s["n_train"] = int(r["n_train"])
        s["n_level"] = int(r["n_level"])
    print(f"(d) table backoff level for the validation situations (missing from table: {missing})")
    lv_all = {}
    for s in sits:
        if "level" not in s:
            continue
        lv_all.setdefault(s["level"], {"sits": 0, "dp": 0, "gain_w": 0.0, "weak": 0})
        lv_all[s["level"]]["sits"] += 1
        lv_all[s["level"]]["dp"] += s["n"]
        lv_all[s["level"]]["gain_w"] += s["n"] * s["gain"][m]
        lv_all[s["level"]]["weak"] += 1 if s["gain"][m] < 0 else 0
    for lv, v in sorted(lv_all.items(), key=lambda kv: -kv[1]["dp"]):
        print(f"    level '{lv}': {v['sits']} situations, {v['dp']} datapoints ({v['dp']/N*100:.2f}%), "
              f"datapoint-weighted gain {v['gain_w']/v['dp']:+.4f}, weak in {v['weak']}")
    print()
    print("    small-n situations: median n_train and level (n<=10 vs n>=100)")
    small = [s for s in sits if s.get("level") and s["n"] <= 10]
    big = [s for s in sits if s.get("level") and s["n"] >= 100]
    for name, grp in [("n<=10", small), ("n>=100", big)]:
        if not grp:
            continue
        nt = sorted(s["n_train"] for s in grp)
        print(f"    {name}: {len(grp)} situations, median n_train {nt[len(nt)//2]}, "
              f"share with table level 'full situation' "
              f"{sum(1 for s in grp if s['level']=='full situation')/len(grp)*100:.1f}%")
    print()
    print("    worst-by-gain situations: table level and n_train")
    for s in sorted(sits, key=lambda s: s["gain"][m])[:10]:
        if "level" in s:
            print(f"      gain {s['gain'][m]:+.4f} n={s['n']} level='{s['level']}' n_train={s['n_train']} n_level={s['n_level']} key={s['key']}")
    print()

    # (e) gain by table training size band (n_train), validation datapoints only
    print("(e) datapoint-weighted gain for {} by the table's own training count n_train:".format(m))
    bands = [(0, 29), (30, 99), (100, 999), (1000, 9999), (10000, 10**9)]
    for lo, hi in bands:
        sel = [s for s in sits if "n_train" in s and lo <= s["n_train"] <= hi]
        if not sel:
            print(f"    n_train {lo}-{hi}: 0")
            continue
        dp = sum(s["n"] for s in sel)
        g = sum(s["n"] * s["gain"][m] for s in sel) / dp
        print(f"    n_train {lo:>5}-{('max' if hi>=10**9 else hi):>5}: {len(sel):4d} situations, {dp:6d} datapoints "
              f"({dp/N*100:5.2f}%), datapoint-weighted gain {g:+.4f}, weak sits {sum(1 for s in sel if s['gain'][m]<0)}")


if __name__ == "__main__":
    main()
