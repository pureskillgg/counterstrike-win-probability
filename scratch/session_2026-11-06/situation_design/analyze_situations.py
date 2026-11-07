"""Validation-only analysis of per-situation GDT output (positions_validation_gdt).

Reads only docs/eval_reports/positions_validation_gdt/by_situation.csv and summary.json.
Prints statistics to stdout. Writes nothing into the repository.
"""
import csv
import json
import math
import sys

SRC_CSV = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/docs/eval_reports/positions_validation_gdt/by_situation.csv"
SRC_JSON = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/docs/eval_reports/positions_validation_gdt/summary.json"
SIT_MIN = 500
MODELS = ["p_full_trainctr", "p_base", "p_full_oldctr"]


def load():
    with open(SRC_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({
            "key": (r["alive_ct"], r["alive_t"], r["planted"], r["tb"], r["econ_ct"], r["econ_t"]),
            "n": int(r["datapoints"]),
            "matches": int(r["matches"]),
            "ct_win_share": float(r["ct_win_share"]),
            "p_table": float(r["p_table"]),
            "ll_table": float(r["ll_table"]),
            "models": {m: {"ll": float(r[f"ll_{m}"]), "gain": float(r[f"gain_{m}"]),
                           "mean_p": float(r[f"mean_{m}"])} for m in MODELS},
        })
    return out


def fmt_key(k):
    return f"alive {k[0]}v{k[1]} planted={k[2]} tb={k[3]} econ_ct={k[4]} econ_t={k[5]}"


def main():
    summ = json.load(open(SRC_JSON, encoding="utf-8"))
    sits = load()
    N = sum(s["n"] for s in sits)
    print(f"file rows (situations): {len(sits)}")
    print(f"sum datapoints: {N}  (summary counts.datapoints = {summ['counts']['datapoints']})")
    print(f"situations with n>={SIT_MIN}: {sum(1 for s in sits if s['n'] >= SIT_MIN)}  "
          f"(summary = {summ['counts']['situations_with_500_or_more']})")
    print(f"matches in file max: {max(s['matches'] for s in sits)}; summary matches = {summ['counts']['matches']}")
    print(f"situations with n<{SIT_MIN}: {sum(1 for s in sits if s['n'] < SIT_MIN)}, "
          f"datapoints in them: {sum(s['n'] for s in sits if s['n'] < SIT_MIN)} "
          f"({100*sum(s['n'] for s in sits if s['n'] < SIT_MIN)/N:.2f}%)")
    print(f"situations with n<50: {sum(1 for s in sits if s['n'] < 50)}; n<100: {sum(1 for s in sits if s['n'] < 100)}; "
          f"n<=10: {sum(1 for s in sits if s['n'] <= 10)}")
    ns = sorted(s["n"] for s in sits)
    print(f"datapoints per situation: min {ns[0]}, median {ns[len(ns)//2]}, max {ns[-1]}")
    print()

    for m in MODELS:
        print("=" * 100)
        print(f"MODEL {m}")
        total_model_loss = sum(s["n"] * s["models"][m]["ll"] for s in sits)
        total_table_loss = sum(s["n"] * s["ll_table"] for s in sits)
        gdt_all = sum(s["n"] * s["models"][m]["gain"] for s in sits) / N
        big = [s for s in sits if s["n"] >= SIT_MIN]
        gdt_sit = sum(s["models"][m]["gain"] for s in big) / len(big)
        print(f"  check: GDT all = {gdt_all:.6f} (summary GDT for {m}: "
              f"{summ['models'][m]['GDT']:.6f}); GDT_sit = {gdt_sit:.6f} "
              f"(summary: {summ['models'][m]['GDT_sit']:.6f})")
        print(f"  check: table ll (datapoint-weighted) = {total_table_loss/N:.6f} "
              f"(summary {summ['models'][m]['table_logloss']:.6f}); model ll = {total_model_loss/N:.6f} "
              f"(summary {summ['models'][m]['model_logloss']:.6f})")

        # 2. share of datapoints in situations where the model is worse than the table
        worse = [s for s in sits if s["models"][m]["gain"] < 0]
        worse_big = [s for s in big if s["models"][m]["gain"] < 0]
        n_worse = sum(s["n"] for s in worse)
        print(f"  situations where model worse than table (gain<0): {len(worse)} of {len(sits)} "
              f"({100*len(worse)/len(sits):.1f}%)")
        print(f"    datapoints in them: {n_worse} of {N} = {100*n_worse/N:.2f}% of all datapoints")
        print(f"  n>={SIT_MIN}: worse in {len(worse_big)} of {len(big)} situations; "
              f"datapoints in them: {sum(s['n'] for s in worse_big)} of "
              f"{sum(s['n'] for s in big)} = {100*sum(s['n'] for s in worse_big)/sum(s['n'] for s in big):.2f}% "
              f"of n>={SIT_MIN} datapoints")
        # weight of the weak situations inside the overall GDT (sum of n*gain)
        pos_contrib = sum(s["n"] * s["models"][m]["gain"] for s in sits if s["models"][m]["gain"] > 0)
        neg_contrib = sum(s["n"] * s["models"][m]["gain"] for s in worse)
        print(f"  overall GDT contributions (sum n*gain / N): positive situations {pos_contrib/N:+.5f}, "
              f"worse situations {neg_contrib/N:+.5f}, net {(pos_contrib+neg_contrib)/N:+.5f}")
        print(f"  mean gain over worse situations (datapoint-weighted): "
              f"{neg_contrib/max(n_worse,1):+.4f}; over worse n>={SIT_MIN}: "
              f"{(sum(s['n']*s['models'][m]['gain'] for s in worse_big)/max(sum(s['n'] for s in worse_big),1)) if worse_big else float('nan'):+.4f}")

        # 3. concentration of model log loss
        by_n = sorted(sits, key=lambda s: -s["n"])
        top10n = by_n[:10]
        loss_share_top10n = sum(s["n"] * s["models"][m]["ll"] for s in top10n) / total_model_loss
        dp_share_top10n = sum(s["n"] for s in top10n) / N
        by_loss = sorted(sits, key=lambda s: -(s["n"] * s["models"][m]["ll"]))
        top10l = by_loss[:10]
        loss_share_top10l = sum(s["n"] * s["models"][m]["ll"] for s in top10l) / total_model_loss
        dp_share_top10l = sum(s["n"] for s in top10l) / N
        print(f"  concentration of total model loss ({total_model_loss:.1f} nat-datapoints):")
        print(f"    top 10 by datapoints: {100*dp_share_top10n:.2f}% of datapoints, "
              f"{100*loss_share_top10n:.2f}% of model loss")
        print(f"    top 10 by total loss: {100*dp_share_top10l:.2f}% of datapoints, "
              f"{100*loss_share_top10l:.2f}% of model loss")
        by_mean = sorted(sits, key=lambda s: -s["models"][m]["ll"])
        top10m = by_mean[:10]
        print(f"    top 10 by mean model loss: {100*sum(s['n'] for s in top10m)/N:.3f}% of datapoints, "
              f"{100*sum(s['n']*s['models'][m]['ll'] for s in top10m)/total_model_loss:.2f}% of model loss")
        # how many situations to reach 50% and 80% of model loss, sorted by total loss
        cum, k50, k80 = 0.0, None, None
        for i, s in enumerate(by_loss, 1):
            cum += s["n"] * s["models"][m]["ll"] / total_model_loss
            if k50 is None and cum >= 0.5:
                k50 = i
            if k80 is None and cum >= 0.8:
                k80 = i
                break
        print(f"    situations needed to reach 50% / 80% of model loss (ranked by total loss): {k50} / {k80}")
        print(f"  mean model ll over all datapoints {total_model_loss/N:.5f}; "
              f"median situation mean model ll {sorted(s['models'][m]['ll'] for s in sits)[len(sits)//2]:.5f}")

        # 4. are the worst situations rare?
        print(f"  10 most negative gains (worst vs table), with datapoints:")
        worst = sorted(sits, key=lambda s: s["models"][m]["gain"])[:10]
        for s in worst:
            print(f"    gain {s['models'][m]['gain']:+.4f}  n={s['n']:5d}  matches={s['matches']:3d}  "
                  f"ll_model={s['models'][m]['ll']:.4f}  ll_table={s['ll_table']:.4f}  "
                  f"p_table={s['p_table']:.3f}  ct_win_share={s['ct_win_share']:.3f}  {fmt_key(s['key'])}")
        print(f"    these 10: datapoints total {sum(s['n'] for s in worst)} = "
              f"{100*sum(s['n'] for s in worst)/N:.3f}% of all; median n {sorted(s['n'] for s in worst)[5]}")
        print(f"  10 situations with highest mean model loss, with datapoints:")
        for s in top10m:
            print(f"    ll_model={s['models'][m]['ll']:.4f}  gain={s['models'][m]['gain']:+.4f}  n={s['n']:5d}  "
                  f"ct_win_share={s['ct_win_share']:.3f}  p_table={s['p_table']:.3f}  {fmt_key(s['key'])}")
        print(f"  gain by datapoint-count band (n = datapoints):")
        bands = [(1, 10), (11, 50), (51, 100), (101, 250), (251, 499), (500, 999), (1000, 10**9)]
        for lo, hi in bands:
            sel = [s for s in sits if lo <= s["n"] <= hi]
            if not sel:
                print(f"    n {lo}-{hi if hi < 10**9 else 'max'}: 0 situations")
                continue
            dp = sum(s["n"] for s in sel)
            g = sum(s["n"] * s["models"][m]["gain"] for s in sel) / dp
            nw = sum(1 for s in sel if s["models"][m]["gain"] < 0)
            print(f"    n {lo:>4}-{('max' if hi >= 10**9 else str(hi)):>5}: {len(sel):4d} situations, "
                  f"{dp:6d} datapoints ({100*dp/N:5.2f}%), datapoint-weighted gain {g:+.4f}, "
                  f"worse in {nw} ({100*nw/len(sel):.0f}% of band)")
        print()

    # n >= 500 detail for trainctr
    m = "p_full_trainctr"
    print("=" * 100)
    print(f"n>={SIT_MIN} situations for {m} (sorted by datapoints):")
    for s in sorted([s for s in sits if s["n"] >= SIT_MIN], key=lambda s: -s["n"]):
        print(f"  n={s['n']:5d}  matches={s['matches']:3d}  gain={s['models'][m]['gain']:+.4f}  "
              f"ll_model={s['models'][m]['ll']:.4f}  ll_table={s['ll_table']:.4f}  {fmt_key(s['key'])}")
    print()
    # Check whether the 'worse' situations in the trainctr model overlap with n<500
    print(f"model-vs-model: situations where p_full_trainctr is worse than p_base (gain_trainctr - gain_base < 0): "
          f"{sum(1 for s in sits if s['models'][m]['gain'] < s['models']['p_base']['gain'])}")


if __name__ == "__main__":
    main()
