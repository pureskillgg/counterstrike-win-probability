"""verify_E2_5 analysis: reads the VERIFY_E2_5_JSON lines written by verify_E2_5_bench.py from
logs\swarm2\verify_E2_5.log and prints the summary to stdout (the caller appends it to the same log).

  <cs2gpu_alt>\\envs\\default\\python.exe -B -I -X utf8 scripts\\swarm2\\verify_E2_5_analyze.py

Writes nothing except stdout. Numbers are medians over the reps present; with n reps the median
of an even count is the mean of the middle two (same convention as E2).
"""
import json
import statistics

LOG = r"G:\pureskill\csds-cs2\logs\swarm2\verify_E2_5.log"
PREFIX = "VERIFY_E2_5_JSON "

recs = []
with open(LOG, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if line.startswith(PREFIX):
            recs.append(json.loads(line[len(PREFIX):]))
ok = [r for r in recs if r.get("status") == "ok"]
print(f"records: {len(recs)} total, {len(ok)} status ok")
for r in recs:
    if r.get("status") != "ok":
        print("FAILED:", r.get("variant"), "rep", r.get("rep"), r.get("error_repr"))


def by_variant(v):
    return {r["rep"]: r for r in ok if r["variant"] == v}


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def fmt(xs):
    return ", ".join(f"{x:.3f}" for x in xs)


def paired(num, den, fn):
    """Per-rep ratio fn(num_rec)/fn(den_rec) for reps present in both."""
    reps = sorted(set(num) & set(den))
    return reps, [fn(num[k]) / fn(den[k]) for k in reps]


A_lgb, A_cb = by_variant("claim_lgb"), by_variant("claim_cb")
B_cb1 = by_variant("cb_iter1")
C_le1, C_nobag, C_lgb64 = by_variant("cb_le1"), by_variant("cb_le1_nobag"), by_variant("lgb_l64_b127")

print("\n== 1. Replication of the E2 claim (E2 settings, 1 thread, 100 rounds, same synthetic data) ==")
lgb_train = [A_lgb[k]["fit_wall"] for k in sorted(A_lgb)]
cb_fit = [A_cb[k]["fit_wall"] for k in sorted(A_cb)]
print(f"LightGBM train-only fit_wall, reps {sorted(A_lgb)}: {fmt(lgb_train)}  median {med(lgb_train):.4f}")
print(f"CatBoost fit_wall (includes pool/quantization), reps {sorted(A_cb)}: {fmt(cb_fit)}  median {med(cb_fit):.4f}")
print(f"ratio of medians CB/LGB (claim-style) = {med(cb_fit) / med(lgb_train):.3f}")
reps, pr = paired(A_cb, A_lgb, lambda r: r["fit_wall"])
print(f"paired per-rep ratios CB/LGB: {fmt(pr)}  min {min(pr):.3f}  median {med(pr):.3f}  max {max(pr):.3f}")
print("E2 claim (2 reps, machine ~100% busy): LGB 10.8495 s, CB 20.241 s, ratio 1.865")
print(f"absolute slowdown of E2 vs this run: LGB {10.8495 / med(lgb_train):.2f}x, CB {20.241 / med(cb_fit):.2f}x")
lls_lgb = sorted({A_lgb[k]["logloss"] for k in A_lgb})
lls_cb = sorted({A_cb[k]["logloss"] for k in A_cb})
print(f"valid logloss LGB distinct values: {lls_lgb} (E2: 0.559844)")
print(f"valid logloss CB distinct values: {lls_cb} (E2: 0.552949)")
print(f"thread check fit_cpu/fit_wall (about 1.0 means 1 thread): LGB {fmt([A_lgb[k]['fit_cpu_over_wall'] for k in sorted(A_lgb)])}"
      f" | CB {fmt([A_cb[k]['fit_cpu_over_wall'] for k in sorted(A_cb)])}")
print(f"whole-system busy % during fit: LGB {fmt([A_lgb[k]['busy_fit_pct'] or 0 for k in sorted(A_lgb)])}"
      f" | CB {fmt([A_cb[k]['busy_fit_pct'] or 0 for k in sorted(A_cb)])}")
print(f"CatBoost effective: boosting={A_cb[min(A_cb)]['eff_boosting_type']} bootstrap={A_cb[min(A_cb)]['eff_bootstrap_type']}"
      f" leaf_estimation_iterations={A_cb[min(A_cb)]['eff_leaf_estimation_iterations']}"
      f" leaf_estimation_method={A_cb[min(A_cb)]['eff_leaf_estimation_method']}")

print("\n== 2. Scope-matched comparison (LightGBM dataset build + train vs CatBoost fit, which includes its quantization) ==")
lgb_total = {k: A_lgb[k]["build_wall"] + A_lgb[k]["fit_wall"] for k in A_lgb}
print(f"LightGBM build_wall: {fmt([A_lgb[k]['build_wall'] for k in sorted(A_lgb)])}  median {med([A_lgb[k]['build_wall'] for k in A_lgb]):.4f}")
print(f"LightGBM build+train total: {fmt([lgb_total[k] for k in sorted(lgb_total)])}  median {med(list(lgb_total.values())):.4f}")
reps2, pr2 = [], []
common = sorted(set(A_cb) & set(lgb_total))
pr2 = [A_cb[k]["fit_wall"] / lgb_total[k] for k in common]
print(f"ratio of medians CB fit / LGB (build+train) = {med(cb_fit) / med(list(lgb_total.values())):.3f}")
print(f"paired per-rep ratios CB fit / LGB (build+train): {fmt(pr2)}  median {med(pr2):.3f}")

print("\n== 3. CatBoost fixed overhead (iterations=1) and training-only comparison ==")
oh = [B_cb1[k]["fit_wall"] for k in sorted(B_cb1)]
print(f"CatBoost fit with iterations=1 (pool + quantization + setup + 1 tree), reps {sorted(B_cb1)}: {fmt(oh)}  median {med(oh):.4f}")
oh_med = med(oh)
cb_train = [A_cb[k]["fit_wall"] - oh_med for k in sorted(A_cb)]
print(f"CatBoost training-only estimate (fit - overhead median): {fmt(cb_train)}  median {med(cb_train):.4f}")
print(f"ratio CB training-only / LGB train-only = {med(cb_train) / med(lgb_train):.3f}")
print(f"ratio CB training-only / LGB build+train = {med(cb_train) / med(list(lgb_total.values())):.3f}")

print("\n== 4. Config-closer variants (ONE change at a time from the claim settings; ASSUMPTION: not a full match) ==")
for name, d in [("claim_lgb (LGB num_leaves 63, max_bin 255)", A_lgb),
                ("lgb_l64_b127 (LGB num_leaves 64, max_bin 127)", C_lgb64),
                ("claim_cb (CB default leaf_estimation_iterations=10, MVS bootstrap)", A_cb),
                ("cb_le1 (CB leaf_estimation_iterations=1, MVS bootstrap)", C_le1),
                ("cb_le1_nobag (CB leaf_estimation_iterations=1, bootstrap No)", C_nobag)]:
    if not d:
        print(f"{name}: no ok records")
        continue
    ks = sorted(d)
    fits = [d[k]["fit_wall"] for k in ks]
    lls = sorted({d[k]["logloss"] for k in ks})
    extra = ""
    if "build_wall" in d[ks[0]]:
        extra = f" build median {med([d[k]['build_wall'] for k in ks]):.4f}"
    if "eff_bootstrap_type" in d[ks[0]]:
        extra = f" eff bootstrap={d[ks[0]]['eff_bootstrap_type']} le_iters={d[ks[0]]['eff_leaf_estimation_iterations']}"
    print(f"{name}: fit reps {fmt(fits)} median {med(fits):.4f} logloss {lls}{extra}")

if C_lgb64 and C_nobag:
    ks = sorted(set(C_lgb64) & set(C_nobag))
    r_train = [C_nobag[k]["fit_wall"] / C_lgb64[k]["fit_wall"] for k in ks]
    print(f"\nconfig-closer, train-only: CB(le1,nobag) / LGB(l64,b127) paired {fmt(r_train)} median {med(r_train):.3f}")
    r_tot = [C_nobag[k]["fit_wall"] / (C_lgb64[k]["build_wall"] + C_lgb64[k]["fit_wall"]) for k in ks]
    print(f"config-closer, scope-matched: CB fit / LGB(build+train) paired {fmt(r_tot)} median {med(r_tot):.3f}")
if C_le1 and A_lgb:
    ks = sorted(set(C_le1) & set(A_lgb))
    r_le1 = [C_le1[k]["fit_wall"] / A_lgb[k]["fit_wall"] for k in ks]
    print(f"claim LGB settings, CB(le1 only) / LGB train-only paired {fmt(r_le1)} median {med(r_le1):.3f}")

print("\n== 5. Summary of ratios (CatBoost slower = ratio above 1) ==")
print(f"claim-style CB fit / LGB train:            {med(cb_fit) / med(lgb_train):.3f}")
print(f"scope-matched CB fit / LGB build+train:    {med(cb_fit) / med(list(lgb_total.values())):.3f}")
print(f"training-only CB / LGB train:              {med(cb_train) / med(lgb_train):.3f}")
if C_lgb64 and C_nobag:
    print(f"config-closer train-only CB/LGB:           {med([C_nobag[k]['fit_wall'] for k in C_nobag]) / med([C_lgb64[k]['fit_wall'] for k in C_lgb64]):.3f}")
