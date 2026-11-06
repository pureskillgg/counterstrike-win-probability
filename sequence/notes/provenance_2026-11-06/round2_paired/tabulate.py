import glob, json, os, statistics, collections

SCR = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/final_results"
VOL = SCR + "/vol"
LOCAL = "G:/pureskill/csds-cs2/data/diagnostics"

files = []
for f in glob.glob(VOL + "/**/*metrics.json", recursive=True):
    rel = f.replace("\\", "/")
    if "/local_laptop/" in rel:
        continue  # copies of laptop files already read from the local folder
    files.append(("vol", f))
for f in glob.glob(LOCAL + "/pos24_*/**/*metrics.json", recursive=True):
    rel = f.replace("\\", "/")
    if "pos24_smoke" in rel or "pos24_promo" in rel:
        continue
    files.append(("local", f))

by_key = collections.defaultdict(list)
for src, f in files:
    d = json.load(open(f))
    if "final_val_log_loss" not in d:
        print("SKIP (no final_val_log_loss):", f)
        continue
    by_key[(d["arm"], d["seed"])].append((src, f.replace("\\", "/"), d))

print("== Duplicates (same arm+seed in more than one file) ==")
for k, v in sorted(by_key.items()):
    if len(v) > 1:
        for src, f, d in v:
            print(k, src, f.split("/")[-1], round(d["final_val_log_loss"], 6), d["steps"])

print("\n== Settings check (flag anything not steps 30000 / lookback 20 / test rows 0) ==")
for k, v in sorted(by_key.items()):
    for src, f, d in v:
        flags = []
        if d["steps"] != 30000: flags.append(f"steps={d['steps']}")
        if d["lookback_s"] != 20: flags.append(f"lookback={d['lookback_s']}")
        if d.get("test_rows_used", 0) != 0: flags.append(f"test_rows_used={d.get('test_rows_used')}")
        if d.get("batch") != 2048: flags.append(f"batch={d.get('batch')}")
        if d.get("lr") != 0.001: flags.append(f"lr={d.get('lr')}")
        if flags:
            print(k, src, f.split("/")[-1], flags)

print("\n== Per-arm final-step validation log loss (scored number) ==")
arms = collections.defaultdict(dict)
for (arm, seed), v in by_key.items():
    arms[arm][seed] = v[0][2]

rows = []
for arm, seeds in arms.items():
    vals = [seeds[s]["final_val_log_loss"] for s in sorted(seeds)]
    best = [seeds[s]["best_val_log_loss"] for s in sorted(seeds)]
    d0 = seeds[min(seeds)]
    rows.append({
        "arm": arm,
        "n": len(seeds),
        "seeds": sorted(seeds),
        "mean": statistics.mean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else float("nan"),
        "seed0": seeds[0]["final_val_log_loss"] if 0 in seeds else float("nan"),
        "per_seed": {s: round(seeds[s]["final_val_log_loss"], 6) for s in sorted(seeds)},
        "best_mean": statistics.mean(best),
        "seq_params": d0["seq_params"],
        "total_params": d0["total_params"],
        "device": d0.get("device"),
    })

rows.sort(key=lambda r: r["mean"])
for r in rows:
    print(f"{r['arm']:<14} n={r['n']} mean={r['mean']:.6f} sd={r['sd']:.6f} seed0={r['seed0']:.6f} "
          f"best_mean={r['best_mean']:.6f} seq_params={r['seq_params']} total_params={r['total_params']} "
          f"per_seed={r['per_seed']}")

print("\n== Seed-0 ranking (round-2 rule: top three by seed-0 final-step loss) ==")
s0 = sorted([r for r in rows if 0 in r["per_seed"]], key=lambda r: r["seed0"])
for i, r in enumerate(s0, 1):
    print(i, r["arm"], round(r["seed0"], 6))

json.dump({"rows": rows}, open(SCR + "/summary.json", "w"), indent=1, default=str)
print("\nwrote", SCR + "/summary.json")
