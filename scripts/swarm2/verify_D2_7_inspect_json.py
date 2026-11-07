"""D2-7 verifier: read-only inspection of the existing cross-period and batch-holdout JSON outputs."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
cp = json.loads((ROOT / "notes" / "model" / "sufficiency" / "cross_period.json").read_text(encoding="utf-8"))
bh = json.loads((ROOT / "notes" / "model" / "sufficiency" / "batch_holdout.json").read_text(encoding="utf-8"))

print("cross_period run_at:", cp["run_at"], "| devices:", cp["devices"])
print("cross_period label_rule:", cp["label_rule"])
print("verdict:", cp["verdict"])
print("counts_2024:")
for k, v in cp["counts_2024"].items():
    print("   ", k, "=", v)
print("drop_reasons_2024:", cp["drop_reasons_2024"])
print("score_checks_2024:", cp["score_checks_2024"])
print("completion_checks_2024:", cp["completion_checks_2024"])
print("cross_tome_duplicate_matches:", cp["cross_tome_duplicate_matches"])
print("metrics:")
for r in cp["metrics"]:
    print("   ", r["model"], "|", r["set"], "| matches", r["matches"], "rounds", r["rounds"], "states", r["states"],
          "| logloss", round(r["logloss"], 6), "| brier", round(r["brier"], 6), "| auc", round(r["auc"], 4))
print()
print("batch_holdout keys:", list(bh.keys()))
print("batch_holdout batch_definition:", bh["batch_definition"])
print("batch_holdout bakeoff_kept_check:", bh["bakeoff_kept_check"])
for R in ("forward", "reverse"):
    print(R, "name", bh[R]["name"], "test_label", bh[R]["test_label"], "verdict", bh[R]["verdict"])
    print("   in_batch LL", bh[R]["metrics"]["in_batch"]["logloss"], "other LL", bh[R]["metrics"]["other"]["logloss"])
    print("   sets", {k: (v["matches"], v["rounds"], v["states"]) for k, v in bh[R]["sets"].items()})
