"""Print selected fields of logs\swarm\E1\e1_results.json (read-only)."""
import json
import os

ROOT = r"G:\pureskill\csds-cs2"
r = json.load(open(os.path.join(ROOT, "logs", "swarm", "E1", "e1_results.json"), encoding="utf-8"))
for k in ["zero_shares", "per_round_distribution", "per_match_distribution", "auc_side_only", "subsets",
          "mean_per_round_by_side", "bootstrap"]:
    print(k, json.dumps(r[k]))
for a in r["auc_univariate_ct_win"]:
    print(json.dumps(a))
for ch in ["item_equip", "item_pickup", "item_dropped", "item_refund"]:
    print(ch, json.dumps(r["channel_profiles"][ch]["top_items_buyish_pre"][:12]))
