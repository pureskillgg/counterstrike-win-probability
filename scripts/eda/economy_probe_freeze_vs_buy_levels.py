# Probe: do equipment and money change between round_freeze_end and buytime_ended?
# Reads only the two snapshot CSVs written by economy_snapshot.py (no E: reads).
import os
import pandas as pd
ROOT = r"G:\pureskill\csds-cs2"
f = pd.read_csv(os.path.join(ROOT, "logs", "eda", "economy_snapshot_rounds.csv"))
b = pd.read_csv(os.path.join(ROOT, "logs", "eda", "economy_snapshot_rounds_buyend.csv"))
p = f.merge(b[["match_key", "round"]], on=["match_key", "round"])
fb = f.merge(b, on=["match_key", "round"], suffixes=("_fz", "_by"))
print("paired rounds", len(fb))
for col in ["t_eq", "ct_eq", "t_money", "ct_money", "t_armor_n", "ct_armor_n", "t_rifle_n", "ct_rifle_n", "t_helmet_n", "ct_helmet_n"]:
    print("%-12s freeze mean %9.1f  buy_end mean %9.1f  corr %.3f" % (
        col, fb[col + "_fz"].mean(), fb[col + "_by"].mean(), fb[col + "_fz"].corr(fb[col + "_by"])))
print("share of paired rounds where team CT eq rises from freeze to buy_end:", round((fb["ct_eq_by"] > fb["ct_eq_fz"]).mean(), 3))
print("share where T eq rises:", round((fb["t_eq_by"] > fb["t_eq_fz"]).mean(), 3))
print("freeze t_eq mean over all kept rounds:", round(f["t_eq"].mean(), 1), " ct_eq:", round(f["ct_eq"].mean(), 1))
