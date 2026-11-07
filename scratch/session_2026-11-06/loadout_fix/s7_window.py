import json, numpy as np, polars as pl
S = r"C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix/inputs"
mk = json.load(open(S + "/match_keys_v2.json"))
split = np.load(S + "/match_split.npy")
h26 = set(pl.read_parquet(r"C:/Users/13144/sprint/holdout_oct26_31_keys.parquet", columns=["match_key"])["match_key"].to_list())
dates = ["-".join(k.split("/")[1:4]) for k in mk]
win2131 = [i for i, d in enumerate(dates) if "2026-10-21" <= d <= "2026-10-31"]
win2631 = [i for i, d in enumerate(dates) if "2026-10-26" <= d <= "2026-10-31"]
print("export keys dated 2026-10-21..31:", len(win2131), "| split counts:", {int(c): int(sum(split[i] == c for i in win2131)) for c in (0, 1, 2)})
print("export keys dated 2026-10-26..31:", len(win2631), "| split counts:", {int(c): int(sum(split[i] == c for i in win2631)) for c in (0, 1, 2)})
print("of export keys dated 2026-10-26..31, present in Oct26-31 key file:", sum(1 for i in win2631 if mk[i] in h26))
print("Oct26-31 key file rows whose path date is in 2026-10-26..31:", sum(1 for k in h26 if "2026-10-26" <= "-".join(k.split("/")[1:4]) <= "2026-10-31"))
