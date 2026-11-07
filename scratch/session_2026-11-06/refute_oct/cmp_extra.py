import json, csv, sys
sys.stdout.reconfigure(encoding="utf-8")
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/"
mine = set(json.load(open(S + "refute_oct/refute_s3_extra.json"))["extra"])
other = set(r["match_key"] for r in csv.DictReader(open(S + "check_unchecked_oct/s3_extra_oct21_25_keys.csv", encoding="utf-8")))
print("mine", len(mine), "other agent", len(other), "identical", mine == other, "diff", len(mine ^ other), flush=True)
