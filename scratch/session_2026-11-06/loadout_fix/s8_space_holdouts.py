import json
import polars as pl
S = r"C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix/inputs"
built = set(json.load(open(S + "/built_match_keys.json")))
exp = set(json.load(open(S + "/match_keys_v2.json")))
tome_lines = [l.strip() for l in open(S + "/space_holdout_tome_match_keys.txt", encoding="utf-8") if l.strip()]
tome = set(tome_lines)
local26 = set(pl.read_parquet(r"C:/Users/13144/sprint/holdout_oct26_31_keys.parquet", columns=["match_key"])["match_key"].to_list())
print("space holdout_tome match_keys.txt: lines", len(tome_lines), "unique", len(tome), "sample", tome_lines[0])
print("  built AND space holdout_tome keys:", len(built & tome))
print("  export AND space holdout_tome keys:", len(exp & tome))
print("  local Oct26-31 parquet keys:", len(local26), "| space tome AND local parquet:", len(tome & local26), "| space tome minus local:", len(tome - local26), "| local minus space tome:", len(local26 - tome))
tome_dates = sorted({"-".join(k.split("/")[1:4]) for k in tome})
print("  space tome path-date min/max:", tome_dates[0], tome_dates[-1])
hb_space = set(json.load(open(S + "/space_adam_holdout_b_keys.json"))["match_keys"])
hb_local = set(json.load(open(r"C:/Users/13144/Documents/pureskill_internship/csds-cs2/data/combined/holdout_22_25_test_keys.json"))["match_keys"])
print("space adam_holdout_b_keys.json:", len(hb_space), "| local holdout B json:", len(hb_local), "| identical sets:", hb_space == hb_local, "| overlap:", len(hb_space & hb_local))
print("  built AND space holdout B:", len(built & hb_space), "| export AND space holdout B:", len(exp & hb_space))
print("  space tome AND holdout B:", len(tome & hb_space), "| local Oct26-31 AND holdout B:", len(local26 & hb_local))
