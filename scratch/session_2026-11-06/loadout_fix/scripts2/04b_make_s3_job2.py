import json
import polars as pl
S = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix"
mi = pl.read_parquet(S + "/out/match_index.parquet")
ins = mi.filter(pl.col("in_scope"))
target2 = ins.filter(pl.col("src").is_in(["local_1hz", "local_1hz_2024"])).sort("match_key")["match_key"].to_list()
print("target2 (in-scope local-covered keys, channel check)", len(target2))
rule = open(S + "/scripts2/loadout_rule.py", encoding="utf-8").read()
body = open(S + "/scripts2/s3_body.py", encoding="utf-8").read()
body = body.replace("__TARGET__", repr(target2)).replace("__P3__", "[]").replace("__P4__", "[]")
body = body.replace('PARITY_OUT = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout_stage/parity_s3_rows.parquet"',
                    'PARITY_OUT = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout_stage/unused_parity2.parquet"')
open(S + "/scripts2/s3_job2.py", "w", encoding="utf-8").write(rule + "\n" + body)
json.dump(target2, open(S + "/out/s3_target2.json", "w"))
print("job2 written")
