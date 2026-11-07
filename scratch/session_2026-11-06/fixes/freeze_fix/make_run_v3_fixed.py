"""Builds run_v3_fixed.py on the space from the v3 producer (loadout_rule.py header + s3_body.py body), with two changes only:
  1. the roster step takes the side of every confirmed (match, round, player) from override_rows.parquet, and adds the unrostered
     alive player of each confirmed round (the v3 roster drops players with no team code);
  2. the target is the 23 investigated matches, no parity sets, and the output arrays are the export_clean_v4 copies.
Everything else (chunk reads, dedupe, aggregation, step grid) is the unchanged v3 code.
"""
import json
import sys

V3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/scripts/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"
V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
inv = json.load(open(FX + "inv_matches.json"))
rule = open(V3 + "loadout_rule.py", encoding="utf-8").read()
body = open(V3 + "s3_body.py", encoding="utf-8").read()
assert "\r" not in body and "\r" not in rule

def rep(src, old, new):
    assert src.count(old) == 1, (src.count(old), old[:80])
    return src.replace(old, new)

body = rep(body, "TARGET = __TARGET__", "TARGET = " + json.dumps(inv))
body = rep(body, "P3 = __P3__", "P3 = []")
body = rep(body, "P4 = __P4__", "P4 = []")
body = rep(body, 'OUTD = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"', 'OUTD = "' + V4 + '"')
body = rep(body, "CHUNKS = [\"c%03d\" % i for i in range(0, 63)]\n",
           "CHUNKS = [\"c%03d\" % i for i in range(0, 63)]\n"
           "# freeze-tick fix: confirmed rounds (override_rows.parquet) take the side from place_name at freeze\n"
           "OVR = pl.read_parquet(\"" + FX + "override_rows.parquet\").select(\n"
           "    [\"match_key\", pl.col(\"round\").cast(pl.Int64), pl.col(\"player_id\").cast(pl.Int64), pl.col(\"side\").cast(pl.Int64).alias(\"team_code\")])\n"
           "OVR = OVR.filter(pl.col(\"match_key\").is_in(json.load(open(\"" + FX + "inv_matches.json\"))))\n"
           "log(\"override rows\", OVR.height, \"rounds\", OVR.select(\"match_key\", \"round\").unique().height)\n")
body = rep(body, '    roster = roster.select(["match_key", "round", "player_id", "team_code"])\n',
           '    roster = roster.select(["match_key", "round", "player_id", "team_code"])\n'
           '    rest = roster.join(OVR, on=["match_key", "round", "player_id"], how="anti")\n'
           '    roster = pl.concat([OVR.select("match_key", "round", "player_id", "team_code"), rest.select("match_key", "round", "player_id", "team_code")], how="vertical_relaxed")\n')
open(FX + "run_v3_fixed.py", "w", encoding="utf-8", newline="\n").write(rule + "\n\n" + body)
print("written", FX + "run_v3_fixed.py", flush=True)
