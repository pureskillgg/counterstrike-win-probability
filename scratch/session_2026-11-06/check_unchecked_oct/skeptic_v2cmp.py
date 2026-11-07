import json, collections
G="G:/pureskill/csds-cs2/data/combined"
D="C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct"
g=json.load(open(G+"/split_manifest.json"))
v=json.load(open(D+"/skeptic_v2_copy.json"))
print("G assign == space v2 assign:", g["assign"]==v["assign"], len(g["assign"]), len(v["assign"]))
print("diff keys", len(set(g["assign"])^set(v["assign"])), "label diffs", sum(1 for k in g["assign"] if k in v["assign"] and g["assign"][k]!=v["assign"][k]))
print("space v2 history events:", [h.get("event") or h.get("action") for h in v.get("history",[])])
print("space v2 rule:", v.get("rule")[:120])
print("space v2 seed/created:", v.get("seed"), v.get("created_at"))
print("space v2 history last:", json.dumps(v.get("history")[-1])[:300])
