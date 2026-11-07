import json

SCR = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
keys = json.load(open(SCR + "/export_match_keys.json"))
have = set(json.load(open(SCR + "/have_keys.json")))
mids = [i for i, k in enumerate(keys) if k in have]
body = open(SCR + "/space_extract_body.py", encoding="utf-8").read()
open(SCR + "/space_extract.py", "w", encoding="utf-8").write("MIDS = " + json.dumps(mids) + "\n" + body)
print("mids", len(mids), "first", mids[:5])
