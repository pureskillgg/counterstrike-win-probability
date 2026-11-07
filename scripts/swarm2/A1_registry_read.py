import json, sys
p = r"G:\pureskill\csds-cs2\tomes\.meridian-outputs-cache\artifact_registry.json"
raw = open(p, "rb").read()
print("bytes", len(raw), "bom", raw[:3] == b"\xef\xbb\xbf")
d = json.loads(raw.decode("utf-8-sig"))
print("type", type(d).__name__, "len", len(d))
if isinstance(d, dict):
    print("top keys", list(d.keys())[:20])
    items = d.get("artifacts") or d.get("records") or d.get("entries") or d
else:
    items = d
recs = items if isinstance(items, list) else list(items.values())
print("n records", len(recs))
for r in recs:
    s = json.dumps(r)
    if "2026-10-01_to_08" in s and ("header" in s or "player_inputs" in s or "round_end" in s or "empty" in s.lower() or "removed" in s.lower()):
        print(json.dumps(r, indent=1)[:2500])
        print("=====")
