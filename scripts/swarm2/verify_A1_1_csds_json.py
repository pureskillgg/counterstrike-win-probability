# verify_A1_1_csds_json.py
# Parse the gzip 'csds' JSON record for the two flagged matches (read-only) and print its top-level keys
# and any score/final/winner fields, to see whether it holds an independent final score.
import os, gzip, json, csv
ROOT = r"G:\pureskill\csds-cs2"
E = r"E:\csds_final\csds\2026\10"
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
rows = list(csv.DictReader(open(os.path.join(SCRATCH, "verify_A1_1_permatch.csv"), encoding="utf-8")))
probe = [r for r in rows if r["key"].endswith("m_4ceed6a37dd561e2028e114d/csds")
         or r["key"].endswith("m_ff06fa88583ee8b6377e2c3a/csds")]

def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if any(s in k.lower() for s in ["score", "final", "winner", "win", "round", "result", "over", "end"]):
                if not isinstance(v, (dict, list)):
                    yield p, v
                else:
                    yield p, f"<{type(v).__name__} len={len(v)}>"
            yield from walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            yield from walk(v, f"{path}[{i}]")

for r in probe:
    key = r["key"]
    dd, mid = key.split("/")[3], key.split("/")[4]
    raw = open(os.path.join(E, dd, mid, "csds"), "rb").read()
    doc = json.loads(gzip.decompress(raw).decode("utf-8"))
    print("\n", key[-40:], "| top-level keys:", list(doc.keys()))
    for p, v in list(walk(doc))[:60]:
        s = str(v)
        print("   ", p, "=", s[:160])
