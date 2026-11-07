# verify A1-2: read-only probe of the raw E: source for match 7368f3ae (gzip JSON per channel)
import gzip, json, os
BASE = r"E:\csds_final\csds\2026\10\01\7368f3ae-7353-4f68-8d04-7c97f22ea2b8"
def read_channel(name):
    p = os.path.join(BASE, name)
    raw = gzip.open(p, "rb").read().decode("utf-8", errors="replace")
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            out.append({"_unparsed": line[:200]})
    return raw, out
raw_h, hdr = read_channel("header")
print("header lines:", len(hdr))
print("header keys sample:", list(hdr[0].keys())[:60] if hdr else None)
for k in ["t_starters_score_final", "ct_starters_score_final", "max_rounds", "match_date", "t_starters_avg_wins", "ct_starters_avg_wins"]:
    for rec in hdr:
        if isinstance(rec, dict) and k in rec:
            print("HEADER", k, "=", rec[k]); break
        if isinstance(rec, dict) and "payload" in rec and isinstance(rec["payload"], dict) and k in rec["payload"]:
            print("HEADER(payload)", k, "=", rec["payload"][k]); break
for name in ["round_end", "score_update", "team_change", "player_info"]:
    raw, recs = read_channel(name)
    print("=====", name, "lines", len(recs), "first:", (raw.splitlines()[0][:400] if raw.splitlines() else None))
