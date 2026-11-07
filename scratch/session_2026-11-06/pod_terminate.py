import json, sys, urllib.request
ENV = "C:/Users/<user>/Documents/dnabert-error-correction/.env"
KEY = ""
for line in open(ENV, encoding="utf-8"):
    if line.startswith("RUNPOD_KEY"):
        KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
        break
pid, want_name, want_ip = sys.argv[1], sys.argv[2], sys.argv[3]
H = {"Authorization": "Bearer " + KEY, "User-Agent": "curl/8.5.0", "Accept": "*/*"}
req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pid}", headers=H)
p = json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))
print("found:", p.get("name"), p.get("publicIp"), p.get("desiredStatus"))
if p.get("name") != want_name or p.get("publicIp") != want_ip:
    print("MISMATCH, not terminating"); sys.exit(1)
req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pid}", headers=H, method="DELETE")
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print("DELETE HTTP", r.status)
except urllib.error.HTTPError as e:
    print("DELETE HTTP", e.code, e.read().decode("utf-8")[:300])
