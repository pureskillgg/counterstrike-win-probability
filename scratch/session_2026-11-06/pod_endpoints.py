import json, os, sys, urllib.request
ENV = "C:/Users/13144/Documents/dnabert-error-correction/.env"
KEY = ""
for line in open(ENV, encoding="utf-8"):
    if line.startswith("RUNPOD_KEY"):
        KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
        break
for pid in sys.argv[1:]:
    req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pid}", headers={"Authorization": "Bearer " + KEY, "User-Agent": "curl/8.5.0", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=60) as r:
        p = json.loads(r.read().decode("utf-8"))
    print(pid, p.get("name"), p.get("desiredStatus"), "ip=", p.get("publicIp"), "ports=", p.get("portMappings"))
