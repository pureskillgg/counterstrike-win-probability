import json, urllib.request
ENV = "C:/Users/<user>/Documents/dnabert-error-correction/.env"
KEY = ""
for line in open(ENV, encoding="utf-8"):
    if line.startswith("RUNPOD_KEY"):
        KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
        break
req = urllib.request.Request("https://rest.runpod.io/v1/pods", headers={"Authorization": "Bearer " + KEY, "User-Agent": "curl/8.5.0", "Accept": "*/*"})
pods = json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))
for p in pods:
    print(p.get("id"), p.get("name"), p.get("desiredStatus"), p.get("costPerHr"), p.get("publicIp"))
