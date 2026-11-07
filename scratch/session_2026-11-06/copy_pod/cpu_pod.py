import json, os, sys, time, urllib.request, urllib.error
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
KEY = os.environ["RUNPOD_KEY"]
API = "https://rest.runpod.io/v1"
REC = Path(os.environ["SCRATCH"]) / "copy_pod" / "cpu_pod_record.json"
PUB = (Path.home() / ".ssh" / "cs2_copy_tmp.pub").read_text().strip()
def call(method, path, payload=None):
    req = urllib.request.Request(API + path, method=method,
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json", "User-Agent": "cs2-copy/1"},
                                 data=None if payload is None else json.dumps(payload).encode())
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            b = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(b) if b.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")[:300]}
payload = {"name": "cs2-copy-cpu", "computeType": "CPU", "cpuFlavorIds": ["cpu3c"], "cpuFlavorPriority": "availability", "vcpuCount": 2,
           "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", "containerDiskInGb": 20, "volumeInGb": 20,
           "volumeMountPath": "/workspace", "ports": ["22/tcp"], "env": {"PUBLIC_KEY": PUB}, "cloudType": "COMMUNITY"}
code, pod = call("POST", "/pods", payload)
print("create HTTP", code, "| id:", pod.get("id"), "| error:", pod.get("error", "")[:200])
if code not in (200, 201) or not pod.get("id"):
    sys.exit(1)
REC.write_text(json.dumps({"pod_id": pod["id"], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "purpose": "space-to-pod copy test"}))
deadline = time.time() + 10 * 60
while time.time() < deadline:
    _, st = call("GET", "/pods/" + pod["id"])
    m = st.get("portMappings") or {}
    if st.get("desiredStatus") == "RUNNING" and st.get("publicIp") and m.get("22"):
        print("RUNNING | ip:", st["publicIp"], "| ssh port:", m["22"], "| cost/hr:", st.get("costPerHr"))
        break
    time.sleep(15)
else:
    print("not RUNNING with ssh within 10 min; status:", st.get("desiredStatus"))
