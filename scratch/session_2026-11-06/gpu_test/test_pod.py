import json, os, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
KEY = os.environ["RUNPOD_KEY"]
API = "https://rest.runpod.io/v1"
DC = sys.argv[1]
GPU = sys.argv[2]
RECORD = Path(os.environ["SCRATCH"]) / "gpu_test" / "pod_record.json"
PUB = (Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()
KEYFILE = str(Path.home() / ".ssh" / "id_ed25519")

def call(method, path, payload=None):
    req = urllib.request.Request(API + path, method=method,
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json", "User-Agent": "cs2-gpu-test/1"},
                                 data=None if payload is None else json.dumps(payload).encode())
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(body) if body.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")[:400]}

payload = {"name": "cs2-gpu-test", "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
           "gpuTypeIds": [GPU], "gpuTypePriority": "custom", "gpuCount": 1, "cloudType": "SECURE", "interruptible": False,
           "dataCenterIds": [DC], "dataCenterPriority": "custom", "containerDiskInGb": 20, "volumeInGb": 10,
           "volumeMountPath": "/workspace", "ports": ["22/tcp"], "env": {"PUBLIC_KEY": PUB}}
code, pod = call("POST", "/pods", payload)
print("create HTTP", code, "| pod id:", pod.get("id"), "| error:", pod.get("error", "")[:200])
if code not in (200, 201) or not pod.get("id"):
    sys.exit(1)
pod_id = pod["id"]
RECORD.write_text(json.dumps({"pod_id": pod_id, "dc": DC, "gpu": GPU, "created": time.strftime("%Y-%m-%d %H:%M:%S")}))
try:
    ip = port = None
    deadline = time.time() + 12 * 60
    while time.time() < deadline:
        _, st = call("GET", f"/pods/{pod_id}")
        status = st.get("desiredStatus")
        mappings = st.get("portMappings") or {}
        if status == "RUNNING" and st.get("publicIp") and mappings.get("22"):
            ip, port = st["publicIp"], mappings["22"]
            print("RUNNING:", st.get("machine", {}).get("gpuDisplayName"), "| cost/hr:", st.get("costPerHr"), "| dc:", st.get("machine", {}).get("location") or st.get("machine", {}).get("dataCenterId"))
            break
        time.sleep(20)
    if not ip:
        print("pod did not reach RUNNING with ssh in 12 min; last status:", st.get("desiredStatus"))
    else:
        for attempt in range(12):
            r = subprocess.run(["ssh", "-i", KEYFILE, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30", "-o", "BatchMode=yes",
                                "-p", str(port), f"root@{ip}",
                                "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; python3 -c 'import torch;print(\"torch cuda:\", torch.cuda.is_available())'"],
                               capture_output=True, text=True, timeout=180)
            if r.returncode == 0:
                print("SSH OK:\n" + r.stdout.strip())
                break
            print("ssh attempt", attempt + 1, "rc", r.returncode, (r.stderr or "")[-160:].strip())
            time.sleep(15)
finally:
    code, _ = call("DELETE", f"/pods/{pod_id}")
    print("terminate HTTP", code)
    time.sleep(5)
    _, st = call("GET", f"/pods/{pod_id}")
    print("after terminate: status", st.get("desiredStatus", st.get("error", "gone")))
