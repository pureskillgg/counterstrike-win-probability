"""Create the GDT-objective swarm pods (RTX 3090 first, RTX 4090 as fallback), EU-RO-1, network volume mounted at /workspace.

Usage: python create_pods.py gdt-swarm-01 gdt-swarm-02 ...
Reads RUNPOD_KEY from the dnabert repo .env without printing it. Prints only pod id, name, status and cost per hour.
Appends each created pod to pods_created.txt so the closing step can find them.
"""
import json
import os
import sys
import urllib.request

ENV = "C:/Users/13144/Documents/dnabert-error-correction/.env"
KEY = None
for line in open(ENV, encoding="utf-8"):
    if line.startswith("RUNPOD_KEY="):
        KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
assert KEY, "RUNPOD_KEY not found"
pub = open(os.path.expanduser("~/.ssh/runpod_pureskill.pub"), encoding="utf-8").read().strip()
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pods_created.txt")

for name in sys.argv[1:]:
    body = {
        "name": name,
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
        "gpuCount": 1,
        "cloudType": "SECURE",
        "dataCenterIds": ["EU-RO-1"],
        "networkVolumeId": "z7yvc40ykx",
        "volumeMountPath": "/workspace",
        "containerDiskInGb": 50,
        "ports": ["22/tcp", "8888/http"],
        "env": {"PUBLIC_KEY": pub},
    }
    req = urllib.request.Request("https://rest.runpod.io/v1/pods", data=json.dumps(body).encode("utf-8"),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json",
                                          "User-Agent": "curl/8.5.0", "Accept": "*/*"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode("utf-8"))
            print("HTTP", r.status, {k: resp.get(k) for k in ["id", "name", "desiredStatus", "costPerHr", "gpu"]}, flush=True)
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(f"{resp.get('id')} {resp.get('name')}\n")
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, name, e.read().decode("utf-8")[:300], flush=True)
