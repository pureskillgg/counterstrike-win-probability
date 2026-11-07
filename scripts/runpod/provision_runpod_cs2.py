#!/usr/bin/env python3
"""Create / inspect / stop / terminate a RunPod pod for the heavy GPU steps of the CS2 win-probability pipeline.

Same method as C:\\Users\\13144\\Documents\\dnabert-error-correction\\provision_runpod_caduceus.py (documented in that repo's HANDOFF_RUNPOD.txt):
RunPod REST API https://rest.runpod.io/v1, Bearer key, RunPod-published PyTorch devel image (it bundles the SSH bootstrap), Secure Cloud,
SSH public key injected as PUBLIC_KEY. Differences: standard library only (no `requests` needed), one file for the whole pod life cycle, and the
pod id is written to logs\\runpod_pod.json at once so a billing pod can never be lost track of.

  python provision_runpod_cs2.py                       # = --dry-run: prints the payload, creates nothing, free
  python provision_runpod_cs2.py --launch --i-confirm-cost   # creates the pod: COSTS MONEY from this moment until it is terminated
  python provision_runpod_cs2.py --status <pod_id> [--wait]   # state, price per hour, ssh host/port (--wait polls until ssh is reachable)
  python provision_runpod_cs2.py --list                # every pod on the account with its state (run this before walking away)
  python provision_runpod_cs2.py --stop <pod_id>       # stop (compute billing stops; the volume still bills a little)
  python provision_runpod_cs2.py --terminate <pod_id>  # delete: billing ends. Copy the results out FIRST.

The key is read from the environment variable RUNPOD_KEY or from RUNPOD_ENV_FILE (default: the dnabert repo's .env, the only place it lives). It is
never printed, logged, put on a command line or written to a file by this script.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
API = "https://rest.runpod.io/v1"
DEFAULT_ENV_FILE = r"C:\Users\13144\Documents\dnabert-error-correction\.env"
POD_RECORD = Path(r"G:\pureskill\csds-cs2\logs\runpod_pod.json")
# The dnabert project's proven choice was an A6000 on Secure Cloud (HANDOFF_RUNPOD.txt). This job is host-RAM and CPU bound, not GPU bound
# (XGBoost on the 5.6 M-row training set needs a few GB of GPU memory), so the default is the CHEAPEST card that fits, tried in this order
# (gpuTypePriority "custom"), on Community Cloud (on demand, not interruptible); the dearer cards are fallbacks when nothing cheaper has capacity.
GPU_TYPE_IDS = ["NVIDIA RTX A4000", "NVIDIA RTX A5000", "NVIDIA RTX A4500", "NVIDIA GeForce RTX 3090", "NVIDIA L4", "NVIDIA GeForce RTX 4090", "NVIDIA RTX 4000 Ada Generation",
                "NVIDIA L40", "NVIDIA RTX A6000", "NVIDIA A40"]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"  # RunPod's own image: it bundles the SSH bootstrap (a stock python image has none)
CLOUD_TYPE = "COMMUNITY"
SSH_PUB = Path.home() / ".ssh" / "id_ed25519.pub"


def read_key():
    k = os.environ.get("RUNPOD_KEY")
    if k:
        return k
    p = Path(os.environ.get("RUNPOD_ENV_FILE", DEFAULT_ENV_FILE))
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("RUNPOD_KEY") and "=" in line:
                return line.partition("=")[2].strip().strip('"').strip("'")
    sys.exit("RUNPOD_KEY not found (environment or the env file). Aborting; never pass the key on a command line.")


def call(method, path, payload=None):
    req = urllib.request.Request(API + path, method=method, headers={"Authorization": f"Bearer {read_key()}", "Content-Type": "application/json",
                                                                      "User-Agent": "cs2-winprob-provisioner/1"},
                                 data=None if payload is None else json.dumps(payload).encode())
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(body) if body.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")[:600]}


def payload(a):
    pub = SSH_PUB.read_text().strip() if SSH_PUB.exists() else ""
    if not pub:
        print(f"WARNING: no SSH public key at {SSH_PUB}: only the web terminal would work", file=sys.stderr)
    gpus = [g.strip() for g in a.gpu.split(",") if g.strip()] if a.gpu else GPU_TYPE_IDS
    p = {"name": a.name, "imageName": IMAGE, "gpuTypeIds": gpus, "gpuTypePriority": "custom", "gpuCount": 1, "cloudType": a.cloud,
         "interruptible": False, "containerDiskInGb": a.container_gb, "volumeInGb": a.volume_gb, "volumeMountPath": "/workspace", "ports": ["22/tcp"],
         "env": {"PUBLIC_KEY": pub}}
    if not a.no_ram_hint:
        p["minRAMPerGPU"] = a.min_ram_gb
        p["minVCPUPerGPU"] = a.min_vcpu
    if a.network_volume_id:
        p["networkVolumeId"] = a.network_volume_id
        if a.datacenter:
            p["dataCenterIds"] = [a.datacenter]
            p["dataCenterPriority"] = "custom"
    return p


def save_record(d):
    POD_RECORD.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(POD_RECORD.read_text()) if POD_RECORD.exists() else {}
    old.update(d)
    POD_RECORD.write_text(json.dumps(old, indent=1))


def summarize(pod):
    ports = pod.get("portMappings") or {}
    ip = pod.get("publicIp")
    out = {"id": pod.get("id"), "status": pod.get("desiredStatus"), "costPerHr": pod.get("costPerHr"),
           "gpu": (pod.get("machine") or {}).get("gpuTypeId") or (pod.get("gpu") or {}).get("displayName"), "publicIp": ip, "ssh_port": ports.get("22")}
    if ip and ports.get("22"):
        out["ssh"] = f"ssh -i ~/.ssh/id_ed25519 -p {ports['22']} root@{ip}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--i-confirm-cost", action="store_true", help="required with --launch: the pod bills per hour until terminated")
    ap.add_argument("--status", metavar="POD_ID")
    ap.add_argument("--wait", action="store_true", help="with --status: poll until the pod runs and ssh has an address (max --wait-min)")
    ap.add_argument("--wait-min", type=float, default=15.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stop", metavar="POD_ID")
    ap.add_argument("--terminate", metavar="POD_ID")
    ap.add_argument("--name", default="cs2-winprob-final")
    ap.add_argument("--gpu", default="", help="comma list of RunPod GPU type ids, tried in this order (default: cheapest first, see GPU_TYPE_IDS)")
    ap.add_argument("--cloud", default=CLOUD_TYPE, choices=["COMMUNITY", "SECURE"])
    ap.add_argument("--container-gb", type=int, default=40)
    ap.add_argument("--volume-gb", type=int, default=80, help="/workspace: data ~10 GB + working copies + outputs")
    ap.add_argument("--min-ram-gb", type=int, default=64, help="minimum host RAM per GPU (the heavy steps hold 10-20 GB; the local machine could not)")
    ap.add_argument("--min-vcpu", type=int, default=8)
    ap.add_argument("--try-all", action="store_true", help="with --launch: try Community with the RAM floor, Community without, Secure with, Secure without, in that order")
    ap.add_argument("--no-ram-hint", action="store_true", help="omit minRAMPerGPU/minVCPUPerGPU if the API rejects them")
    ap.add_argument("--network-volume-id", default="", help="attach this persistent network volume at /workspace (its data centre limits where the pod can run)")
    ap.add_argument("--datacenter", default="", help="data centre id, e.g. with --create-volume (the volume and the pod must share it)")
    ap.add_argument("--create-volume", metavar="NAME", help="create a persistent network volume (costs per GB per month until deleted)")
    ap.add_argument("--volume-size-gb", type=int, default=30)
    ap.add_argument("--list-volumes", action="store_true")
    a = ap.parse_args()

    if a.create_volume:
        if not a.datacenter:
            sys.exit("--create-volume needs --datacenter (the volume lives in one data centre)")
        code, body = call("POST", "/networkvolumes", {"name": a.create_volume, "size": a.volume_size_gb, "dataCenterId": a.datacenter})
        print(f"create volume: HTTP {code} {json.dumps(body)[:400]}")
        if code in (200, 201):
            save_record({"network_volume": body, "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return
    if a.list_volumes:
        code, body = call("GET", "/networkvolumes")
        vols = body if isinstance(body, list) else body.get("networkVolumes", []) if isinstance(body, dict) else []
        print(f"HTTP {code}; {len(vols)} network volume(s)")
        for v in vols:
            print(json.dumps(v))
        return

    if a.list:
        code, body = call("GET", "/pods")
        pods = body if isinstance(body, list) else body.get("pods", []) if isinstance(body, dict) else []
        print(f"HTTP {code}; {len(pods)} pod(s)")
        for p in pods:
            print(json.dumps(summarize(p)))
        return
    if a.status:
        t_end = time.time() + a.wait_min * 60
        while True:
            code, pod = call("GET", f"/pods/{a.status}")
            s = summarize(pod) if code == 200 else {"http": code, **pod}
            print(json.dumps(s), flush=True)
            if code == 200:
                save_record({"last_status": s, "last_checked": time.strftime("%Y-%m-%dT%H:%M:%S")})
            if not a.wait or (code == 200 and s.get("ssh")) or time.time() > t_end:
                return
            time.sleep(10)
    if a.stop or a.terminate:
        pid = a.stop or a.terminate
        code, body = call("POST", f"/pods/{pid}/stop") if a.stop else call("DELETE", f"/pods/{pid}")
        print(f"{'stop' if a.stop else 'terminate'} {pid}: HTTP {code} {json.dumps(body)[:300]}")
        save_record({"stopped" if a.stop else "terminated": time.strftime("%Y-%m-%dT%H:%M:%S"), "http": code})
        sys.exit(0 if code in (200, 202, 204) else 1)

    p = payload(a)
    show = json.loads(json.dumps(p))
    show["env"] = {"PUBLIC_KEY": (p["env"]["PUBLIC_KEY"][:24] + "...") if p["env"]["PUBLIC_KEY"] else ""}
    if a.launch:
        if not a.i_confirm_cost:
            sys.exit("--launch needs --i-confirm-cost (the pod bills per hour until terminated).")
        # a failed attempt creates nothing and costs nothing; the attempts go from the cheapest setting to the dearest and stop at the first that works
        combos = [(a.cloud, True)] if not a.try_all else [("COMMUNITY", True), ("COMMUNITY", False), ("SECURE", True), ("SECURE", False)]
        code, pod = 0, {}
        for cloud, ram_hint in combos:
            q = dict(p, cloudType=cloud)
            if ram_hint and not a.no_ram_hint:
                q["minRAMPerGPU"], q["minVCPUPerGPU"] = a.min_ram_gb, a.min_vcpu
            else:
                q.pop("minRAMPerGPU", None), q.pop("minVCPUPerGPU", None)
            code, pod = call("POST", "/pods", q)
            print(f"attempt cloud={cloud} ram-floor={'%d GB' % a.min_ram_gb if 'minRAMPerGPU' in q else 'none'}: HTTP {code} {'' if code in (200, 201) else json.dumps(pod)[:160]}", flush=True)
            if code in (200, 201):
                break
        if code not in (200, 201):
            sys.exit(1)
        s = summarize(pod)
        save_record({"id": pod.get("id"), "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": a.name, "payload": show, "created_response": s})
        print(json.dumps(s, indent=1))
        print(f"\nPod {pod.get('id')} is BILLING. Next: --status {pod.get('id')} --wait   (and TERMINATE it when the outputs are copied out)")
        return
    print("=== DRY RUN: no pod created, nothing spent ===")
    print(json.dumps(show, indent=2))


if __name__ == "__main__":
    main()
