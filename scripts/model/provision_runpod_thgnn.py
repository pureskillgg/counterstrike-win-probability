#!/usr/bin/env python3
"""
Provision (or terminate) a RunPod GPU pod for the temporal-model runs (GRU and THGNN-lite).

Same pattern as the dnabert project's provision_runpod_caduceus.py: RunPod REST API (https://rest.runpod.io/v1), bearer auth,
a secure (non-spot) pod, a public PyTorch devel image, a volume mounted at /workspace, and SSH via the PUBLIC_KEY env var.
Uses only the standard library, so it runs in any Python.

The key is read from a .env file (default: the Robotics project's .env, which already holds RUNPOD_KEY; nothing is copied).
It is never printed, logged, or passed on the command line.

Usage:
    python provision_runpod_thgnn.py --dry-run          # print the request, create nothing
    python provision_runpod_thgnn.py                    # create the pod (COSTS MONEY)
    python provision_runpod_thgnn.py --status POD_ID    # print id, status and ssh host/port (no key)
    python provision_runpod_thgnn.py --terminate POD_ID # delete the pod (stops billing)
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"
DEFAULT_ENV = Path("C:/Users/<user>/Documents/PhD/Robotics_ABE6399/advanced_perception_stuff/.env")
GPU_TYPE_IDS = ["NVIDIA RTX A6000"]   # the same tier the dnabert project chose: about 2x the A40's FP16 throughput
CLOUD_TYPE = "SECURE"                 # non-spot, so a pod is not reclaimed mid-run
DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"  # public image, no registry auth
CONTAINER_DISK_GB = 20
VOLUME_GB = 40                        # inputs about 6 GB, the repo and outputs; headroom for pip and checkpoints
VOLUME_MOUNT_PATH = "/workspace"
SSH_PUBLIC_KEY_PATH = Path.home() / ".ssh" / "id_ed25519.pub"


def read_key(env_path):
    """RUNPOD_KEY from a .env file. The value is returned, never printed."""
    if not env_path.exists():
        sys.exit(f"{env_path} not found; aborting")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("RUNPOD_KEY="):
            return line.partition("=")[2].strip()
    sys.exit(f"RUNPOD_KEY not found in {env_path}; aborting")


def request(method, path, key, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{API}{path}", data=data, method=method,
        # an explicit User-Agent: RunPod's Cloudflare front rejects urllib's default signature with error 1010
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "pureskill-provision/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]  # the RunPod error text; it never contains the key
        sys.exit(f"RunPod {method} {path} failed: HTTP {exc.code} {exc.reason}: {detail}")


def build_payload(name, gpu_types=None):
    ssh_pub_key = SSH_PUBLIC_KEY_PATH.read_text(encoding="utf-8").strip() if SSH_PUBLIC_KEY_PATH.exists() else ""
    if not ssh_pub_key:
        print(f"WARNING: no SSH public key at {SSH_PUBLIC_KEY_PATH}; only the web terminal will reach the pod.", file=sys.stderr)
    return {
        "name": name,
        "imageName": DOCKER_IMAGE,
        "gpuTypeIds": gpu_types or GPU_TYPE_IDS,
        "cloudType": CLOUD_TYPE,
        "containerDiskInGb": CONTAINER_DISK_GB,
        "volumeInGb": VOLUME_GB,
        "volumeMountPath": VOLUME_MOUNT_PATH,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": ssh_pub_key},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the request payload without creating a pod")
    ap.add_argument("--env", type=Path, default=DEFAULT_ENV, help=".env file holding RUNPOD_KEY")
    ap.add_argument("--name", default="pureskill-temporal")
    ap.add_argument("--cpu", action="store_true", help="create a CPU-only pod (cheap staging: holds data, runs no GPU job)")
    ap.add_argument("--cpu-flavor", default="cpu3c", help="CPU flavour id for --cpu: cpu3c, cpu3g, cpu3m, cpu5c, cpu5g or cpu5m")
    ap.add_argument("--vcpus", type=int, default=2, help="vCPU count for --cpu")
    ap.add_argument("--gpu", action="append", metavar="GPU_TYPE_ID",
                    help="RunPod GPU type id to request, repeatable, in preference order (default: NVIDIA RTX A6000)")
    ap.add_argument("--status", metavar="POD_ID", help="print the pod's status and ssh endpoint")
    ap.add_argument("--terminate", metavar="POD_ID", help="delete the pod")
    ap.add_argument("--get", metavar="PATH", help="read-only: print the JSON the RunPod API returns for PATH, e.g. /networkvolumes")
    ap.add_argument("--create-volume", metavar="NAME", help="create a network volume of --volume-size GB in --datacenter (COSTS MONEY)")
    ap.add_argument("--volume-size", type=int, default=40, help="network volume size in GB for --create-volume")
    ap.add_argument("--datacenter", help="RunPod data center id; required for --create-volume, and the pod is pinned to it")
    ap.add_argument("--volume-id", help="attach this network volume to the new pod (the pod must be in the volume's data center)")
    a = ap.parse_args()

    # RUNPOD_API_KEY in the environment takes precedence; the .env file is the fallback (the key is never printed either way).
    key = os.environ.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_KEY") or read_key(a.env)
    if a.get:
        print(json.dumps(request("GET", a.get, key), indent=2))
        return
    if a.create_volume:
        if not a.datacenter:
            sys.exit("--create-volume needs --datacenter")
        vol = request("POST", "/networkvolumes", key, {"name": a.create_volume, "size": a.volume_size, "dataCenterId": a.datacenter})
        print(json.dumps(vol, indent=2))
        return
    if a.status:
        pod = request("GET", f"/pods/{a.status}", key)
        print(json.dumps({k: pod.get(k) for k in ("id", "name", "desiredStatus", "gpu", "machine", "ports", "publicIp", "portMappings")}, indent=2))
        return
    if a.terminate:
        request("DELETE", f"/pods/{a.terminate}", key)
        print(f"pod {a.terminate} deleted")
        return

    payload = build_payload(a.name, a.gpu)
    if a.volume_id:
        payload["networkVolumeId"] = a.volume_id
        payload.pop("volumeInGb", None)  # the attached network volume replaces the pod's own volume; /workspace is the mount
    if a.datacenter:
        payload["dataCenterIds"] = [a.datacenter]
    if a.cpu:
        # CPU-only pod for staging data (cheap); no GPU fields. Flavour ids per the RunPod REST docs (POST /pods, computeType CPU).
        payload.pop("gpuTypeIds", None)
        payload.update({"computeType": "CPU", "cpuFlavorIds": [a.cpu_flavor], "cpuFlavorPriority": "availability",
                        "vcpuCount": a.vcpus})
    if a.dry_run:
        print("=== DRY RUN: no pod created ===")
        print(json.dumps(payload, indent=2))
        return
    pod = request("POST", "/pods", key, payload)
    print(json.dumps({k: pod.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "gpu", "machine")}, indent=2))
    print(f"\nPod id: {pod.get('id')}  (check with --status {pod.get('id')}; delete with --terminate {pod.get('id')})")


if __name__ == "__main__":
    main()
