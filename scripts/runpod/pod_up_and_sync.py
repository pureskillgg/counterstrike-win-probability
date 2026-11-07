#!/usr/bin/env python3
"""One entry point for the RunPod work pod: reuse a RUNNING pod (or create one), wait for SSH, copy the S3 profile in,
install the AWS CLI and start the sync of csds/ from the network volume hyt5kma12z (US-IL-1) to /workspace/csds on the pod.

  python scripts/runpod/pod_up_and_sync.py                     # reuse the running pod, or create one (COSTS MONEY), then sync
  python scripts/runpod/pod_up_and_sync.py --no-create         # only use a pod that is already RUNNING; never creates one
  python scripts/runpod/pod_up_and_sync.py --status            # list running pods and the ssh command (free, read-only)
  python scripts/runpod/pod_up_and_sync.py --terminate POD_ID  # delete the pod: billing ends (copy results out first)

The RunPod key is read by provision_runpod_cs2.py from its env file. The S3 key is never printed and never written to the pod's
environment or to disk on this PC: only the [runpod] section of ~/.aws/credentials is streamed over ssh stdin into ~/.aws/credentials
on the pod. Sync progress: ssh into the pod and run `tail -n 3 /workspace/sync.log`.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROVISION = HERE / "provision_runpod_cs2.py"
AWS_CREDENTIALS = Path.home() / ".aws" / "credentials"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
S3_SOURCE = "s3://hyt5kma12z/csds/"
S3_REGION = "US-IL-1"
S3_ENDPOINT = "https://s3api-us-il-1.runpod.io/"
POD_DISK_GB = 40
POD_NAME = "cs2-graph-transformer"


def provision(*args):
    """Run provision_runpod_cs2.py and return its stdout. The key is read inside that script, never here."""
    done = subprocess.run([sys.executable, str(PROVISION), *args], capture_output=True, text=True, timeout=900)
    if done.returncode != 0 and not done.stdout.strip():
        sys.exit(f"provision_runpod_cs2.py {' '.join(args)} failed: {done.stderr.strip()[:300]}")
    return done.stdout


def json_lines(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


def running_pods():
    return [p for p in json_lines(provision("--list")) if p.get("status") == "RUNNING"]


def create_pod():
    out = provision("--launch", "--i-confirm-cost", "--volume-gb", str(POD_DISK_GB), "--no-ram-hint", "--name", POD_NAME)
    found = re.search(r"Pod (\w+) is BILLING", out)
    if not found:
        sys.exit("No pod was created. Every RunPod community GPU was out of stock for this request; try again later.\n" + out[-400:])
    return found.group(1)


def wait_for_ssh(pod_id):
    last = {}
    for line in provision("--status", pod_id, "--wait", "--wait-min", "15").splitlines():
        if line.startswith("{"):
            last = json.loads(line)
    if not last.get("publicIp") or not last.get("ssh_port"):
        sys.exit(f"Pod {pod_id} did not expose SSH within 15 minutes. Last status: {last}")
    return last["publicIp"], int(last["ssh_port"])


def ssh(ip, port, command, stdin_text=None, timeout=300):
    argv = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20", "-i", str(SSH_KEY), "-p", str(port), f"root@{ip}", command]
    return subprocess.run(argv, input=stdin_text, capture_output=True, text=True, timeout=timeout)


def runpod_profile_section():
    """The [runpod] block of ~/.aws/credentials, returned as text for ssh stdin only. Never printed."""
    if not AWS_CREDENTIALS.exists():
        sys.exit(f"{AWS_CREDENTIALS} not found")
    keep, text = False, []
    for line in AWS_CREDENTIALS.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("["):
            keep = line.strip() == "[runpod]"
        if keep:
            text.append(line)
    if not text:
        sys.exit("No [runpod] profile in ~/.aws/credentials")
    return "\n".join(text) + "\n"


def install_profile(ip, port):
    done = ssh(ip, port, "mkdir -p ~/.aws && cat > ~/.aws/credentials && chmod 600 ~/.aws/credentials", stdin_text=runpod_profile_section(), timeout=120)
    if done.returncode != 0:
        sys.exit(f"Could not install the S3 profile on the pod: {done.stderr.strip()[:300]}")


def install_cli(ip, port):
    done = ssh(ip, port, "pip install -q awscli && aws --version", timeout=600)
    if done.returncode != 0:
        sys.exit(f"awscli install failed: {done.stderr.strip()[:300]}")
    print(done.stdout.strip())


def start_sync(ip, port):
    """Start the copy detached on the pod, so it keeps running if this PC disconnects."""
    command = (f"mkdir -p /workspace/csds && nohup setsid aws s3 sync {S3_SOURCE} /workspace/csds/ --profile runpod "
               f"--region {S3_REGION} --endpoint-url {S3_ENDPOINT} > /workspace/sync.log 2>&1 < /dev/null &")
    done = ssh(ip, port, command, timeout=120)
    if done.returncode != 0:
        sys.exit(f"Could not start the sync: {done.stderr.strip()[:300]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-create", action="store_true", help="use a RUNNING pod only; never create one")
    ap.add_argument("--status", action="store_true", help="list running pods and exit (read-only)")
    ap.add_argument("--terminate", metavar="POD_ID", help="delete a pod (billing ends)")
    args = ap.parse_args()

    if args.terminate:
        print(provision("--terminate", args.terminate))
        return 0
    pods = running_pods()
    if args.status:
        for p in pods:
            print(json.dumps(p))
        if not pods:
            print("No RUNNING pod.")
        return 0

    if pods:
        pod_id = pods[0]["id"]
        if len(pods) > 1:
            print(f"Warning: {len(pods)} running pods; using {pod_id}. Terminate the others with --terminate.")
        print(f"Reusing running pod {pod_id}.")
    elif args.no_create:
        sys.exit("No RUNNING pod and --no-create was given.")
    else:
        pod_id = create_pod()
        print(f"Created pod {pod_id} (billing from now until it is terminated).")

    ip, port = wait_for_ssh(pod_id)
    install_profile(ip, port)
    install_cli(ip, port)
    start_sync(ip, port)
    print(f"\nPod {pod_id} is up. Sync of {S3_SOURCE} started on the pod (about 3 minutes at ~85 MB/s).")
    print(f"Connect:      ssh -i ~/.ssh/id_ed25519 -p {port} root@{ip}")
    print("Sync status:  tail -n 3 /workspace/sync.log")
    print(f"Stop billing: python scripts/runpod/pod_up_and_sync.py --terminate {pod_id}   (copy results out first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
