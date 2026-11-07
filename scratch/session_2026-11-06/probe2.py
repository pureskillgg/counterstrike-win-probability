import json, subprocess
F = "/home/sagemaker-user/adam/builder/final"
m = json.load(open(F+"/split_manifest_v2.json"))
print("manifest top keys:", list(m.keys())[:20])
for k in list(m.keys())[:20]:
    v = m[k]
    if isinstance(v, dict):
        print(" ", k, "dict keys:", list(v.keys())[:12], "len", len(v))
    elif isinstance(v, list):
        print(" ", k, "list len", len(v), "sample", str(v[:2])[:200])
    else:
        print(" ", k, repr(v)[:200])
hb = [l.strip() for l in open(F+"/holdout_b_keys.txt") if l.strip()]
print("holdout_b_keys.txt lines", len(hb), "sample", hb[:2])
r = subprocess.run(["aws","s3","cp","s3://pureskillgg-csgo-production-sagemaker-0/adam/tomes/holdout_2026-10-26_31/match_keys.txt","-"], capture_output=True)
print("s3 rc", r.returncode, "bytes", len(r.stdout), "stderr", r.stderr[:200])
ho = [l.strip() for l in r.stdout.decode().splitlines() if l.strip()]
print("holdout tome keys", len(ho), "unique", len(set(ho)), "sample", ho[:2])
json.dump(ho, open("/dev/null","w"))
