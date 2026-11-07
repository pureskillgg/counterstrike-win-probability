import boto3, json, time, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config

B = "pureskillgg-csgo-production-dataexchange-csds-unredacted-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
cfg = Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=64)
targets = {}
for r in csv.DictReader(open("/home/sagemaker-user/adam/builder/fixes/fix332/undetermined_matches.csv")):
    targets[r["match_key"].split("/")[4]] = r["match_key"]


def cps(s3, prefix):
    out = []
    tok = None
    while True:
        kw = dict(Bucket=B, Prefix=prefix, Delimiter="/", MaxKeys=1000)
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        out += [x["Prefix"] for x in r.get("CommonPrefixes", [])]
        if r.get("IsTruncated"):
            tok = r["NextContinuationToken"]
        else:
            break
    return out


t0 = time.time()
s3 = boto3.client("s3", config=cfg)
months = ["csds/2026/08/", "csds/2026/09/", "csds/2026/10/", "csds/2026/11/"]
days = []
for m in months:
    days += cps(s3, m)
print("day prefixes", len(days), round(time.time() - t0, 1), flush=True)


def uuids_of(dp):
    return [(dp, u) for u in cps(boto3.client("s3", config=cfg), dp)]


rows = []
with ThreadPoolExecutor(max_workers=32) as ex:
    for res in ex.map(uuids_of, days):
        rows += res
print("uuid folders listed", len(rows), round(time.time() - t0, 1), flush=True)
with open(W + "skeptic_unred_folders.tsv", "w") as f:
    f.write("day_prefix\tfolder_prefix\n")
    for dp, fp in rows:
        f.write(f"{dp}\t{fp}\n")

# targets: check player_info presence and file-name sets for the folder
tgt_rows = [(u, mk) for u, mk in targets.items()]
found = {}


def chk(item):
    u, mk = item
    s3c = boto3.client("s3", config=cfg)
    p = mk.split("/")
    pfx = f"csds/{p[1]}/{p[2]}/{p[3]}/{u}/"
    names = []
    tok = None
    while True:
        kw = dict(Bucket=B, Prefix=pfx, MaxKeys=1000)
        if tok:
            kw["ContinuationToken"] = tok
        r = s3c.list_objects_v2(**kw)
        names += [x["Key"][len(pfx):] for x in r.get("Contents", [])]
        if r.get("IsTruncated"):
            tok = r["NextContinuationToken"]
        else:
            break
    return {"uuid": u, "key": mk, "n_files": len(names), "has_player_info": "player_info" in names, "names": sorted(names)}


with ThreadPoolExecutor(max_workers=32) as ex:
    res = list(ex.map(chk, tgt_rows))
summ = {
    "targets": len(res),
    "targets_with_folder": sum(1 for r in res if r["n_files"] > 0),
    "targets_with_player_info": sum(1 for r in res if r["has_player_info"]),
    "files_per_target_hist": {},
}
h = {}
for r in res:
    h[str(r["n_files"])] = h.get(str(r["n_files"]), 0) + 1
summ["files_per_target_hist"] = dict(sorted(h.items(), key=lambda kv: int(kv[0])))
summ["seconds"] = round(time.time() - t0, 1)
with open(W + "skeptic_unred_targets.json", "w") as f:
    json.dump({"summary": summ, "rows": res}, f)
print("SUMMARY", json.dumps(summ), flush=True)
print("UNRED_DONE", flush=True)
