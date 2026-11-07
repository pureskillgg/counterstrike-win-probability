import polars as pl, json, collections, numpy as np, glob
B = "/home/sagemaker-user/adam/builder"; F = B + "/final"
bm = pl.read_parquet(F+"/bucket_matches.parquet"); lm = pl.read_parquet(F+"/laptop_matches.parquet"); cm = pl.read_parquet(F+"/clean_matches.parquet")
BK = set(bm["match_key"].to_list()); LK = set(lm["match_key"].to_list()); CK = set(cm["match_key"].to_list())
EK = set(json.load(open(F+"/export_clean_v2/match_keys.json")))
HB = set(l.strip() for l in open(F+"/holdout_b_keys.txt") if l.strip())
v1 = json.load(open(F+"/split_manifest.json"))["assign"]
mp = {}
for df in (bm, lm):
    for k, m in zip(df["match_key"].to_list(), df["map_name"].to_list()): mp[k] = m
MIR1 = {k for k,s in v1.items() if s=='test' and mp.get(k)=='de_mirage'}
usable_l = lm.filter(pl.col("dup_keeper") & pl.col("in_scope") & (pl.col("n_ok_rounds")>0))["match_key"].to_list()
LOK = set(usable_l)
lap_only_nonHT = (LK - BK) - (set(l for l in LK if False))
HT = set(l.strip() for l in __import__('subprocess').run(["aws","s3","cp","s3://pureskillgg-csgo-production-sagemaker-0/adam/tomes/holdout_2026-10-26_31/match_keys.txt","-"],capture_output=True).stdout.decode().splitlines() if l.strip())
lo_nh = (LK - BK) - HT
print("laptop-only non-HT", len(lo_nh), "usable (laptop flags)", len(lo_nh & LOK), "usable in BK?", len(LOK & BK))
exp_side = (EK - BK) | (MIR1 & (LK - BK))
print("exp side (EK-BK) + MIR1 laptop-only", len(exp_side), "| usable-not-explained", len((lo_nh & LOK) - exp_side), "| explained-not-usable", len(exp_side - (lo_nh & LOK)))
unexp = list((lo_nh & LOK) - exp_side)
for k in unexp:
    r = lm.filter(pl.col("match_key")==k).row(0, named=True)
    print("UNEXPLAINED laptop usable key", k, {c: r[c] for c in ["map_name","match_date","in_scope","dup_keeper","n_ok_rounds","dup_group","dup_group_size","sig_hash","scope_reason","n_rounds_re"] if c in r}, "in CK", k in CK, "in MIR1", k in MIR1, "in HB", k in HB)
b_nc = [k for k in BK if k not in CK and k not in (HB | MIR1)]
print("BK not in CK and not excluded", len(b_nc))
for k in b_nc:
    r = bm.filter(pl.col("match_key")==k).row(0, named=True)
    print("  BK key", k, {c: r[c] for c in ["map_name","match_date","in_scope","dup_keeper","n_ok_rounds","dup_group","dup_group_size","scope_reason"] if c in r}, "in LK", k in LK)
# laptop-only key not in CK and not excluded
lo_notCK = [k for k in LK - BK if k not in CK and k not in HT and k not in MIR1 and k not in HB]
print("laptop-only not in CK, not HT/MIR1/HB", len(lo_notCK))
for k in lo_notCK:
    r = lm.filter(pl.col("match_key")==k).row(0, named=True)
    print("  LO key", k, {c: r[c] for c in ["map_name","match_date","in_scope","dup_keeper","n_ok_rounds","dup_group","dup_group_size","scope_reason"] if c in r})
# zero-keeper groups
g = cm.group_by("gkey").agg(pl.len().alias("n"), pl.col("dup_keeper").sum().alias("nk"), pl.col("in_scope").sum().alias("nin"), pl.col("n_ok_rounds").gt(0).sum().alias("nok")).sort("gkey")
print("clean_matches groups", g.height, "keeper count dist", g.group_by("nk").len().sort("nk").rows())
zero = g.filter(pl.col("nk")==0)
print("zero-keeper groups", zero.height, "size dist", zero.group_by("n").len().rows())
zk = cm.filter(pl.col("gkey").is_in(zero["gkey"].to_list()))
print("members of zero-keeper groups", zk.height, "src", zk.group_by("src").len().rows(), "in_scope&n_ok>0", zk.filter(pl.col("in_scope") & (pl.col("n_ok_rounds")>0)).height)
print("zero-keeper groups members in_scope&n_ok>0 src", zk.filter(pl.col("in_scope") & (pl.col("n_ok_rounds")>0)).group_by("src").len().rows())
# sig_hash partners for those groups
allm = pl.concat([bm.select(["match_key","sig_hash","map_name"]), lm.select(["match_key","sig_hash","map_name"])]).unique("match_key")
sigs = set(zk["match_key"].to_list())
sh = allm.filter(pl.col("match_key").is_in(list(sigs)))["sig_hash"].to_list()
partners = allm.filter(pl.col("sig_hash").is_in(sh) & ~pl.col("match_key").is_in(list(sigs)))
print("sig partners outside zero-keeper groups", partners.height, "in HB", partners.filter(pl.col("match_key").is_in(list(HB))).height, "in MIR1", partners.filter(pl.col("match_key").is_in(list(MIR1))).height, "in EK", partners.filter(pl.col("match_key").is_in(list(EK))).height)
# prep report totals
tot = collections.Counter(); usesum = 0; rt = 0; st = collections.Counter(); matches = 0
for c in range(68):
    rep = json.load(open(f"{B}/data/c{c:03d}/prep_report.json"))
    rt += rep.get("rounds_total", 0); usesum += rep.get("use_matches", 0); matches += rep.get("matches", 0)
    for k, v in rep.get("round_status", {}).items(): st[k.split("|")[-1]] += v
print("prep rounds_total sum", rt, "use_matches sum", usesum, "matches sum", matches)
print("prep round_status sums", dict(st), "sum", sum(st.values()))
cr = pl.read_parquet(F+"/clean_rounds.parquet", columns=["match_key","round","src","round_status","use"])
print("clean_rounds by src", cr.group_by("src").len().rows())
print("clean_rounds bucket-src round_status", cr.filter(pl.col("src")=="bucket").group_by("round_status").len().sort("len", descending=True).rows()[:10])
print("clean_rounds bucket-src use True", cr.filter((pl.col("src")=="bucket") & pl.col("use")).height)
