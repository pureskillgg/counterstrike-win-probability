import json, collections, polars as pl
G="G:/pureskill/csds-cs2/data/combined"
D="C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct"
def day(k):
    p=k.split("/"); return "-".join(p[1:4])
te=set(json.load(open(D+"/training_exclude_copy.json"))["match_keys"])
man=json.load(open(G+"/split_manifest.json"))["assign"]
bk=json.load(open(G+"/_backup_pre_22_26_20261027/split_manifest.json"))["assign"]
ext=set(pl.scan_parquet(G+"/cache_snapshot_2026_ext.parquet").select("match_key").unique().collect()["match_key"].to_list())
exp=set(pl.scan_parquet(G+"/cache_snapshot_2026_oct23_31.parquet").select("match_key").unique().collect()["match_key"].to_list())
ext21={k for k in ext if day(k)=="2026-10-21"}
print("ext Oct21 keys", len(ext21), "all in exclusion list", ext21<=te, "all in backup Oct21 train", all(bk.get(k)=="train" for k in ext21))
print("exclusion list keys in ext cache", len(te & ext), "in exp cache", len(te & exp))
print("exclusion keys in backup manifest", sum(1 for k in te if k in bk), "backup train", sum(1 for k in te if bk.get(k)=="train"))
print("backup Oct21 train keys not in ext cache:", [k for k,v in bk.items() if day(k)=="2026-10-21" and v=="train" and k not in ext])
print("manifest Oct21 train keys:", sum(1 for k,v in man.items() if day(k)=="2026-10-21" and v=="train"))
print("exclusion list vs manifest Oct21-25 train+val:", te==set(k for k,v in man.items() if v in ("train","val") and "2026-10-21"<=day(k)<="2026-10-25"))
# ext Oct14-20 train rows vs exclusion (should be zero)
print("exclusion keys dated Oct 14-20:", sum(1 for k in te if "2026-10-14"<=day(k)<="2026-10-20"))
