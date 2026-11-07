import json, glob, collections
import numpy as np
import pyarrow.parquet as pq

B = "/home/sagemaker-user/adam/builder"
FIN = B + "/final"
EXP = FIN + "/export_clean_v2"

mk = json.load(open(EXP + "/match_keys.json"))
mks = set(mk)
rm = np.asarray(np.load(EXP + "/round_match.npy"))
rn = np.asarray(np.load(EXP + "/round_num.npy"))
po = np.asarray(np.load(EXP + "/pos_offsets.npy"))
r_rows = np.diff(po)

ex = {}
for r in range(len(rm)):
    ex[(mk[rm[r]], int(rn[r]))] = int(r_rows[r])
print("EXPORT (match, round) pairs", len(ex))

pp = {}
for p in sorted(glob.glob(FIN + "/pos_parts/bucket_c*.parquet")):
    t = pq.read_table(p, columns=["match_key", "round"])
    g = t.group_by(["match_key", "round"]).aggregate([("round", "count")]).to_pydict()
    for k, r, c in zip(g["match_key"], g["round"], g["round_count"]):
        kk = (k, int(r))
        pp[kk] = pp.get(kk, 0) + int(c)
print("POS_PARTS (match, round) pairs", len(pp), "rows", sum(pp.values()))

pp_keys = set(k for k, _ in pp)
ex_A = {kk: v for kk, v in ex.items() if kk[0] in pp_keys}
pp_A = pp
print("GROUP A export (match, round) pairs", len(ex_A), "pos rows", sum(ex_A.values()))
print("GROUP A pos_parts (match, round) pairs", len(pp_A), "rows", sum(pp_A.values()))

pp_only = [kk for kk in pp_A if kk not in ex_A]
ex_only = [kk for kk in ex_A if kk not in pp_A]
both = [kk for kk in pp_A if kk in ex_A]
same = [kk for kk in both if pp_A[kk] == ex_A[kk]]
diff = [kk for kk in both if pp_A[kk] != ex_A[kk]]
print("ROUNDS pos_parts only (absent from export)", len(pp_only), "rows", sum(pp_A[kk] for kk in pp_only))
print("ROUNDS export only (absent from pos_parts)", len(ex_only), "rows", sum(ex_A[kk] for kk in ex_only))
print("ROUNDS both same count", len(same), "| both differ count", len(diff))
if diff:
    d = [ex_A[kk] - pp_A[kk] for kk in diff]
    print("DIFF export-minus-pp: sum", sum(d), "positive", sum(1 for x in d if x > 0),
          "negative", sum(1 for x in d if x < 0), "min", min(d), "max", max(d))
    print("DIFF sample (key, round, pp, export)", [(kk[0][-30:], kk[1], pp_A[kk], ex_A[kk]) for kk in diff[:5]])
    print("DIFF distinct matches affected", len(set(kk[0] for kk in diff)))
print("POS_PARTS rounds for keys not in export (47 excluded matches):",
      sum(1 for kk in pp_A if kk[0] not in mks), "rows", sum(pp_A[kk] for kk in pp_A if kk[0] not in mks))
