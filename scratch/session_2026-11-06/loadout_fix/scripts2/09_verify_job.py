
# ---- verification on the space: per-match hashes (sample), parity rows vs written arrays, ranges, row counts ----
import json, sys, time, hashlib
import numpy as np
import polars as pl

T0 = time.time()


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, file=sys.stderr, flush=True)


EXP = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
STAGE = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout_stage/"
OUTD = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
keys = json.load(open(EXP + "match_keys.json"))
off = np.load(EXP + "offsets.npy").astype(np.int64)
rm = np.load(EXP + "round_match.npy").astype(np.int64)
NS = int(off[-1])
key_m = {k: i for i, k in enumerate(keys)}
SEL_KEYS = ['csds/2026/09/02/m_1bb933475ba53b592312040c/csds', 'csds/2026/09/02/m_2a99cefc92c6876d22f0690b/csds', 'csds/2026/09/02/m_ebfa08ccce5229aa4b72ccf2/csds', 'csds/2026/09/03/m_60e05d64f57824bee9e0348b/csds', 'csds/2026/09/04/m_2bc98015bac9cab355a1da42/csds', 'csds/2026/09/05/m_62a685d3c6491368c4cd9446/csds', 'csds/2026/09/05/m_6f98586af082bed26ae82cd7/csds', 'csds/2026/09/06/m_c7b4a6c7dad4b77acd16084e/csds', 'csds/2026/09/06/m_4a4903b052effe2d93c1a033/csds', 'csds/2026/09/06/m_4a804d15342657f4874c5acf/csds', 'csds/2026/09/06/m_41901db77d56500540c13171/csds', 'csds/2026/09/06/m_ead12bd0debc9a3605442947/csds', 'csds/2026/09/06/m_76876ea2dc863f311157fd7e/csds', 'csds/2026/09/06/m_bc04ca7d7484e393931cd5f1/csds', 'csds/2026/09/07/m_3ff8ad858aef8df499aa6936/csds', 'csds/2026/09/07/m_75067e60d4afa517b9d60c2f/csds', 'csds/2026/09/07/m_075259a6c9d49075c50fc303/csds', 'csds/2026/09/09/m_714adf173f9a116ba3d407c6/csds', 'csds/2026/09/09/m_0005a944f9fe18507707870c/csds', 'csds/2026/09/09/m_9248482304bb251e84c1d50f/csds', 'csds/2026/09/09/m_021095d6cdbdcc650d109e96/csds', 'csds/2026/09/10/m_1f6b28b10ea6a5dc0aa7444d/csds', 'csds/2026/09/10/m_49283d1de249725a33922ed8/csds', 'csds/2026/09/11/m_8632a99b5cb4e5d6d29172fb/csds', 'csds/2026/09/12/m_02bfa8e1820981f41c39ad77/csds', 'csds/2026/09/12/m_af4dc58fc9dab1dd5dbb2873/csds', 'csds/2026/09/13/m_ea88d632f9c66dc4f78a6b79/csds', 'csds/2026/09/15/m_b0e746672d9b80d1a6ec098a/csds', 'csds/2026/09/15/m_e0b2c2273d1ab5da9f27025e/csds', 'csds/2026/09/16/m_3cf9cf0c9c8619d2bc3f0f5f/csds', 'csds/2026/09/17/m_122b5190ef6d17618e4f73fa/csds', 'csds/2026/09/17/m_78e4e35220d88f76a13145ae/csds', 'csds/2026/09/18/m_daa0d173e152caede31fc59b/csds', 'csds/2026/09/18/m_85ca859b320ae344a157974e/csds', 'csds/2026/09/19/m_7a300a950fafb6c7764a4a31/csds', 'csds/2026/09/20/m_341eb274a3bb232a8abb7ad6/csds', 'csds/2026/09/20/m_83c2c394433dfbefdf730dc0/csds', 'csds/2026/09/20/m_31dd6cdebb0628a0b9544526/csds', 'csds/2026/09/21/m_3b3771070c6d6667c0929f62/csds', 'csds/2026/09/21/m_4d6e2fe0af0eecd4e06b3d9b/csds', 'csds/2026/09/21/m_6d0d0635367206babe90a00f/csds', 'csds/2026/09/23/m_160bed6354d7676c06df0e12/csds', 'csds/2026/09/23/m_b33c8d8db6fc526b4d017936/csds', 'csds/2026/09/23/m_5499ae2eb8f69a964b246add/csds', 'csds/2026/09/23/m_d1cba658be804227f296d34a/csds', 'csds/2026/09/25/m_348ee57d100658fc5bebbe19/csds', 'csds/2026/09/25/m_77337faf79df41322909d411/csds', 'csds/2026/09/28/m_8409d9cd99d7f712868e44b1/csds', 'csds/2026/09/28/m_fd92cec8bb102e43f92695e9/csds', 'csds/2026/09/28/m_28ac94691bcb8a856c4f606d/csds', 'csds/2026/09/30/m_7610c8bac631e464e69ab342/csds', 'csds/2026/10/03/m_5bdb2618ebdc2c58435b896c/csds', 'csds/2026/10/03/m_6d1c472562c712004cba52fe/csds', 'csds/2026/10/04/m_e526bc62e419e6ff059dbba0/csds', 'csds/2026/10/05/m_4400b1b948acf5c73988008a/csds', 'csds/2026/10/05/m_2fe8728688d9a0fc2caf6f24/csds', 'csds/2026/10/05/m_bd20ad7e3e8309a0d94fffb1/csds', 'csds/2026/10/06/m_f2f0844ebad023d10ddf14db/csds', 'csds/2026/10/06/m_edbbd043fc46df8289b5bc84/csds', 'csds/2026/10/06/m_87b9fe5582598dadc0a4109d/csds', 'csds/2026/10/08/m_1ea7fa019ad708ec9a1237dd/csds', 'csds/2026/10/08/m_289e9e3e2b1274d5f0b9acdc/csds', 'csds/2026/10/08/m_e49695e2da703f6dd9cd030f/csds', 'csds/2026/10/08/m_0098355f1eb7b12107bde318/csds', 'csds/2026/10/08/m_0c1857f0b74372398a46e03d/csds', 'csds/2026/10/08/m_6bdb156cec3c5f3c91fca34e/csds', 'csds/2026/10/08/m_cc089f797ffecc24fcf40c52/csds', 'csds/2026/10/08/m_3e663f05d4e243a7798f3f26/csds', 'csds/2026/10/08/m_e06366956353a139c2ee5101/csds', 'csds/2026/10/08/m_2e281b2c530f829fee17240e/csds', 'csds/2026/10/08/m_547b069a66025778af4cb9b6/csds', 'csds/2026/10/08/m_6ba4651d6a05090572561463/csds', 'csds/2026/10/08/m_c9fe5c093b02735dbae06986/csds', 'csds/2026/10/08/m_ed97405eb2cd769dd2824211/csds', 'csds/2026/10/08/m_6cbe5954c8ec6d2f93a36fbc/csds', 'csds/2026/10/08/m_3d6a135db79cc16acd2609b8/csds', 'csds/2026/10/08/m_4689eb6d733f6a634df098b5/csds', 'csds/2026/10/08/m_b94f5739003c0e022a3ca96f/csds', 'csds/2026/10/08/m_de0c9dd0c8d9f2fd316544c9/csds', 'csds/2026/10/08/m_7d043b9651ce8e44c9348e44/csds', 'csds/2026/10/08/m_75e8731b961a9a80a60ad861/csds', 'csds/2026/10/08/m_ffdbb81d1d4d0953a9c52075/csds', 'csds/2026/10/08/m_1691ebbeb5b68d049468081e/csds', 'csds/2026/10/08/m_4cbe1858dc5bae0b43f521ae/csds', 'csds/2026/10/08/m_9ad708bd899e6afccea5468c/csds', 'csds/2026/10/08/m_ff5c5ba15dea3b18d2fe8299/csds', 'csds/2026/10/09/m_1373b2cc1d709f5bce1be16e/csds', 'csds/2026/10/09/m_8c113158abf916569a5c6b1b/csds', 'csds/2026/10/09/m_c32cc94d832bb8a7873ccae7/csds', 'csds/2026/10/09/m_91b66fbda0840a78212cbc84/csds', 'csds/2026/10/09/m_cacab9f8e85f565c43a0f4b3/csds', 'csds/2026/10/09/m_9406a3ce720e3891f7dd0d0d/csds', 'csds/2026/10/09/m_0dc986245838d4371ed4ba81/csds', 'csds/2026/10/09/m_f4a92d516adce691a93858f5/csds', 'csds/2026/10/09/m_88aa16d0eaea658562084da3/csds', 'csds/2026/10/09/m_de584ce890518bf601d1c8ea/csds', 'csds/2026/10/09/m_1868eed09429b3adbc15286d/csds', 'csds/2026/10/09/m_bf50bb6231f116dcd14f4d4a/csds', 'csds/2026/10/09/m_168b0f0646f0e3872111ae45/csds', 'csds/2026/10/09/m_eb30432d15cef87a0deb2ef1/csds', 'csds/2026/10/09/m_752df0459155567ff971e077/csds', 'csds/2026/10/09/m_5b8f5847e3e241596b366a6d/csds', 'csds/2026/10/09/m_e95a615f079628bef69c08a9/csds', 'csds/2026/10/09/m_564c4ec65f6af3106fa44a8f/csds', 'csds/2026/10/09/m_0ce772d394416cb75053c4b4/csds', 'csds/2026/10/09/m_fa2fdf6ec2b504f636659e05/csds', 'csds/2026/10/09/m_72fc14de32dcbb7b079ba2eb/csds', 'csds/2026/10/09/m_755d1c13d14cc32ea6460ca7/csds', 'csds/2026/10/09/m_509b2ada3b079982d35031a3/csds', 'csds/2026/10/09/m_c7e2526c4162ede8f0d16787/csds', 'csds/2026/10/09/m_23d667577b60ca257da20664/csds', 'csds/2026/10/09/m_f80958cce10e932b147fca04/csds', 'csds/2026/10/09/m_4aa4e551b9e3174bac77dead/csds', 'csds/2026/10/09/m_5354a8ff6b749c209d4566bc/csds', 'csds/2026/10/09/m_61d0d3d03e6af06604f1c9ca/csds', 'csds/2026/10/10/m_8a597c276f031cc0e66373b9/csds', 'csds/2026/10/10/m_3b5016a160b7da1fb3487b21/csds', 'csds/2026/10/10/m_42dc67abcf745fa35f2fb421/csds', 'csds/2026/10/10/m_bf53d64d51c9c44f7a96ea01/csds', 'csds/2026/10/10/m_d3fd18592d289f2c424f9de9/csds', 'csds/2026/10/10/m_3c05d159fc957c67a0a02e09/csds', 'csds/2026/10/10/m_9514dfee0e787f317e616bcc/csds', 'csds/2026/10/10/m_b89c2371795509cb0127f4a0/csds', 'csds/2026/10/10/m_ff27e44cc27f2c26a76cc7e0/csds', 'csds/2026/10/10/m_0a144fad9d342effbe3d6df2/csds', 'csds/2026/10/10/m_01567b37746298154f5462c1/csds', 'csds/2026/10/10/m_5e9bc3fd9bacabd3bd54adfd/csds', 'csds/2026/10/10/m_1d257ab3efe761c79490508a/csds', 'csds/2026/10/10/m_6dc5b10dc8e83ffa86ab39bf/csds', 'csds/2026/10/10/m_541550a2b47d83d5ca7e7f8b/csds', 'csds/2026/10/10/m_dc5c81b69c5768660719e6f0/csds', 'csds/2026/10/10/m_3d611b09e787641ba95a13e2/csds', 'csds/2026/10/10/m_13221d5e2b3ec211269def43/csds', 'csds/2026/10/10/m_1f425b2d1904a02faeeffe1f/csds', 'csds/2026/10/10/m_61d0c4748b4f627fd10360cb/csds', 'csds/2026/10/10/m_ff7efbbee77feb49e4364566/csds', 'csds/2026/10/10/m_ba180a5da1439d6d2c174c65/csds', 'csds/2026/10/10/m_740a88b76e3057760427276b/csds', 'csds/2026/10/10/m_11bc6c2c111e7b8d9ecab713/csds', 'csds/2026/10/10/m_121c92ba18fb99f9990283cd/csds', 'csds/2026/10/10/m_d7676fd0caee6ab4ba8d566e/csds', 'csds/2026/10/10/m_a24bed67a2e14447976aa10a/csds', 'csds/2026/10/10/m_a9156edf674cf905c5399810/csds', 'csds/2026/10/10/m_15fe45baeb1c473afe6fe9c6/csds', 'csds/2026/10/11/m_9140923b643fca3b1499a9dd/csds', 'csds/2026/10/11/m_9aa50115fd582789877078ca/csds', 'csds/2026/10/11/m_bd9c2d8e56f773b8c4267e23/csds', 'csds/2026/10/11/m_28a26bcb5a173f28ce3b621f/csds', 'csds/2026/10/11/m_2e9b7f44cd543955dfab8539/csds', 'csds/2026/10/11/m_d81ba814d9719e54e8b9fcb0/csds', 'csds/2026/10/11/m_059f2416f30ddceda127dbf0/csds', 'csds/2026/10/11/m_98b669e3a55316d732023280/csds', 'csds/2026/10/11/m_d1f8e91bf1d510d4a23c3612/csds', 'csds/2026/10/11/m_5b1e7600a82c65a22e781842/csds', 'csds/2026/10/11/m_c5e96ffc3f4f836a879df66d/csds', 'csds/2026/10/11/m_1aebf5ca34a6474e7c16cca0/csds', 'csds/2026/10/11/m_b352a34bf1dfc512888565c0/csds', 'csds/2026/10/11/m_9bf0c01b76c10f1ba3aab097/csds', 'csds/2026/10/11/m_5e6d49596c9ee25c0c6f42c1/csds', 'csds/2026/10/11/m_95e389b512efedd45cf097e9/csds', 'csds/2026/10/11/m_2b15a2b053afa11b65d5a015/csds', 'csds/2026/10/11/m_aa08cefe2ed0a8779c01b835/csds', 'csds/2026/10/11/m_b60b9fa83e9d79757bce699d/csds', 'csds/2026/10/11/m_f5b49bfec81c91740655287b/csds', 'csds/2026/10/11/m_12789cf18902e2e01179260c/csds', 'csds/2026/10/11/m_e24c9922f5c8dc563073ff1c/csds', 'csds/2026/10/11/m_80a64eeb6f1946aca43458b0/csds', 'csds/2026/10/11/m_423981bd1fbffa818da413d0/csds', 'csds/2026/10/11/m_f8795b3c30cd145b2df8e65b/csds', 'csds/2026/10/11/m_1f098111388bd8cf40ba710d/csds', 'csds/2026/10/11/m_a4c902377d200bfc89fe8414/csds', 'csds/2026/10/11/m_2ea2045f6f9d5ba20b1659d7/csds', 'csds/2026/10/11/m_3bca7d90aff544c630699217/csds', 'csds/2026/10/11/m_026f41f7d4a6f522325b33cf/csds', 'csds/2026/10/11/m_84dc44d5d37c8a6cb6a4f532/csds', 'csds/2026/10/12/m_4b6d0188abe1c13be325171d/csds', 'csds/2026/10/12/m_5002ea405ebe4e9b21a391d8/csds', 'csds/2026/10/12/m_85d198cad04b96691be8fe28/csds', 'csds/2026/10/12/m_6824af825c93a330c8812664/csds', 'csds/2026/10/12/m_df0f4d13483b093dbf833aaa/csds', 'csds/2026/10/12/m_dc4fd7ea179f7b8d100924c9/csds', 'csds/2026/10/12/m_6a5b5843f5136c310a39908d/csds', 'csds/2026/10/20/m_f3898e2cf9b5d8a3fdbb768c/csds', 'csds/2026/10/20/m_0eeefc310f48d572a7ee7c27/csds', 'csds/2026/10/20/m_f6231fd390c9076b0fa94eb4/csds', 'csds/2026/10/20/m_d32db025d173abb52752f8b2/csds', 'csds/2026/10/20/m_6eff6b816481778213cb3535/csds', 'csds/2026/10/20/m_24913dcb3188ed5f647c5da9/csds', 'csds/2026/10/20/m_be2ffd4abe2d9cbc0e9094eb/csds', 'csds/2026/10/20/m_4419e4ec7788342d6537aa75/csds', 'csds/2026/10/20/m_35fb66959ac8775154eaf17a/csds', 'csds/2026/10/20/m_d9f7273e1736bc5aad0c1da9/csds', 'csds/2026/10/20/m_c6b0164eacc6c4535ce35298/csds', 'csds/2026/10/20/m_4868fa95dc84d8700e5e1552/csds', 'csds/2026/10/20/m_b99802754b57300ac4dbd246/csds', 'csds/2026/10/20/m_309163410093f386c045d12c/csds', 'csds/2026/10/20/m_68d9413c7d857d8760e6d532/csds', 'csds/2026/10/20/m_9dd05f6c8db74ac3557cefb5/csds', 'csds/2026/10/20/m_606f076ff486ef627c2e2dd3/csds', 'csds/2026/10/20/m_6c20b46387382b94ce4f8c26/csds']
arr16 = np.load(OUTD + "loadout_step_i16.npy", mmap_mode="r")
arr_eq = np.load(OUTD + "loadout_equip_i32.npy", mmap_mode="r")
assert arr16.shape == (NS, 2, 19) and arr_eq.shape == (NS, 2)
res = {"hashes": {}}
rounds_by_m = {}
order = np.argsort(rm, kind="stable")
rm_sorted = rm[order]
for mk in SEL_KEYS:
    m = key_m[mk]
    lo = np.searchsorted(rm_sorted, m, side="left")
    hi = np.searchsorted(rm_sorted, m, side="right")
    rr = np.sort(order[lo:hi])
    S_all = np.concatenate([np.arange(off[r], off[r + 1]) for r in rr])
    a = np.ascontiguousarray(np.asarray(arr16[S_all]).astype(np.int16)).tobytes()
    b = np.ascontiguousarray(np.asarray(arr_eq[S_all]).astype(np.int32)).tobytes()
    res["hashes"][mk] = hashlib.sha256(a + b).hexdigest()
log("hashes done", len(res["hashes"]))
# parity rows written by the S3 job vs the arrays as written
pr = pl.read_parquet(STAGE + "parity_s3_rows.parquet")
FIELDS19 = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle",
            "sniper_other", "smg", "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy"]
pk = sorted(pr["match_key"].unique().to_list())
cells = bad = 0
bad_keys = []
for mk in pk:
    m = key_m[mk]
    lo = np.searchsorted(rm_sorted, m, side="left")
    hi = np.searchsorted(rm_sorted, m, side="right")
    rr = np.sort(order[lo:hi])
    S_all = np.concatenate([np.arange(off[r], off[r + 1]) for r in rr])
    sub = pr.filter(pl.col("match_key") == mk)
    pos = np.searchsorted(S_all, sub["s"].to_numpy())
    assert (S_all[pos] == sub["s"].to_numpy()).all()
    dense = np.zeros((len(S_all), 2, 20), dtype=np.int64)
    si = sub["side_idx"].to_numpy()
    dense[pos[:, None], si[:, None], np.arange(20)[None, :]] = np.stack([sub[c].to_numpy() for c in FIELDS19 + ["equip"]], axis=1)
    got = np.zeros((len(S_all), 2, 20), dtype=np.int64)
    got[:, :, :19] = np.asarray(arr16[S_all]).astype(np.int64)
    got[:, :, 19] = np.asarray(arr_eq[S_all]).astype(np.int64)
    d = got != dense
    cells += d.shape[0] * 2
    bad += int(d.any(axis=2).sum())
    if d.any():
        bad_keys.append(mk)
res["parity_rows_vs_written"] = {"keys": len(pk), "side_steps": cells, "mismatch_side_steps": bad, "mismatch_keys": bad_keys[:10]}
log("parity vs written", res["parity_rows_vs_written"])
# ranges and row counts over the whole array
CH = 4_000_000
mx = np.full((2, 19), -10**9, dtype=np.int64)
mn = np.full((2, 19), 10**9, dtype=np.int64)
eq_mx = np.full(2, -10**9, dtype=np.int64)
rows_with_data = np.zeros(2, dtype=np.int64)
steps_with_data = 0
for i in range(0, NS, CH):
    blk = np.asarray(arr16[i:i + CH]).astype(np.int64)
    eb = np.asarray(arr_eq[i:i + CH]).astype(np.int64)
    mx = np.maximum(mx, blk.max(axis=0))
    mn = np.minimum(mn, blk.min(axis=0))
    eq_mx = np.maximum(eq_mx, eb.max(axis=0))
    nz = (blk != 0).any(axis=2) | (eb != 0)
    rows_with_data += nz.sum(axis=0)
    steps_with_data += int(nz.any(axis=1).sum())
res["ranges"] = {"min": mn.tolist(), "max": mx.tolist(), "equip_max": eq_mx.tolist()}
res["rows_with_data_CT_T"] = rows_with_data.tolist()
res["steps_with_any_data"] = steps_with_data
res["total_steps"] = NS
log("ranges and counts", json.dumps(res["ranges"]), res["rows_with_data_CT_T"], steps_with_data)
print("#RESULT " + json.dumps(res), flush=True)
