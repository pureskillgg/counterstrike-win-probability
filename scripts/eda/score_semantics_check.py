import glob, os, pandas as pd
root = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
d = [x for x in os.listdir(root) if x.startswith("basic_round_state.")][0]
df = pd.read_parquet(sorted(glob.glob(os.path.join(root, d, "dataframe_*")))[0], columns=["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = df[df["event_type"] == "round_freeze_end"].sort_values(["match_key", "round", "tick"]).drop_duplicates(["match_key", "round"])
p = fe.pivot(index="match_key", columns="round", values=["ct_score", "t_score"])
have = p.dropna(subset=[("ct_score", 12), ("ct_score", 13), ("t_score", 12), ("t_score", 13)])
a = have[("ct_score", 13)] - have[("ct_score", 12)]   # change in the ct_score field across the half
b = have[("t_score", 13)] - have[("t_score", 12)]
team_attached = ((a.isin([0, 1])) & (b.isin([0, 1])) & ((a + b) == 1)).sum()
sw_a = have[("ct_score", 13)] - have[("t_score", 12)]   # if the fields swap at half, ct(13) continues from t(12)
sw_b = have[("t_score", 13)] - have[("ct_score", 12)]
side_attached = ((sw_a.isin([0, 1])) & (sw_b.isin([0, 1])) & ((sw_a + sw_b) == 1)).sum()
both = (((a.isin([0, 1])) & (b.isin([0, 1])) & ((a + b) == 1)) & ((sw_a.isin([0, 1])) & (sw_b.isin([0, 1])) & ((sw_a + sw_b) == 1))).sum()
print("matches with rounds 12 and 13:", len(have))
print("consistent with TEAM-attached fields (no swap at round 13):", int(team_attached))
print("consistent with SIDE-attached fields (swap at round 13):", int(side_attached))
print("consistent with both (ambiguous, e.g. tied scores):", int(both))
