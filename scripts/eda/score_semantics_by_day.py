import glob, os, pandas as pd
root = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
d = [x for x in os.listdir(root) if x.startswith("basic_round_state.")][0]
df = pd.read_parquet(sorted(glob.glob(os.path.join(root, d, "dataframe_*")))[0], columns=["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = df[df["event_type"] == "round_freeze_end"].sort_values(["match_key", "round", "tick"]).drop_duplicates(["match_key", "round"])
p = fe.pivot(index="match_key", columns="round", values=["ct_score", "t_score"])
h = p.dropna(subset=[("ct_score", 12), ("ct_score", 13), ("t_score", 12), ("t_score", 13)]).copy()
a = h[("ct_score", 13)] - h[("ct_score", 12)]; b = h[("t_score", 13)] - h[("t_score", 12)]
sa = h[("ct_score", 13)] - h[("t_score", 12)]; sb = h[("t_score", 13)] - h[("ct_score", 12)]
team = a.isin([0, 1]) & b.isin([0, 1]) & ((a + b) == 1)
side = sa.isin([0, 1]) & sb.isin([0, 1]) & ((sa + sb) == 1)
nontied = h[("ct_score", 12)] != h[("t_score", 12)]
out = pd.DataFrame({"day": [k.split("/")[3] for k in h.index], "team_only": (team & ~side) & nontied, "side_only": (side & ~team) & nontied, "neither": ~team & ~side, "nontied": nontied}, index=h.index)
g = out.groupby("day").agg(matches=("nontied", "size"), nontied=("nontied", "sum"), side_only=("side_only", "sum"), team_only=("team_only", "sum"), neither=("neither", "sum"))
print(g.to_string())
