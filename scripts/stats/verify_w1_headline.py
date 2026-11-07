"""Independent recomputation of W1 headline numbers (kills by weapon, shots by weapon).

Reads only the needed columns of the 2026 tome via lazy polars scans.
Match-cluster bootstrap: 1000 resamples of matches, seed 0.
Writes its console output to logs\\stats\\verify_w1.log (appended by verify_w1_rifle.py later).
"""
import glob
import os

import numpy as np
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
LOG = r"G:\pureskill\csds-cs2\logs\stats\verify_w1.log"
NB = 1000
SEED = 0
out = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


def files(ch):
    return sorted(glob.glob(os.path.join(TOME, f"{ch}.2026-09-18,2026-10-08", "dataframe_*")))


log("== verify_w1_headline")

# --- match universe from round_end
re_lf = pl.scan_parquet(files("basic_round_end")).select(["match_key"])
matches = re_lf.select(pl.col("match_key").unique()).collect()["match_key"].sort().to_list()
n_match = len(matches)
mindex = {m: i for i, m in enumerate(matches)}
log("round_end distinct match_key:", n_match)

# --- player_death: classify deaths
deaths = (
    pl.scan_parquet(files("basic_player_death"))
    .select(
        [
            "match_key",
            "round",
            "player_id",
            "attacker_id",
            "player_team_code",
            "attacker_team_code",
            "weapon_name",
        ]
    )
    .collect()
)
log("player_death rows:", deaths.height, "distinct matches:", deaths["match_key"].n_unique())
log("attacker_team_code value counts:", deaths["attacker_team_code"].value_counts(sort=True).rows())
log("player_team_code value counts:", deaths["player_team_code"].value_counts(sort=True).rows())
log("attacker_id null:", deaths["attacker_id"].null_count(), " attacker_id==0:", int((deaths["attacker_id"] == 0).sum()))

# Fill null team codes from player_info (most common team per match, round, player)
pi = (
    pl.scan_parquet(files("basic_player_info"))
    .select(["match_key", "round", "player_id", "team_code"])
    .filter(pl.col("team_code").is_in([2, 3]))
    .group_by(["match_key", "round", "player_id"])
    .agg(pl.col("team_code").first().alias("pi_team"))
    .collect()
)
deaths = deaths.join(pi, on=["match_key", "round", "player_id"], how="left")
deaths = deaths.join(
    pi.rename({"player_id": "attacker_id", "pi_team": "att_pi_team"}),
    on=["match_key", "round", "attacker_id"],
    how="left",
)
deaths = deaths.with_columns(
    pl.coalesce([pl.col("player_team_code"), pl.col("pi_team")]).alias("vic_team"),
    pl.coalesce([pl.col("attacker_team_code"), pl.col("att_pi_team")]).alias("atk_team"),
)
log("victim team still null after fill:", deaths.filter(pl.col("vic_team").is_null()).height)

is_self = (pl.col("attacker_id") == pl.col("player_id")) & pl.col("attacker_id").is_not_null()
d = deaths.with_columns(
    pl.when(is_self & (pl.col("weapon_name") != "world"))
    .then(pl.lit("self"))
    .when(
        (pl.col("weapon_name") == "world")
        | (pl.col("atk_team").is_null() & pl.col("vic_team").is_not_null())
        | (pl.col("atk_team") == 0)
    )
    .then(pl.lit("world"))
    .when(pl.col("atk_team").is_in([2, 3]) & (pl.col("atk_team") == pl.col("vic_team")))
    .then(pl.lit("team"))
    .when(pl.col("atk_team").is_in([2, 3]) & pl.col("vic_team").is_in([2, 3]))
    .then(pl.lit("enemy"))
    .otherwise(pl.lit("unknown"))
    .alias("death_class")
)
cls = d["death_class"].value_counts(sort=True).rows()
log("death classes (team codes filled from player_info):", cls, "total:", d.height)
log("report: enemy 247,629 / world 5,100 / team 669 / self 73 / unknown 9")

enemy = d.filter(pl.col("death_class") == "enemy")
# merge silencer-off labels into silencer-on names (as the W1 report does for kills)
enemy = enemy.with_columns(
    pl.col("weapon_name").replace({"m4a1_silencer_off": "m4a1_silencer", "usp_silencer_off": "usp_silencer"})
)
log("enemy kills:", enemy.height)

# per-match enemy kill counts by weapon (raw weapon_name)
ek = enemy.with_columns(pl.col("match_key").replace_strict(mindex, default=-1).alias("mi"))
ek = ek.filter(pl.col("mi") >= 0)
weapons_k = sorted(ek["weapon_name"].unique().to_list())
wk = {w: i for i, w in enumerate(weapons_k)}
K = len(weapons_k)
M_kill = np.zeros((n_match, K), dtype=np.int64)
M_kill_ct = np.zeros((n_match, K), dtype=np.int64)
grp = ek.group_by(["mi", "weapon_name", "attacker_team_code"]).agg(pl.len().alias("n"))
for mi, wn, atk, n in grp.iter_rows():
    j = wk[wn]
    M_kill[mi, j] += n
    if atk == 3:
        M_kill_ct[mi, j] += n
total_enemy = M_kill.sum()
log("enemy kills matched to tome matches:", int(total_enemy))

# knives
knife_names = [w for w in weapons_k if "knife" in w or "bayonet" in w]
log("knife-like kill names:", knife_names, "kills:", int(sum(M_kill[:, wk[w]].sum() for w in knife_names)))

# --- weapon_fire shots
wf = (
    pl.scan_parquet(files("basic_weapon_fire"))
    .select(["match_key", "weapon_name"])
    .with_columns(pl.col("match_key").replace_strict(mindex, default=-1).alias("mi"))
    .filter(pl.col("mi") >= 0)
    .group_by(["mi", "weapon_name"])
    .agg(pl.len().alias("n"))
    .collect()
)
weapons_s = sorted(wf["weapon_name"].unique().to_list())
ws = {w: i for i, w in enumerate(weapons_s)}
M_shot = np.zeros((n_match, len(weapons_s)), dtype=np.int64)
for mi, wn, n in wf.iter_rows():
    M_shot[mi, ws[wn]] += n
total_shots = int(M_shot.sum())
log("weapon_fire events (all matches):", total_shots, "distinct weapon_name:", len(weapons_s))

# --- bootstrap (match clusters, seed 0)
rng = np.random.default_rng(SEED)
boot_idx = rng.integers(0, n_match, size=(NB, n_match))


def share_ci(M, col):
    tot_b = np.array([M[idx].sum() for idx in boot_idx])
    num_b = np.array([M[idx, col].sum() for idx in boot_idx])
    point = M[:, col].sum() / M.sum()
    lo, hi = np.percentile(num_b / tot_b, [2.5, 97.5])
    return point, lo, hi


def ct_ci(Mk, Mct, col):
    num_b = np.array([Mct[idx, col].sum() for idx in boot_idx])
    den_b = np.array([Mk[idx, col].sum() for idx in boot_idx])
    point = Mct[:, col].sum() / Mk[:, col].sum()
    lo, hi = np.percentile(num_b / den_b, [2.5, 97.5])
    return point, lo, hi


# Table 1 analogue
order = np.argsort(-M_kill.sum(0))
log("")
log("TOP 20 ENEMY-KILL WEAPONS (raw weapon_name, enemy kills, share % [95% CI], CT-attacker % [95% CI])")
for r, col in enumerate(order[:20], 1):
    name = weapons_k[col]
    k = int(M_kill[:, col].sum())
    p, lo, hi = share_ci(M_kill, col)
    cp, clo, chi = ct_ci(M_kill, M_kill_ct, col) if k > 0 else (float("nan"),) * 3
    log(
        f"{r:2d} {name:16s} {k:8,d}  {100*p:6.2f} [{100*lo:6.2f}, {100*hi:6.2f}]  "
        f"CT {100*cp:5.1f} [{100*clo:5.1f}, {100*chi:5.1f}]"
    )

# Table 2 analogue
order_s = np.argsort(-M_shot.sum(0))
log("")
log("TOP 10 SHOTS (weapon_fire, raw weapon_name)")
for r, col in enumerate(order_s[:10], 1):
    name = weapons_s[col]
    k = int(M_shot[:, col].sum())
    p, lo, hi = share_ci(M_shot, col)
    log(f"{r:2d} {name:16s} {k:12,d}  {100*p:6.2f} [{100*lo:6.2f}, {100*hi:6.2f}]")

# specific checks for the headline numbers
ak_k = int(M_kill[:, wk["ak47"]].sum())
ak_s = int(M_shot[:, ws["ak47"]].sum())
log("")
log("HEADLINE CHECKS")
log("H1 enemy kills total:", int(total_enemy), "(report 247,629)")
log("H1 ak47 kills:", ak_k, "(report 84,922)")
p, lo, hi = share_ci(M_kill, wk["ak47"])
log(f"H1 ak47 share: {100*p:.2f}% [{100*lo:.2f}, {100*hi:.2f}] (report 34.29 [33.95, 34.66])")
log("H2 weapon_fire events total:", total_shots, "(report 5,483,781)")
log("H2 ak47 shots:", ak_s, "(report 1,678,356)")
p, lo, hi = share_ci(M_shot, ws["ak47"])
log(f"H2 ak47 shot share: {100*p:.2f}% [{100*lo:.2f}, {100*hi:.2f}] (report 30.61 [30.31, 30.91])")

# classification totals for the reconciliation
log("")
log("death class totals:", dict(cls))
log("matches with at least one enemy kill:", int((M_kill.sum(1) > 0).sum()), "of", n_match)

with open(LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
log("wrote", LOG)
