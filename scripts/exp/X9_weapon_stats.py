import os, time
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import polars as pl

T0 = time.time()
ROOT = {
    "2024": r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds",
    "2026": r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds",
}
DATES = {"2024": "2023-12-28,2024-02-25", "2026": "2026-09-18,2026-10-08"}
ERAS = ["2024", "2026"]
B = 2000
SEED = 20261013
NONWEAPON = ("knife", "grenade", "molotov", "incgrenade", "inferno", "flash", "smoke", "decoy", "bomb", "c4")


def scan(era, ch, cols):
    p = os.path.join(ROOT[era], f"basic_{ch}.{DATES[era]}", "dataframe_*")
    return pl.scan_parquet(p, glob=True, missing_columns="insert", extra_columns="ignore",
                           cast_options=pl.ScanCastOptions(integer_cast="allow-float")).select(cols)


def is_weapon(name):
    if not name:
        return False
    n = name.lower()
    return not any(s in n for s in NONWEAPON)


def team_ok(a, p):
    return (pl.col(a) != pl.col(p)) | pl.col(a).is_null() | pl.col(p).is_null()


def kill_filter():
    return (pl.col("attacker_id_fixed").is_not_null()
            & (pl.col("attacker_id_fixed") != pl.col("player_id_fixed"))
            & team_ok("attacker_team_code", "player_team_code"))


def load(era):
    univ = sorted(scan(era, "round_start", ["match_key"]).unique().collect()["match_key"].to_list())
    sh = (scan(era, "weapon_fire", ["match_key", "weapon_name"])
          .group_by(["match_key", "weapon_name"]).agg(pl.len().alias("shots")).collect())
    kl = (scan(era, "player_death", ["match_key", "weapon_name", "attacker_id_fixed", "player_id_fixed",
                                     "attacker_team_code", "player_team_code", "is_headshot"])
          .filter(kill_filter())
          .group_by(["match_key", "weapon_name"])
          .agg(pl.len().alias("kills"), pl.col("is_headshot").cast(pl.Int64).sum().alias("hs_kills"))
          .collect())
    hu = (scan(era, "player_hurt", ["match_key", "weapon_name", "attacker_id_fixed", "player_id_fixed",
                                    "attacker_team_code", "player_team_code", "health_removed", "hit_box_code"])
          .filter(kill_filter())
          .group_by(["match_key", "weapon_name"])
          .agg(pl.len().alias("hits"),
               pl.col("health_removed").cast(pl.Float64).sum().alias("dmg"),
               (pl.col("hit_box_code") == 1).cast(pl.Int64).sum().alias("head_hits"))
          .collect())
    return univ, sh, kl, hu


def to_per_match(univ, df, col, weapon):
    uidx = pl.DataFrame({"match_key": univ, "mi": np.arange(len(univ), dtype=np.int64)})
    sub = df.filter(pl.col("weapon_name") == weapon).join(uidx, on="match_key", how="left")
    unmatched = int(sub["mi"].is_null().sum())
    sub = sub.filter(pl.col("mi").is_not_null())
    arr = np.zeros(len(univ), dtype=np.float64)
    np.add.at(arr, sub["mi"].to_numpy(), sub[col].cast(pl.Float64).fill_null(0.0).to_numpy())
    return arr, unmatched


data = {}
for era in ERAS:
    univ, sh, kl, hu = load(era)
    data[era] = dict(univ=univ, sh=sh, kl=kl, hu=hu)
    print(f"[{era}] matches(round_start)={len(univ)} shot_rows={sh.height} kill_rows={kl.height} hurt_rows={hu.height} "
          f"t={time.time()-T0:.1f}s", flush=True)
    print(f"[{era}] shots total={int(sh['shots'].sum())} kills total={int(kl['kills'].sum())} hits total={int(hu['hits'].sum())}", flush=True)
    print(f"[{era}] weapon_fire names: {sorted(sh['weapon_name'].unique().to_list())}", flush=True)
    print(f"[{era}] player_death names: {sorted(kl['weapon_name'].unique().to_list())}", flush=True)
    print(f"[{era}] player_hurt names: {sorted(hu['weapon_name'].unique().to_list())}", flush=True)

top = {}
for era in ERAS:
    t = (data[era]["kl"].group_by("weapon_name").agg(pl.col("kills").sum())
         .filter(pl.col("weapon_name").map_elements(is_weapon, return_dtype=pl.Boolean))
         .sort("kills", descending=True).head(10))
    top[era] = t["weapon_name"].to_list()
    print(f"[{era}] top10 by kills: " + ", ".join(f"{w}={k}" for w, k in zip(t["weapon_name"].to_list(), t["kills"].to_list())), flush=True)

W = []
for era in ERAS:
    for w in top[era]:
        if w not in W:
            W.append(w)
print("union weapons:", W, flush=True)

rng = np.random.default_rng(SEED)
IDX = {era: rng.integers(0, len(data[era]["univ"]), size=(B, len(data[era]["univ"]))) for era in ERAS}


def ratio_boot(num, den, idx):
    n = num[idx].sum(axis=1)
    d = den[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(d > 0, n / d, np.nan)
    return r


def pct(x):
    return np.nanpercentile(x, [2.5, 97.5])


stats = {}
for era in ERAS:
    d = data[era]
    for w in W:
        s, us = to_per_match(d["univ"], d["sh"], "shots", w)
        k, uk = to_per_match(d["univ"], d["kl"], "kills", w)
        hs, _ = to_per_match(d["univ"], d["kl"], "hs_kills", w)
        h, uh = to_per_match(d["univ"], d["hu"], "hits", w)
        dm, _ = to_per_match(d["univ"], d["hu"], "dmg", w)
        hh, _ = to_per_match(d["univ"], d["hu"], "head_hits", w)
        ix = IDX[era]
        r_kps = ratio_boot(k, s, ix)
        r_dmg = ratio_boot(dm, h, ix)
        r_hsh = ratio_boot(hh, h, ix)
        r_khs = ratio_boot(hs, k, ix)
        stats[(era, w)] = dict(
            shots=int(s.sum()), kills=int(k.sum()), hits=int(h.sum()),
            kps=float(k.sum() / s.sum()) if s.sum() > 0 else float("nan"),
            kps_ci=pct(r_kps),
            dmg=float(dm.sum() / h.sum()) if h.sum() > 0 else float("nan"),
            dmg_ci=pct(r_dmg), r_dmg=r_dmg,
            head_share=float(hh.sum() / h.sum()) if h.sum() > 0 else float("nan"),
            head_ci=pct(r_hsh), r_hsh=r_hsh,
            khs=float(hs.sum() / k.sum()) if k.sum() > 0 else float("nan"),
            khs_ci=pct(r_khs), r_kps=r_kps,
            matches_with_shots=int((s > 0).sum()), unmatched=uk + uh,
        )

print(f"\n=== PER-WEAPON STATS (kills/shot: match-cluster bootstrap 95% CI, B={B}) ===", flush=True)
hdr = f"{'weapon':<22}{'era':<6}{'shots':>9}{'kills':>8}{'k/shot':>9}{'95% CI':>22}{'hits':>9}{'dmg/hit':>9}{'dmgCI':>18}{'head_hit%':>10}{'kill_HS%':>9}"
print(hdr, flush=True)
for w in W:
    for era in ERAS:
        st = stats[(era, w)]
        print(f"{w:<22}{era:<6}{st['shots']:>9}{st['kills']:>8}{st['kps']:>9.4f}"
              f"{'[%.4f, %.4f]' % tuple(st['kps_ci']):>22}{st['hits']:>9}{st['dmg']:>9.2f}"
              f"{'[%.2f, %.2f]' % tuple(st['dmg_ci']):>18}{st['head_share']*100:>9.1f}%{st['khs']*100:>8.1f}%", flush=True)

print("\n=== ERA DIFFERENCE 2026 minus 2024 (independent bootstrap, paired by rep) ===", flush=True)
print(f"{'weapon':<22}{'d_k/shot':>10}{'95% CI':>24}{'excl0':>7}{'d_dmg/hit':>11}{'95% CI':>22}{'excl0':>7}{'d_head%':>9}{'95% CI (pp)':>24}{'excl0':>7}", flush=True)
for w in W:
    a, b = stats[("2024", w)], stats[("2026", w)]
    dk = b["r_kps"] - a["r_kps"]
    dd = b["r_dmg"] - a["r_dmg"]
    dh = b["r_hsh"] - a["r_hsh"]
    ck, cd, chh = pct(dk), pct(dd), pct(dh)
    ex = lambda c: "yes" if (c[0] > 0 or c[1] < 0) else "no"
    print(f"{w:<22}{(b['kps']-a['kps']):>10.4f}{'[%.4f, %.4f]' % tuple(ck):>24}{ex(ck):>7}"
          f"{(b['dmg']-a['dmg']):>11.2f}{'[%.2f, %.2f]' % tuple(cd):>22}{ex(cd):>7}"
          f"{(b['head_share']-a['head_share'])*100:>9.1f}{'[%.1f, %.1f]' % tuple(chh*100):>24}{ex(chh):>7}", flush=True)

print(f"\nelapsed {time.time()-T0:.1f}s", flush=True)
