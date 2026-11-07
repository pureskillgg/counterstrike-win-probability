"""PISTOL lens verification for the buy-type pilot. Read-only over data; writes only logs/buy_types/verify_pistol.txt."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"G:\pureskill\csds-cs2")
SS = ROOT / "data" / "state_sample"
HDR = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds" / "header" / "dataframe_00000"
OUT = ROOT / "logs" / "buy_types" / "verify_pistol.txt"

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
KEY = ["match_key", "round"]

lines = []


def out(s=""):
    s = str(s)
    lines.append(s)
    print(s)


def short(mk):
    m = UUID_RE.search(str(mk))
    return m.group(0)[:8] if m else str(mk)


def qfmt(x):
    x = np.asarray(x, dtype=float)
    return " ".join(f"{np.quantile(x, q):9.1f}" for q in QS)


def main():
    out("# PISTOL lens verification (verify_buy_pistol.py)")
    out(f"# state sample: {SS}")
    out(f"# tome header:  {HDR}")
    out("")

    pl = pd.read_parquet(SS / "players_1hz.parquet")
    rd = pd.read_parquet(SS / "rounds.parquet")
    bt = pd.read_parquet(SS / "round_buy_types.parquet")
    hd = pd.read_parquet(HDR)

    # ---------- 0. inventory ----------
    out("## 0. Inventory")
    out(f"players_1hz rows: {len(pl):,}; distinct match_key: {pl.match_key.nunique()}; "
        f"distinct (match_key, round): {pl.groupby(KEY).ngroups}")
    out(f"rounds.parquet rows: {len(rd):,}; distinct match_key: {rd.match_key.nunique()}; "
        f"round min/max: {rd['round'].min()}/{rd['round'].max()}")
    out(f"rounds.parquet round == round_number for all rows: {bool((rd['round'] == rd['round_number']).all())} "
        f"(mismatches: {int((rd['round'] != rd['round_number']).sum())})")
    out(f"round_buy_types rows: {len(bt):,}; distinct match_key: {bt.match_key.nunique()}")
    out(f"header rows: {len(hd):,}")
    out("")

    # ---------- 1. freeze-end snapshot and pistol economy ----------
    out("## 1. Pistol rounds: freeze-end equipment and money")
    fe_min = pl.groupby(KEY, as_index=False)["tick"].min().rename(columns={"tick": "fe_tick"})
    pl2 = pl.merge(fe_min, on=KEY, how="left")
    fe = pl2[pl2["tick"] == pl2["fe_tick"]].copy()
    dup = int(fe.duplicated(["match_key", "round", "player_id"]).sum())
    out(f"freeze-end snapshot = min tick per (match_key, round); player-rows at freeze-end: {len(fe):,}; "
        f"duplicate (match,round,player): {dup}")

    rr = rd.merge(fe_min, on=KEY, how="left")
    n_missing = int(rr["fe_tick"].isna().sum())
    n_eq = int((rr["fe_tick"] == rr["live_start_tick"]).sum())
    out(f"rounds with freeze-end snapshot: {len(rr) - n_missing} of {len(rr)}; "
        f"fe_tick == live_start_tick in {n_eq} of {len(rr)} rounds")
    out("")

    groups = [
        ("R1 (pistol)", fe["round"] == 1),
        ("R13 (pistol, 2nd half)", fe["round"] == 13),
        ("R2-12", fe["round"].between(2, 12)),
        ("R14-24", fe["round"].between(14, 24)),
        ("R25+ (OT, ref only)", fe["round"] >= 25),
    ]
    out("Money at freeze-end: q05..q95 | mean | max  (rounds = distinct match,round; playerrows = freeze-end player-rows)")
    for name, mask in groups:
        sub = fe[mask]
        nr = sub[["match_key", "round"]].drop_duplicates().shape[0]
        m = sub["money"].to_numpy(float)
        out(f"{name:24s} rounds={nr:4d} playerrows={len(sub):5d} | {qfmt(m)} | {m.mean():7.1f} | {int(m.max())}")
    out("")
    out("Equipment (equipment_value_calc) at freeze-end: q05..q95 | mean | max | share>1000 | share>2700")
    for name, mask in groups:
        sub = fe[mask]
        e = sub["equipment_value_calc"].to_numpy(float)
        out(f"{name:24s} | {qfmt(e)} | {e.mean():7.1f} | {int(e.max())} | "
            f"{(e > 1000).mean():.4f} | {(e > 2700).mean():.4f}")
    out("")
    out("Share of player-rows with money exactly 800 / money > 800:")
    for name, mask in groups:
        sub = fe[mask]
        m = sub["money"]
        out(f"{name:24s} money==800: {(m == 800).mean():.4f}  money>800: {(m > 800).mean():.4f}  n={len(sub)}")
    out("")
    for name, mask in [("R1", fe["round"] == 1), ("R13", fe["round"] == 13), ("R2-12", fe["round"].between(2, 12))]:
        vc = fe[mask]["money"].value_counts().head(8)
        out(f"top money values {name}: " + ", ".join(f"{int(k)}:{int(v)}" for k, v in vc.items()))
    out("")

    # ---------- 2. halftime side swap ----------
    out("## 2. Halftime side swap (round 12 -> round 13), freeze-end team_code per player_id")
    f12 = fe[fe["round"] == 12][["match_key", "player_id", "team_code"]].rename(columns={"team_code": "t12"})
    f13 = fe[fe["round"] == 13][["match_key", "player_id", "team_code"]].rename(columns={"team_code": "t13"})
    both_keys = sorted(set(f12["match_key"]) & set(f13["match_key"]))
    out(f"matches with freeze-end rows in both round 12 and round 13: {len(both_keys)}")
    out(f"team_code values r12: {sorted(f12['t12'].unique().tolist())}, r13: {sorted(f13['t13'].unique().tolist())}")

    pairs_all = f12.merge(f13, on=["match_key", "player_id"])
    pairs_all["flip"] = pairs_all["t12"] != pairs_all["t13"]
    out(f"ALL such matches: players in both rounds: {len(pairs_all):,}; flipped: {int(pairs_all['flip'].sum()):,}; "
        f"not flipped: {int((~pairs_all['flip']).sum()):,}")

    idx = np.unique(np.linspace(0, len(both_keys) - 1, 20).round().astype(int))
    sample = [both_keys[i] for i in idx]
    out(f"sample: {len(sample)} matches, evenly spaced over the sorted list (deterministic)")
    ps = pairs_all[pairs_all["match_key"].isin(sample)]
    out(f"{'match':10s} {'players':>7s} {'flipped':>7s} {'r12 T/CT':>9s} {'r13 T/CT':>9s}")
    n_all_flip = 0
    for mk in sample:
        s = ps[ps["match_key"] == mk]
        r12 = f12[f12["match_key"] == mk]["t12"].value_counts()
        r13 = f13[f13["match_key"] == mk]["t13"].value_counts()
        nflip = int(s["flip"].sum())
        if len(s) and nflip == len(s):
            n_all_flip += 1
        out(f"{short(mk):10s} {len(s):7d} {nflip:7d} {int(r12.get(2, 0)):>4d}/{int(r12.get(3, 0)):<4d} "
            f"{int(r13.get(2, 0)):>4d}/{int(r13.get(3, 0)):<4d}")
    out(f"SAMPLE totals: matches {len(sample)}; players compared {len(ps)}; flipped {int(ps['flip'].sum())}; "
        f"matches where every compared player flipped {n_all_flip} of {len(sample)}")
    non = ps[~ps["flip"]]
    out(f"SAMPLE non-flipped player-rows: {len(non)}" + ("" if len(non) == 0 else
        " -> " + "; ".join(f"{short(r.match_key)} pid={r.player_id} t12={r.t12} t13={r.t13}" for r in non.itertuples())))
    out("")

    out("Flip pattern between consecutive rounds (all matches, players present in both rounds):")
    out(f"{'pair':8s} {'players':>8s} {'flipped':>8s} {'matches':>8s}")
    for r in range(1, 31):
        a = fe[fe["round"] == r][["match_key", "player_id", "team_code"]].rename(columns={"team_code": "ta"})
        b = fe[fe["round"] == r + 1][["match_key", "player_id", "team_code"]].rename(columns={"team_code": "tb"})
        m = a.merge(b, on=["match_key", "player_id"])
        if len(m) == 0:
            continue
        fl = m["ta"] != m["tb"]
        out(f"{r:>2d}->{r + 1:<3d} {len(m):8d} {int(fl.sum()):8d} {m.loc[fl, 'match_key'].nunique():8d}")
    out("")

    # ---------- 3. max_rounds from tome header ----------
    out("## 3. max_rounds from tome header")
    key_uuid = hd["key"].astype(str).str.extract(f"({UUID_RE.pattern})", expand=False)
    mid_uuid = hd["match_id"].astype(str).str.extract(f"({UUID_RE.pattern})", expand=False)
    out(f"header key sample: {hd['key'].iloc[0]}  | match_id sample: {hd['match_id'].iloc[0]}")
    out(f"header rows with uuid in key: {int(key_uuid.notna().sum())}; in match_id: {int(mid_uuid.notna().sum())}")
    h = hd.assign(uuid=key_uuid.fillna(mid_uuid))[["uuid", "max_rounds", "is_wingman", "match_type", "is_gotv_recording"]]
    h = h.drop_duplicates("uuid")
    rd_m = rd[["match_key"]].drop_duplicates().assign(
        uuid=lambda d: d["match_key"].str.extract(f"({UUID_RE.pattern})", expand=False))
    rd_m = rd_m.merge(h, on="uuid", how="left")
    out(f"match_keys in rounds.parquet: {len(rd_m)}; found in header: {int(rd_m['max_rounds'].notna().sum())}")
    out("max_rounds distribution over all state-sample matches:")
    vc = rd_m["max_rounds"].value_counts(dropna=False)
    out("  " + ", ".join(f"{k}:{v}" for k, v in vc.items()))
    out(f"is_wingman among these matches: {rd_m['is_wingman'].value_counts(dropna=False).to_dict()}")
    out(f"match_type among these matches: {rd_m['match_type'].value_counts(dropna=False).to_dict()}")
    smp = rd_m[rd_m["match_key"].isin(sample)]
    out(f"sampled halftime matches ({len(smp)}): max_rounds ->")
    out("  " + ", ".join(f"{short(r.match_key)}:{r.max_rounds}" for r in smp.itertuples()))
    out(f"sampled matches with max_rounds == 24: {int((smp['max_rounds'] == 24).sum())} of {len(smp)}")
    out("")

    # ---------- 4. pistol labels ----------
    out("## 4. Pistol labels in round_buy_types")
    out(f"is_pistol rows: {int(bt['is_pistol'].sum())}; rows by round among pistol:")
    out("  " + str(bt.loc[bt["is_pistol"], "round"].value_counts().sort_index().to_dict()))
    mism = int((bt["is_pistol"] != bt["round"].isin([1, 13])).sum())
    out(f"rows where is_pistol != (round in {{1,13}}): {mism}")
    out(f"max round labelled pistol: {int(bt.loc[bt['is_pistol'], 'round'].max())}")
    hi = bt[bt["round"] >= 25]
    out(f"kept rounds with round>=25: {len(hi)}; of these labelled pistol: {int(hi['is_pistol'].sum())}")
    rd_hi = rd[rd["round"] >= 25]
    out(f"rounds.parquet rows with round>=25: {len(rd_hi)}; max round in rounds.parquet: {int(rd['round'].max())}")
    out(f"matches with a kept round 1: {bt[bt['round'] == 1]['match_key'].nunique()}; "
        f"kept round 13: {bt[bt['round'] == 13]['match_key'].nunique()}; "
        f"kept pistol-labelled matches: {bt[bt['is_pistol']]['match_key'].nunique()}")
    out(f"raw rounds.parquet: matches with round 1: {rd[rd['round'] == 1]['match_key'].nunique()}; "
        f"matches with round 13: {rd[rd['round'] == 13]['match_key'].nunique()}")
    per_match = rd.groupby("match_key").agg(n_rounds=("round", "nunique"), max_r=("round", "max"))
    per_match["has1"] = rd[rd["round"] == 1].groupby("match_key").size().reindex(per_match.index).fillna(0) > 0
    per_match["has13"] = rd[rd["round"] == 13].groupby("match_key").size().reindex(per_match.index).fillna(0) > 0
    out(f"matches with fewer than 13 rounds in rounds.parquet: {int((per_match['n_rounds'] < 13).sum())}; "
        f"matches with no round 13: {int((~per_match['has13']).sum())}; with no round 1: {int((~per_match['has1']).sum())}")
    short_m = per_match[per_match["n_rounds"] < 13]
    out(f"  short matches: n_rounds {sorted(short_m['n_rounds'].tolist())}; max round {sorted(short_m['max_r'].tolist())}")
    no13_not_short = per_match[(~per_match["has13"]) & (per_match["n_rounds"] >= 13)]
    out(f"  matches with no round 13 but >=13 rounds: {len(no13_not_short)}")
    no1 = per_match[~per_match["has1"]]
    out("  matches with no round 1 (n_rounds, max_r): " +
        ", ".join(f"({a},{b})" for a, b in zip(no1["n_rounds"], no1["max_r"])))
    out("")

    rk = rd[["match_key", "round"]].merge(bt[["match_key", "round"]], how="left", indicator=True)
    dropped = rk[rk["_merge"] == "left_only"]
    out(f"rounds.parquet rows not in round_buy_types: {len(dropped)}")
    for r in dropped.itertuples():
        sub = fe[(fe["match_key"] == r.match_key) & (fe["round"] == r.round)]
        out(f"  dropped: {short(r.match_key)} round {r.round}; team_code counts: {sub['team_code'].value_counts().to_dict()}")
    out("")

    side = fe.groupby(["match_key", "round", "team_code"], as_index=False)["equipment_value_calc"].sum()
    ct = side[side["team_code"] == 3].rename(columns={"equipment_value_calc": "ct_eq_re"}).drop(columns="team_code")
    tt = side[side["team_code"] == 2].rename(columns={"equipment_value_calc": "t_eq_re"}).drop(columns="team_code")
    chk = bt.merge(ct, on=KEY, how="left").merge(tt, on=KEY, how="left")
    d_ct = (chk["ct_equipment"] - chk["ct_eq_re"]).abs()
    d_t = (chk["t_equipment"] - chk["t_eq_re"]).abs()
    out(f"re-derived per-side equipment vs stored (team 3 = CT, team 2 = T): "
        f"max |diff| CT={float(d_ct.max()):.1f}, T={float(d_t.max()):.1f}; rows with any diff>0: "
        f"{int(((d_ct > 0) | (d_t > 0)).sum())}")
    pc = fe.groupby(["match_key", "round", "team_code"]).size().unstack(fill_value=0)
    out(f"player-rows per side (T=2, CT=3) across all freeze-end rounds: value counts of (T,CT) pairs: "
        f"{pc.value_counts().head(8).to_dict() if False else pc.groupby([pc[2], pc[3]]).size().head(12).to_dict()}")
    pist = chk[chk["is_pistol"]]
    out(f"pistol rows: side-rounds {len(pist) * 2}; stored ct_buy values: {pist['ct_buy'].value_counts().to_dict()}")
    out(f"pistol rows: stored t_buy values: {pist['t_buy'].value_counts().to_dict()}")
    out(f"pistol rows: max stored side equipment: CT={int(pist['ct_equipment'].max())}, T={int(pist['t_equipment'].max())}")
    out("")

    # ---------- 5. follow-up: money + equipment identity at freeze-end ----------
    out("## 5. Money + equipment identity at freeze-end (s = money + equipment_value_calc)")
    out("Pistol start = 800 money + free default pistol, so s should be about 1000 for a pistol-start player.")
    fe["s"] = fe["money"].astype(float) + fe["equipment_value_calc"].astype(float)
    fe["start_like"] = fe["s"] <= 1100
    for name, mask in [("R1", fe["round"] == 1), ("R13", fe["round"] == 13), ("R12", fe["round"] == 12),
                       ("R2-11", fe["round"].between(2, 11)), ("R14-24", fe["round"].between(14, 24))]:
        sub = fe[mask]
        s = sub["s"].to_numpy(float)
        out(f"{name:7s} playerrows={len(sub):5d} s q05/q50/q95={np.quantile(s, 0.05):.0f}/{np.quantile(s, 0.5):.0f}/"
            f"{np.quantile(s, 0.95):.0f} share s<=1000={(s <= 1000).mean():.4f} share s<=1100={(s <= 1100).mean():.4f} "
            f"max={int(s.max())}")
    out("")
    for name, mask in [("R1", fe["round"] == 1), ("R12", fe["round"] == 12), ("R13", fe["round"] == 13),
                       ("R2-11", fe["round"].between(2, 11))]:
        sub = fe[mask]
        out(f"{name:6s} armor>0 share: {(sub['armor'] > 0).mean():.4f}  helmet share: {sub['has_helmet'].mean():.4f}  "
            f"defuser share: {sub['has_defuser'].mean():.4f}  n={len(sub)}")
    out("")
    for name, mask in [("R1", fe["round"] == 1), ("R13", fe["round"] == 13)]:
        o = fe[mask & (fe["equipment_value_calc"] > 1000)]
        out(f"{name}: player-rows with equipment>1000: {len(o)} of {int(mask.sum())}; money of those: "
            f"{sorted(o['money'].astype(int).tolist())}; max equipment {int(o['equipment_value_calc'].max())}; "
            f"max s {int(o['s'].max())}; helmets among them: {int(o['has_helmet'].sum())}")
    out("")

    # ---------- 6. per-round pistol-start signature ----------
    out("## 6. Per-round pistol-start signature (sides = match x round x team)")
    side = fe.groupby(["match_key", "round", "team_code"]).agg(
        n=("player_id", "size"), all_start=("start_like", "all")).reset_index()
    rnd = fe.groupby("round").agg(playerrows=("player_id", "size"),
                                   share_start_like=("start_like", "mean"),
                                   share_money_gt800=("money", lambda x: (x > 800).mean()),
                                   median_s=("s", "median"))
    sides_t = side.groupby("round").agg(sides=("n", "size"), sides_all_start=("all_start", "sum"))
    matches_t = fe.groupby("round")["match_key"].nunique().rename("matches")
    tt_r = rnd.join(sides_t).join(matches_t)
    out("round matches playerrows share(s<=1100) share(money>800) median(s) sides sides_all_start")
    for r, row in tt_r.iterrows():
        out(f"{int(r):5d} {int(row['matches']):7d} {int(row['playerrows']):10d} {row['share_start_like']:14.4f} "
            f"{row['share_money_gt800']:16.4f} {row['median_s']:9.1f} {int(row['sides']):5d} {int(row['sides_all_start']):14d}")
    ss = side[side["all_start"]]
    out(f"pistol-start-like sides (all 5 players s<=1100): {len(ss)}; by round: {ss.groupby('round').size().to_dict()}")
    out(f"rounds 1 and 13 sides that are NOT pistol-start-like: "
        f"{int(((side['round'].isin([1, 13])) & (~side['all_start'])).sum())} of {int(side['round'].isin([1, 13]).sum())}")
    out(f"player-rows with money>800 in rounds 1 or 13: {int(((fe['round'].isin([1, 13])) & (fe['money'] > 800)).sum())}")
    out("")

    # ---------- 7. round coverage per match ----------
    out("## 7. Round coverage per match (rounds.parquet)")
    gg = rd.groupby("match_key").agg(n=("round", "nunique"), mn=("round", "min"), mx=("round", "max"))
    gg["gap"] = gg["mx"] - gg["mn"] + 1 - gg["n"]
    has1 = set(rd.loc[rd["round"] == 1, "match_key"])
    has13 = set(rd.loc[rd["round"] == 13, "match_key"])
    out(f"matches: {len(gg)}; first round != 1: {int((gg['mn'] != 1).sum())}; with interior gaps: "
        f"{int((gg['gap'] > 0).sum())}; missing round numbers inside ranges: {int(gg['gap'].sum())}")
    out(f"matches with contiguous 1..max sequence: {int(((gg['mn'] == 1) & (gg['gap'] == 0)).sum())}")
    out(f"matches with fewer than 13 rounds: {int((gg['n'] < 13).sum())} (their rounds total {int(gg.loc[gg['n'] < 13, 'n'].sum())})")
    out(f"matches with round 1: {len(has1)}; round 13: {len(has13)}; both: {len(has1 & has13)}; "
        f"neither: {len(set(gg.index) - (has1 | has13))}")
    out(f"matches with round 13 but no freeze-end round 12: {len(has13 - set(f12['match_key']))}")
    out(f"matches with >=13 rounds but no round 13: {int(((gg['n'] >= 13) & (~gg.index.isin(list(has13)))).sum())}")
    out("")

    out("## 8. Done")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
