"""MC step 2: streak-conditioned economy and win-rate tables, loss-bonus ladder, bootstrap, JSON and SVG.

Input (scratch, written by MC_01_team_rounds.py):
  <scratchpad>/MC_team_rounds.parquet
Cross-check (read only):
  data/state_sample/round_buy_types.parquet

Outputs:
  data/viz/MC.json
  data/viz/MC.svg

Definitions
  Unit: team-round. Primary population = rounds 2..max excluding pistol round 13 (round 1 has no prior
    round, so it is excluded from streak tables). Sensitivity population also keeps round 13.
  Loss streak k: the team lost exactly k consecutive rounds before this round (k capped at 5 = "5+").
    k = 0 means the previous round was a win. Win streak k: same with wins; k = 0 means previous round lost.
  Economy at freeze end (first live second): team equipment = sum of equipment_value_calc over 5 players;
    team money = sum of money over 5 players. Bankroll proxy = money + equipment (per player); see caveats.
  Win rate = share of team-rounds won.
  Residual = won - E[won | equipment-difference quintile], where E is the primary-population mean in that
    quintile (equipment difference = team equipment minus opponent equipment, same round, freeze end).
  Intervals: match-cluster bootstrap, 1000 resamples of the 150 matches with replacement, seed 0,
    numpy default_rng, 95% percentile interval (numpy linear). Estimators are pooled ratios of sums.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
SAMPLE = ROOT / "data/state_sample"
SCRATCH = Path("C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/"
               "e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad")
OUT_JSON = ROOT / "data/viz/MC.json"
OUT_SVG = ROOT / "data/viz/MC.svg"

N_BOOT = 1000
SEED = 0
CAP = 5
BUCKETS = [0, 1, 2, 3, 4, 5]  # 5 means 5+
REF_LOSS_BONUS = {1: 1400, 2: 1900, 3: 2400, 4: 2900, 5: 3400}  # external CS2 rule, NOT verified here

CT_BLUE = "#2563eb"  # reserved for CT (not used for streak series)
T_ORANGE = "#d97706"  # reserved for T (not used for streak series)
NEUTRAL = "#475569"


def label(k):
    return "5+" if k == CAP else str(k)


def main():
    df = pl.read_parquet(SCRATCH / "MC_team_rounds.parquet")
    matches = sorted(df["match_key"].unique().to_list())
    M = len(matches)
    midx = {m: i for i, m in enumerate(matches)}
    mi = np.array([midx[m] for m in df["match_key"].to_list()])
    rnd = df["round"].to_numpy().astype(int)
    won = df["won"].to_numpy().astype(float)
    teq = df["team_eq"].to_numpy().astype(float)
    tmo = df["team_money"].to_numpy().astype(float)
    oeq = df["opp_eq"].to_numpy().astype(float)
    lrun = df["loss_run"].to_numpy().astype(int)
    lok = df["loss_ok"].to_numpy().astype(bool)
    wrun = df["win_run"].to_numpy().astype(int)
    wok = df["win_ok"].to_numpy().astype(bool)
    eqdiff = teq - oeq

    primary = (rnd >= 2) & (rnd != 13)
    sensitivity = rnd >= 2

    # Equipment-difference quintiles from the primary population
    edges = np.quantile(eqdiff[primary], [0.2, 0.4, 0.6, 0.8])
    qidx = np.digitize(eqdiff, edges)
    ehat = np.array([won[primary & (qidx == q)].mean() for q in range(5)])
    resid = won - ehat[qidx]

    # Bootstrap draws (shared across all tables so intervals use the same resamples)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, M, size=(N_BOOT, M))
    W = np.zeros((N_BOOT, M))
    for i in range(N_BOOT):
        W[i] = np.bincount(draws[i], minlength=M)

    vals = {"n": np.ones(len(df)), "won": won, "eq": teq, "mo": tmo, "ed": eqdiff, "res": resid}

    def sums(sel, bucket_arr):
        S = {k: np.zeros((M, len(BUCKETS))) for k in vals}
        nm = []
        for j, bk in enumerate(BUCKETS):
            s = sel & (bucket_arr == bk)
            nm.append(len(np.unique(mi[s])))
            for k, v in vals.items():
                S[k][:, j] = np.bincount(mi[s], weights=v[s], minlength=M)
        return S, nm

    def ci(S, num_key, den_key="n", scale=1.0, extra=None):
        tot_num = S[num_key].sum(axis=0)
        tot_den = S[den_key].sum(axis=0)
        est = tot_num / tot_den * scale
        bn = W @ S[num_key]
        bd = W @ S[den_key]
        with np.errstate(divide="ignore", invalid="ignore"):
            boot = bn / bd * scale
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        return [float(x) for x in est], [float(x) for x in lo], [float(x) for x in hi]

    def bankroll_ci(S):
        tot = (S["eq"].sum(axis=0) + S["mo"].sum(axis=0)) / (5 * S["n"].sum(axis=0))
        bn = (W @ S["eq"] + W @ S["mo"]) / (5 * (W @ S["n"]))
        lo, hi = np.percentile(bn, [2.5, 97.5], axis=0)
        return [float(x) for x in tot], [float(x) for x in lo], [float(x) for x in hi]

    def build_table(sel_pop, ok, run, kind):
        sel = sel_pop & ok
        bucket_arr = np.minimum(run, CAP)
        S, nm = sums(sel, bucket_arr)
        out = {}
        out["team_rounds"] = [int(S["n"][:, j].sum()) for j in range(len(BUCKETS))]
        out["matches"] = nm
        wr = ci(S, "won")
        te = ci(S, "eq")
        tm = ci(S, "mo")
        pe = ci(S, "eq", scale=1 / 5)
        pm = ci(S, "mo", scale=1 / 5)
        ed = ci(S, "ed")
        rs = ci(S, "res")
        out["win_rate"] = {"est": wr[0], "lo": wr[1], "hi": wr[2]}
        out["team_equipment"] = {"est": te[0], "lo": te[1], "hi": te[2]}
        out["team_money"] = {"est": tm[0], "lo": tm[1], "hi": tm[2]}
        out["per_player_equipment"] = {"est": pe[0], "lo": pe[1], "hi": pe[2]}
        out["per_player_money"] = {"est": pm[0], "lo": pm[1], "hi": pm[2]}
        out["equipment_difference_vs_opponent"] = {"est": ed[0], "lo": ed[1], "hi": ed[2]}
        out["win_rate_residual_given_equipment_diff"] = {"est": rs[0], "lo": rs[1], "hi": rs[2]}
        out["labels"] = [label(k) for k in BUCKETS]
        out["kind"] = kind
        return out

    prim_loss = build_table(primary, lok, lrun, "after_loss_streak")
    prim_win = build_table(primary, wok, wrun, "after_win_streak")
    sens_loss = build_table(sensitivity, lok, lrun, "after_loss_streak_incl_r13")
    sens_win = build_table(sensitivity, wok, wrun, "after_win_streak_incl_r13")

    # Round 1 (no prior round) and round 13 (pistol) descriptive, not streak-conditioned
    r1 = rnd == 1
    r13 = rnd == 13
    pistol_info = {}
    for name, sel in [("round_1", r1), ("round_13", r13)]:
        pistol_info[name] = {
            "team_rounds": int(sel.sum()),
            "win_rate": float(won[sel].mean()),
            # Pistol-round check only: at pistol rounds nothing is carried over, so money + equipment per
            # player should sit near the start money plus the default pistol (about 1000 if so).
            "per_player_money_plus_equipment_median": float(np.median((teq[sel] + tmo[sel]) / 5)),
            "per_player_money_plus_equipment_mean": float(((teq[sel] + tmo[sel]) / 5).mean()),
        }

    # Loss-bonus ladder: losing side after k consecutive losses (k = loss run including the previous round)
    ladder = []
    prev = None
    for j, k in enumerate(BUCKETS):
        if k == 0:
            continue
        row = {
            "consecutive_losses": label(k),
            "team_rounds": prim_loss["team_rounds"][j],
            "matches": prim_loss["matches"][j],
            "per_player_money": prim_loss["per_player_money"]["est"][j],
            "per_player_money_lo": prim_loss["per_player_money"]["lo"][j],
            "per_player_money_hi": prim_loss["per_player_money"]["hi"][j],
            "per_player_equipment": prim_loss["per_player_equipment"]["est"][j],
            "per_player_equipment_lo": prim_loss["per_player_equipment"]["lo"][j],
            "per_player_equipment_hi": prim_loss["per_player_equipment"]["hi"][j],
            "reference_loss_bonus_external": REF_LOSS_BONUS[k],
        }
        if prev is not None:
            row["step_money_vs_prev"] = row["per_player_money"] - prev["per_player_money"]
            row["step_equipment_vs_prev"] = row["per_player_equipment"] - prev["per_player_equipment"]
        ladder.append(row)
        prev = row

    # Checks
    cross = pl.read_parquet(SAMPLE / "round_buy_types.parquet", columns=["match_key", "round", "ct_equipment",
                                                                         "t_equipment", "ct_money", "t_money"])
    x = df.join(cross, on=["match_key", "round"], how="inner")
    side_eq = pl.when(pl.col("side") == 3).then(pl.col("ct_equipment")).otherwise(pl.col("t_equipment"))
    side_mo = pl.when(pl.col("side") == 3).then(pl.col("ct_money")).otherwise(pl.col("t_money"))
    x = x.with_columns(side_eq.alias("buy_eq"), side_mo.alias("buy_mo"))
    eq_match = int((x["team_eq"].to_numpy() == x["buy_eq"].to_numpy()).sum())
    mo_match = int((x["team_money"].to_numpy() == x["buy_mo"].to_numpy()).sum())
    rounds_all = pl.read_parquet(SAMPLE / "rounds.parquet", columns=["match_key", "round"])
    gap_stats = rounds_all.group_by("match_key").agg(pl.col("round").max().alias("mx"),
                                                     pl.col("round").n_unique().alias("n"))
    n_gap = int((gap_stats["mx"] != gap_stats["n"]).sum())
    # Post-buy check on round 1: armor buyers (Kevlar 650) should show money near 800 - 650 = 150 if money is
    # the balance after the buy, and money + equipment near 1000 (800 start + 200 default pistol).
    snap1 = (pl.scan_parquet(SAMPLE / "players_1hz.parquet")
             .select(["round", "seconds_since_live", "money", "equipment_value_calc", "armor"])
             .filter((pl.col("seconds_since_live") == 0) & (pl.col("round") == 1))
             .collect())
    buyers = snap1.filter(pl.col("armor") > 0)
    post_buy_check = {
        "round1_armor_buyers": int(buyers.height),
        "round1_armor_buyers_money_share_equal_150": float((buyers["money"] == 150).mean()),
        "round1_armor_buyers_money_plus_equipment_share_equal_1000": float(
            ((buyers["money"] + buyers["equipment_value_calc"]) == 1000).mean()),
    }

    checks = {
        "post_buy_money_check_round1": post_buy_check,
        "team_rounds_total": int(len(df)),
        "rounds_kept": int(df["round"].count() // 2),
        "matches": M,
        "matches_with_gaps_in_round_numbering": n_gap,
        "win_rate_all_rows": float(won.mean()),
        "equipment_identity_vs_round_buy_types_match": f"{eq_match}/{x.height}",
        "money_identity_vs_round_buy_types_match": f"{mo_match}/{x.height}",
        "team_rounds_loss_streak_exact_primary": int((primary & lok).sum()),
        "team_rounds_win_streak_exact_primary": int((primary & wok).sum()),
        "team_rounds_primary": int(primary.sum()),
        "pistol_round_info": pistol_info,
        "equipment_quintile_edges": [float(e) for e in edges],
        "expected_win_by_equipment_quintile": [float(e) for e in ehat],
    }

    result = {
        "id": "MC",
        "title": "Streak mechanism: economy after loss and win streaks",
        "source": ("data/state_sample/players_1hz.parquet (freeze-end snapshot, seconds_since_live == 0); "
                   "data/state_sample/rounds.parquet (ct_win); data/state_sample/round_buy_types.parquet "
                   "(identity check). Build: scripts/momentum/MC_01_team_rounds.py; analysis: "
                   "scripts/momentum/MC_02_streak_economy.py"),
        "definition": {
            "unit": "team-round (each round gives one row for each team)",
            "primary_population": "rounds 2 to max, excluding pistol round 13 (round 1 has no prior round)",
            "sensitivity_population": "rounds 2 to max, including round 13",
            "loss_streak": "consecutive losses immediately before the round, capped at 5 (5 = 5+); 0 = previous round won",
            "win_streak": "consecutive wins immediately before the round, capped at 5; 0 = previous round lost",
            "exact_streak": "run ends at an observed opposite round or at match start; runs that hit an unobserved "
                            "round before reaching 5 are excluded (censored)",
            "team_identity": "per match, the 5-player group containing the smallest player_id is team X; "
                             "partitions are fixed per match",
            "economy": "freeze-end snapshot (first live second): team equipment = sum equipment_value_calc; "
                       "team money = sum money",
            "excluded_rounds": "anomaly round m_7392c5c6 round 16 (9 players, flagged in the handoff); rounds whose "
                               "partition differs from the match's dominant partition (5 rounds in one match)",
        },
        "method": {
            "estimator": "pooled ratio of sums over team-rounds",
            "interval": "match-cluster bootstrap, 95% percentile (2.5/97.5, numpy linear)",
            "resamples": N_BOOT,
            "seed": SEED,
            "numpy_rng": "default_rng",
            "residual": "won minus primary-population mean win rate in the row's equipment-difference quintile",
        },
        "tables": {
            "primary": {"after_loss_streak": prim_loss, "after_win_streak": prim_win},
            "sensitivity_incl_round_13": {"after_loss_streak": sens_loss, "after_win_streak": sens_win},
        },
        "loss_bonus_ladder": ladder,
        "checks": checks,
        "key_numbers": {
            "win_rate_after_1_loss": prim_loss["win_rate"]["est"][1],
            "win_rate_after_1_win": prim_win["win_rate"]["est"][1],
            "win_rate_after_0_losses_prev_won": prim_loss["win_rate"]["est"][0],
            "win_rate_after_0_wins_prev_lost": prim_win["win_rate"]["est"][0],
            "team_equipment_after_0_losses": prim_loss["team_equipment"]["est"][0],
            "team_equipment_after_1_loss": prim_loss["team_equipment"]["est"][1],
            "team_equipment_after_5plus_losses": prim_loss["team_equipment"]["est"][5],
            "per_player_money_after_1_loss": prim_loss["per_player_money"]["est"][1],
            "per_player_money_after_5plus_losses": prim_loss["per_player_money"]["est"][5],
            "residual_after_1_loss_pp": prim_loss["win_rate_residual_given_equipment_diff"]["est"][1] * 100,
            "residual_after_1_win_pp": prim_win["win_rate_residual_given_equipment_diff"]["est"][1] * 100,
        },
        "caveats": [
            "Association only. Streak effects here are not causal estimates.",
            "Freeze-end money is the balance after the buy. Equipment at freeze end includes items carried over "
            "from earlier rounds, so money + equipment is not a pre-buy bankroll outside the pistol rounds. "
            "This is why no bankroll ladder is reported.",
            "Round 1 and round 13 money + equipment per player is about 1000 (median), consistent with 800 start "
            "money plus a 200 default pistol. This is an inference, not checked against game source.",
            "The loss-bonus ladder is not visible as a clean monotone rise in freeze-end money: money falls with "
            "streak length, and equipment rises only at 4 or more losses. Bonus received cannot be separated from "
            "spending with freeze-end snapshots alone.",
            "The residual conditions on equipment difference, which is itself a possible mediator of the streak "
            "effect; a small residual means economy explains the streak effect, it does not prove mechanism.",
            "Overtime side rule observed in the data: sides flip every 3 rounds from round 28 (rounds 25-27 keep the "
            "round-24 sides). The task text said swaps every 3 rounds after 24 in 6-round blocks; the data does not "
            "show a flip at round 25.",
            "Matches have gaps in round numbering (sample is partial in places); censored streaks are excluded, not "
            "imputed.",
            "The loss-bonus reference values are an external CS2 rule and were not verified in this data.",
            "Sample is 150 matches (days 01-07). Team-rounds from the same match are correlated, so the bootstrap "
            "resamples matches, not rows.",
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    write_svg(prim_loss, prim_win, OUT_SVG)

    # Console summary
    print("PRIMARY after loss streak (k = consecutive losses before round)")
    for j, k in enumerate(BUCKETS):
        print(f"  k={label(k):>2} n={prim_loss['team_rounds'][j]:5d} m={prim_loss['matches'][j]:3d} "
              f"win={prim_loss['win_rate']['est'][j]*100:5.1f} [{prim_loss['win_rate']['lo'][j]*100:5.1f},{prim_loss['win_rate']['hi'][j]*100:5.1f}] "
              f"teamEq={prim_loss['team_equipment']['est'][j]:8.0f} teamMoney={prim_loss['team_money']['est'][j]:8.0f} "
              f"ppMoney={prim_loss['per_player_money']['est'][j]:7.0f} ppEq={prim_loss['per_player_equipment']['est'][j]:7.0f} "
              f"resid={prim_loss['win_rate_residual_given_equipment_diff']['est'][j]*100:6.2f}pp "
              f"[{prim_loss['win_rate_residual_given_equipment_diff']['lo'][j]*100:6.2f},{prim_loss['win_rate_residual_given_equipment_diff']['hi'][j]*100:6.2f}]")
    print("PRIMARY after win streak (k = consecutive wins before round)")
    for j, k in enumerate(BUCKETS):
        print(f"  k={label(k):>2} n={prim_win['team_rounds'][j]:5d} m={prim_win['matches'][j]:3d} "
              f"win={prim_win['win_rate']['est'][j]*100:5.1f} [{prim_win['win_rate']['lo'][j]*100:5.1f},{prim_win['win_rate']['hi'][j]*100:5.1f}] "
              f"teamEq={prim_win['team_equipment']['est'][j]:8.0f} teamMoney={prim_win['team_money']['est'][j]:8.0f} "
              f"ppMoney={prim_win['per_player_money']['est'][j]:7.0f} ppEq={prim_win['per_player_equipment']['est'][j]:7.0f} "
              f"resid={prim_win['win_rate_residual_given_equipment_diff']['est'][j]*100:6.2f}pp "
              f"[{prim_win['win_rate_residual_given_equipment_diff']['lo'][j]*100:6.2f},{prim_win['win_rate_residual_given_equipment_diff']['hi'][j]*100:6.2f}]")
    print("LADDER (losing side)")
    for r in ladder:
        print("  ", r)
    print("SENSITIVITY incl r13 loss win:", [round(v*100,1) for v in sens_loss["win_rate"]["est"]],
          [round(v*100,1) for v in sens_win["win_rate"]["est"]])
    print("CHECKS", json.dumps({k: v for k, v in checks.items() if k != 'expected_win_by_equipment_quintile'}, default=str))
    print("wrote", OUT_JSON, OUT_SVG)


def write_svg(loss, win, path):
    W_, H_ = 800, 450
    lines = []
    add = lines.append
    add(f'<svg viewBox="0 0 {W_} {H_}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">')
    add(f'<rect x="0" y="0" width="{W_}" height="{H_}" fill="#ffffff"/>')
    add('<text x="400" y="26" text-anchor="middle" font-size="16" font-weight="bold" fill="#0f172a">'
        'Streak mechanism: economy and win rate by streak length</text>')
    add('<text x="400" y="44" text-anchor="middle" font-size="11" fill="#475569">'
        'Economy channel for momentum. Team-rounds, non-pistol rounds, 150-match 1 Hz sample. Association only.</text>')

    def panel(x0, x1, y0, y1, ymin, ymax, ticks, fmt, series, title, ylabel):
        def X(j):
            return x0 + (x1 - x0) * (j + 0.5) / len(BUCKETS)

        def Y(v):
            return y1 - (y1 - y0) * (v - ymin) / (ymax - ymin)

        add(f'<text x="{x0}" y="{y0-14}" font-size="13" font-weight="bold" fill="#0f172a">{title}</text>')
        for t in ticks:
            add(f'<line x1="{x0}" y1="{Y(t):.1f}" x2="{x1}" y2="{Y(t):.1f}" stroke="#e2e8f0" stroke-width="1"/>')
            add(f'<text x="{x0-6}" y="{Y(t)+4:.1f}" text-anchor="end" font-size="10" fill="#475569">{fmt(t)}</text>')
        add(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#475569" stroke-width="1"/>')
        add(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#475569" stroke-width="1"/>')
        for j, k in enumerate(BUCKETS):
            add(f'<text x="{X(j):.1f}" y="{y1+16}" text-anchor="middle" font-size="11" fill="#475569">{label(k)}</text>')
        add(f'<text x="{(x0+x1)/2:.1f}" y="{y1+34}" text-anchor="middle" font-size="11" fill="#475569">'
            f'Streak length before the round</text>')
        add(f'<text x="{x0-44}" y="{(y0+y1)/2:.1f}" text-anchor="middle" font-size="11" fill="#475569" '
            f'transform="rotate(-90 {x0-44} {(y0+y1)/2:.1f})">{ylabel}</text>')
        for s in series:
            pts = [(X(j), Y(v)) for j, v in enumerate(s["est"])]
            dash = ' stroke-dasharray="6 4"' if s["dash"] else ""
            poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            add(f'<polyline points="{poly}" fill="none" stroke="{NEUTRAL}" stroke-width="2"{dash}/>')
            for j, (px, py) in enumerate(pts):
                lo, hi = s["lo"][j], s["hi"][j]
                if lo is not None:
                    add(f'<line x1="{px:.1f}" y1="{Y(lo):.1f}" x2="{px:.1f}" y2="{Y(hi):.1f}" stroke="{NEUTRAL}" stroke-width="1" opacity="0.7"/>')
                    add(f'<line x1="{px-4:.1f}" y1="{Y(lo):.1f}" x2="{px+4:.1f}" y2="{Y(lo):.1f}" stroke="{NEUTRAL}" stroke-width="1" opacity="0.7"/>')
                    add(f'<line x1="{px-4:.1f}" y1="{Y(hi):.1f}" x2="{px+4:.1f}" y2="{Y(hi):.1f}" stroke="{NEUTRAL}" stroke-width="1" opacity="0.7"/>')
                fill = NEUTRAL if not s["dash"] else "#ffffff"
                add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{fill}" stroke="{NEUTRAL}" stroke-width="1.5"/>')

    eq_loss = {"est": [v / 1000 for v in loss["team_equipment"]["est"]],
               "lo": [v / 1000 for v in loss["team_equipment"]["lo"]],
               "hi": [v / 1000 for v in loss["team_equipment"]["hi"]], "dash": False}
    eq_win = {"est": [v / 1000 for v in win["team_equipment"]["est"]],
              "lo": [v / 1000 for v in win["team_equipment"]["lo"]],
              "hi": [v / 1000 for v in win["team_equipment"]["hi"]], "dash": True}
    eq_max = max(max(eq_loss["hi"]), max(eq_win["hi"]))
    eq_top = float(np.ceil(eq_max / 5) * 5)
    panel(84, 370, 90, 330, 0, eq_top, list(np.arange(0, eq_top + 0.1, 5)),
          lambda t: f"{t:.0f}", [eq_loss, eq_win],
          "A. Team equipment at freeze end", "Team equipment ($k, 5 players)")

    wr_loss = {"est": loss["win_rate"]["est"], "lo": loss["win_rate"]["lo"], "hi": loss["win_rate"]["hi"], "dash": False}
    wr_win = {"est": win["win_rate"]["est"], "lo": win["win_rate"]["lo"], "hi": win["win_rate"]["hi"], "dash": True}
    lo_all = min(min(wr_loss["lo"]), min(wr_win["lo"]))
    hi_all = max(max(wr_loss["hi"]), max(wr_win["hi"]))
    ymin = float(np.floor((lo_all - 0.03) / 0.05) * 0.05)
    ymax = float(np.ceil((hi_all + 0.03) / 0.05) * 0.05)
    panel(450, 740, 90, 330, ymin, ymax, list(np.arange(ymin, ymax + 1e-9, 0.05)),
          lambda t: f"{t*100:.0f}%", [wr_loss, wr_win],
          "B. Round win rate by streak", "Team win rate (95% CI)")

    # Legend
    add(f'<line x1="120" y1="378" x2="150" y2="378" stroke="{NEUTRAL}" stroke-width="2"/>')
    add(f'<circle cx="135" cy="378" r="4" fill="{NEUTRAL}" stroke="{NEUTRAL}" stroke-width="1.5"/>')
    add(f'<text x="158" y="382" font-size="11" fill="#0f172a">After a loss streak (k consecutive losses before the round)</text>')
    add(f'<line x1="120" y1="398" x2="150" y2="398" stroke="{NEUTRAL}" stroke-width="2" stroke-dasharray="6 4"/>')
    add(f'<circle cx="135" cy="398" r="4" fill="#ffffff" stroke="{NEUTRAL}" stroke-width="1.5"/>')
    add(f'<text x="158" y="402" font-size="11" fill="#0f172a">After a win streak (k consecutive wins before the round)</text>')
    add('<text x="400" y="426" text-anchor="middle" font-size="10" fill="#475569">'
        'Loss series streak 0 = previous round won; win series streak 0 = previous round lost.</text>')
    add('<text x="400" y="442" text-anchor="middle" font-size="10" fill="#475569">'
        'Bars: 95% match-cluster bootstrap (1000 resamples, seed 0). Association only, not causal.</text>')
    add("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
