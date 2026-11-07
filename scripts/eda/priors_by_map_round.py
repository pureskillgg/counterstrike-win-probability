"""Task C: pre-round CT win priors from the 2026 tomes (header, round_end, round_state only).

Clean rounds: header is_wingman False and round_end.player_count == 10 (5v5).
Outcome: ct_win = round_end.winner_team_code == 3 (side-based; winner codes 2 = T, 3 = CT).
Live start: first round_state row with event_type round_freeze_end.
Score difference: ct_score - t_score at live start, clipped to [-4, 4]. These columns follow the TEAM
across side swaps (scripts\\eda\\priors_score_diag2.py: every transition fits; swaps only at halftime and OT),
so this is the lead of the team now on CT over the team now on T.
Halves: rounds 1-12 and 13-24 (MR12; the halftime swap follows round 12). Overtime = rounds 25+.

Prior model (spec): smoothed CT rate table over (map_name, round bucket), round buckets 1..24 and OT.
  p_round = (k_r + A * p0) / (n_r + A),   p_map_round = (k_mr + B * p_round) / (n_mr + B),  A = B = 20 (fixed before the test run).
  Match-level 80/20 split: sklearn GroupShuffleSplit(test_size=0.2, random_state=0) on match_key.
  Compared against a constant guess (train CT rate), round-only and map-only tables.
Extension (not in the spec, labelled as such): the same two-level structure on the pre-round score difference.

Outputs: notes\\eda\\priors_by_map_round.md (all tables with counts and tests) and stdout.
Run: pixi run python -I scripts\\eda\\priors_by_map_round.py
"""
import collections
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
NOTE = ROOT / "notes" / "eda" / "priors_by_map_round.md"
T_SIDE, CT_SIDE = 2, 3
REG_ROUNDS = 24
HALF_ROUNDS = 12
OT_BUCKET = 25
SPLIT_SEED = 0
TEST_SIZE = 0.2
PSEUDO_ROUND = 20.0
PSEUDO_MAP = 20.0
ALPHA = 0.05
MIN_OMNI = 30  # cells with fewer rounds are left out of the chi-square omnibus test
MIN_MATCHES_KW = 5  # groups with fewer matches are left out of the match-level Kruskal-Wallis
BOOT = 1000

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def read(ch, cols=None):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def pct(x):
    return f"{100 * x:.2f}%"


def load():
    hdr = read("header", ["key", "match_date", "map_name", "is_wingman", "platform", "match_type", "max_rounds"])
    hdr = (hdr.rename({"key": "match_key"})
           .with_columns(pl.col("match_date").str.slice(0, 10).alias("day"))
           .with_columns(pl.when(pl.col("day") == "2026-10-08").then(pl.lit("new_1008"))
                         .when((pl.col("day") >= "2026-10-01") & (pl.col("day") <= "2026-10-07"))
                         .then(pl.lit("old_1001_1007"))
                         .otherwise(pl.lit("other_dates")).alias("batch")))
    re_ = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    rs = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    live = (rs.filter(pl.col("event_type") == "round_freeze_end")
            .sort(["match_key", "round", "tick"])
            .group_by(["match_key", "round"])
            .agg(pl.col("t_score").first().alias("t_live"), pl.col("ct_score").first().alias("ct_live")))
    rounds = (re_.join(hdr, on="match_key", how="left")
              .join(live, on=["match_key", "round"], how="left"))
    return hdr, rounds


def clean_rounds(rounds):
    c = rounds.filter((~pl.col("is_wingman")) & (pl.col("player_count") == 10))
    c = c.with_columns([
        (pl.col("winner_team_code") == CT_SIDE).cast(pl.Int64).alias("y"),
        pl.when(pl.col("round") > REG_ROUNDS).then(pl.lit(OT_BUCKET)).otherwise(pl.col("round")).alias("rb"),
        pl.when(pl.col("round") <= HALF_ROUNDS).then(pl.lit("1st half R1-12"))
          .when(pl.col("round") <= REG_ROUNDS).then(pl.lit("2nd half R13-24"))
          .otherwise(pl.lit("OT R25+")).alias("half"),
        pl.when(pl.col("round") == 1).then(pl.lit("R1 pistol"))
          .when(pl.col("round") == 13).then(pl.lit("R13 pistol"))
          .otherwise(pl.lit("non-pistol")).alias("pistol"),
        (pl.col("ct_live") - pl.col("t_live")).clip(-4, 4).alias("diff"),
    ])
    return c


def dim_table(df, col, label_fn=None, title=""):
    """Counts, CT win rate, binomial test vs the subset's overall rate, Bonferroni flag, chi-square omnibus."""
    p0 = df["y"].sum() / df.height
    g = (df.group_by(col).agg(pl.len().alias("n"), pl.col("y").sum().alias("k"),
                              pl.col("match_key").n_unique().alias("m"))
         .sort(col).to_dicts())
    ncell = len(g)
    bonf = ALPHA / max(ncell, 1)
    rows = []
    small = []
    for r in g:
        n, k = int(r["n"]), int(r["k"])
        pv = stats.binomtest(k, n, p0).pvalue
        lab = label_fn(r[col]) if label_fn else str(r[col])
        if n < MIN_OMNI:
            small.append(lab)
        rows.append([lab, r["m"], n, k, pct(k / n), f"{100 * (k / n - p0):+.2f}", f"{pv:.2g}",
                     "yes" if pv < bonf else ""])
    say(f"\n**{title}** (overall CT rate {pct(p0)} over {df.height} rounds; Bonferroni alpha {bonf:.4g} across {ncell} cells)\n")
    say(md_table(["cell", "matches", "rounds", "CT wins", "CT win rate", "diff vs overall (pp)",
                  "binomial p vs overall", "sig. after Bonferroni"], rows))
    use = [r for r in g if int(r["n"]) >= MIN_OMNI]
    if len(use) >= 2:
        tab = np.array([[r["k"] for r in use], [r["n"] - r["k"] for r in use]], dtype=float)
        chi2, pc, dof, expd = stats.chi2_contingency(tab)
        say(f"\nChi-square homogeneity over {len(use)} cells with >= {MIN_OMNI} rounds: chi2 {chi2:.2f}, df {dof}, "
            f"p = {pc:.2g}; min expected count {expd.min():.1f}. Cells left out of the omnibus test (< {MIN_OMNI} rounds): "
            f"{', '.join(small) if small else 'none'}.")
    return g


def match_level_kw(df, col, title):
    """Per-match CT share, Kruskal-Wallis across groups of a match-level variable (accounts for clustering)."""
    per = df.group_by(["match_key", col]).agg(pl.len().alias("n"), pl.col("y").sum().alias("k")).to_dicts()
    groups = collections.defaultdict(list)
    for r in per:
        groups[str(r[col])].append(r["k"] / r["n"])
    keep = {g: v for g, v in groups.items() if len(v) >= MIN_MATCHES_KW}
    dropped = {g: len(v) for g, v in groups.items() if len(v) < MIN_MATCHES_KW}
    h, p = stats.kruskal(*keep.values())
    say(f"Match-level check, {title}: Kruskal-Wallis on per-match CT share over {len(keep)} groups "
        f"({', '.join(f'{g} {len(v)} matches' for g, v in sorted(keep.items()))}): H {h:.2f}, p = {p:.2g}. "
        f"Groups with < {MIN_MATCHES_KW} matches left out: {dropped or 'none'}.")


# ---- prior model: map x round (spec) -----------------------------------------------------------

def fit(train, a_round=PSEUDO_ROUND, a_map=PSEUDO_MAP):
    n = len(train)
    p0 = sum(r["y"] for r in train) / n
    cnt_r, k_r = collections.Counter(), collections.Counter()
    cnt_mr, k_mr = collections.Counter(), collections.Counter()
    cnt_m, k_m = collections.Counter(), collections.Counter()
    for r in train:
        cnt_r[r["rb"]] += 1
        k_r[r["rb"]] += r["y"]
        key = (r["map_name"], r["rb"])
        cnt_mr[key] += 1
        k_mr[key] += r["y"]
        cnt_m[r["map_name"]] += 1
        k_m[r["map_name"]] += r["y"]
    p_r = {rb: (k_r[rb] + a_round * p0) / (cnt_r[rb] + a_round) for rb in cnt_r}
    p_mr = {key: (k_mr[key] + a_map * p_r[key[1]]) / (cnt_mr[key] + a_map) for key in cnt_mr}
    p_m = {m: (k_m[m] + a_round * p0) / (cnt_m[m] + a_round) for m in cnt_m}
    return dict(p0=p0, p_r=p_r, p_mr=p_mr, p_m=p_m, cells=len(cnt_mr))


def predict(model, rows, kind):
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        if kind == "full":
            out[i] = model["p_mr"].get((r["map_name"], r["rb"]), model["p_r"].get(r["rb"], model["p0"]))
        elif kind == "round":
            out[i] = model["p_r"].get(r["rb"], model["p0"])
        elif kind == "map":
            out[i] = model["p_m"].get(r["map_name"], model["p0"])
        else:
            out[i] = model["p0"]
    return out


# ---- extension: pre-round score difference ---------------------------------------------------

def fit_score(train_e, p0, a=PSEUDO_ROUND):
    cnt_d, k_d = collections.Counter(), collections.Counter()
    cnt_r, k_r = collections.Counter(), collections.Counter()
    cnt_rd, k_rd = collections.Counter(), collections.Counter()
    for r in train_e:
        cnt_d[r["diff"]] += 1
        k_d[r["diff"]] += r["y"]
        cnt_r[r["rb"]] += 1
        k_r[r["rb"]] += r["y"]
        key = (r["rb"], r["diff"])
        cnt_rd[key] += 1
        k_rd[key] += r["y"]
    p_d = {dv: (k_d[dv] + a * p0) / (cnt_d[dv] + a) for dv in cnt_d}
    p_r = {rb: (k_r[rb] + a * p0) / (cnt_r[rb] + a) for rb in cnt_r}
    p_rd = {key: (k_rd[key] + a * p_r[key[0]]) / (cnt_rd[key] + a) for key in cnt_rd}
    return dict(p0=p0, p_d=p_d, p_r=p_r, p_rd=p_rd)


def predict_score(m, rows, kind):
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        if kind == "score":
            out[i] = m["p_d"].get(r["diff"], m["p0"])
        else:
            out[i] = m["p_rd"].get((r["rb"], r["diff"]), m["p_r"].get(r["rb"], m["p0"]))
    return out


def metrics(y, p):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(ll=log_loss(y, pc, labels=[0, 1]), brier=brier_score_loss(y, pc),
                auc=roc_auc_score(y, pc) if len(np.unique(y)) == 2 else float("nan"))


def boot_diff(y, p_model, p_const, match_ids, rng, n_boot):
    """Match-level bootstrap of (log loss model - log loss constant) and (AUC - 0.5)."""
    eps = 1e-6
    pc = np.clip(p_model, eps, 1 - eps)
    qc = np.clip(p_const, eps, 1 - eps)
    ll_m = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    ll_c = -(y * np.log(qc) + (1 - y) * np.log(1 - qc))
    uniq = np.unique(match_ids)
    by_m = {m: np.where(match_ids == m)[0] for m in uniq}
    dll, dauc = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_m[m] for m in pick])
        dll.append(ll_m[idx].mean() - ll_c[idx].mean())
        if len(np.unique(y[idx])) == 2:
            dauc.append(roc_auc_score(y[idx], pc[idx]) - 0.5)
    return np.array(dll), np.array(dauc)


def run_prior(clean):
    d = clean.select(["match_key", "map_name", "rb", "y", "batch", "diff"]).sort(["match_key", "rb"])
    rows = d.to_dicts()
    groups = np.array([r["match_key"] for r in rows])
    y = np.array([r["y"] for r in rows], dtype=int)
    X = np.zeros((len(rows), 1))
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    train = [rows[i] for i in tr_idx]
    test = [rows[i] for i in te_idx]
    tr_m = set(groups[tr_idx])
    te_m = set(groups[te_idx])
    say(f"\nMatch split (seed {SPLIT_SEED}, test {TEST_SIZE}): train matches {len(tr_m)}, test matches {len(te_m)}; "
        f"train rounds {len(train)}, test rounds {len(test)}; overlap {len(tr_m & te_m)} matches.")

    model = fit(train)
    yt = np.array([r["y"] for r in test], dtype=int)
    ytr = np.array([r["y"] for r in train], dtype=int)
    te_match = np.array([r["match_key"] for r in test])
    say(f"Train CT rate {pct(model['p0'])}; test CT rate {pct(yt.mean())}; (map x round) cells seen in train: {model['cells']}.")

    p_full = predict(model, test, "full")
    p_round = predict(model, test, "round")
    p_map = predict(model, test, "map")
    p_const = np.full(len(test), model["p0"])
    res = {
        "constant guess (train CT rate)": metrics(yt, p_const),
        "round number only (smoothed)": metrics(yt, p_round),
        "map only (smoothed)": metrics(yt, p_map),
        "map x round (smoothed, primary)": metrics(yt, p_full),
    }
    base = res["constant guess (train CT rate)"]["ll"]
    rows_out = []
    for name, m in res.items():
        rows_out.append([name, f"{m['ll']:.4f}", f"{m['brier']:.4f}", f"{m['auc']:.3f}",
                         f"{100 * (1 - m['ll'] / base):+.2f}%"])
    say("\n**Prior model, test set (match-level 80/20, seed 0)**\n")
    say(md_table(["model", "test log loss", "test Brier", "test AUC", "log-loss skill vs constant"], rows_out))
    tr_ll = metrics(ytr, predict(model, train, "full"))["ll"]
    say(f"\nPrimary model in-sample (train) log loss {tr_ll:.4f} vs test {res['map x round (smoothed, primary)']['ll']:.4f} "
        f"(gap {res['map x round (smoothed, primary)']['ll'] - tr_ll:+.4f}).")
    say(f"Predicted CT probability on test: min {p_full.min():.3f}, max {p_full.max():.3f}, sd {p_full.std():.3f}.")

    rng = np.random.default_rng(SPLIT_SEED)
    dll, dauc = boot_diff(yt, p_full, p_const, te_match, rng, BOOT)
    lo, hi = np.percentile(dll, [2.5, 97.5])
    say(f"Bootstrap over test matches ({BOOT} resamples): log-loss difference model - constant = "
        f"{dll.mean():+.4f} (95% CI {lo:+.4f} to {hi:+.4f}); positive means the model is worse than the constant.")
    alo, ahi = np.percentile(dauc, [2.5, 97.5])
    say(f"Bootstrap AUC minus 0.5: {dauc.mean():+.4f} (95% CI {alo:+.4f} to {ahi:+.4f}).")

    sens = []
    for a in (5.0, 20.0, 100.0):
        mm = metrics(yt, predict(fit(train, a_round=a, a_map=a), test, "full"))
        sens.append([f"{a:g}", f"{mm['ll']:.4f}", f"{mm['brier']:.4f}", f"{mm['auc']:.3f}"])
    say("\nPseudo-count sensitivity (both levels set to the value shown; 20 is the primary setting):\n")
    say(md_table(["pseudo-count", "test log loss", "test Brier", "test AUC"], sens))

    edges = np.quantile(p_full, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(p_full, edges[1:-1])
    cal = []
    for b in range(5):
        msk = bins == b
        if msk.sum() == 0:
            continue
        cal.append([f"Q{b + 1}", int(msk.sum()), f"{p_full[msk].mean():.3f}", f"{yt[msk].mean():.3f}"])
    say("\nCalibration on test, quintiles of predicted CT probability:\n")
    say(md_table(["quintile", "rounds", "mean predicted", "observed CT rate"], cal))

    old_msk = np.array([r["batch"] == "old_1001_1007" for r in test])
    mo = metrics(yt[old_msk], p_full[old_msk])
    mc = metrics(yt[old_msk], p_const[old_msk])
    say(f"\nTest rounds from 2026-10-01 to 10-07 only: n {int(old_msk.sum())}; primary model log loss {mo['ll']:.4f}, "
        f"AUC {mo['auc']:.3f}; constant log loss {mc['ll']:.4f}.")

    # ---- extension: pre-round score difference (not in the spec) ----
    say("\n### Extension (not in the spec): pre-round score difference")
    train_e = [r for r in train if r["diff"] is not None]
    test_e = [r for r in test if r["diff"] is not None]
    yte = np.array([r["y"] for r in test_e], dtype=int)
    mid_e = np.array([r["match_key"] for r in test_e])
    say(f"Rounds with a live-start score: train {len(train_e)}, test {len(test_e)} (rounds without one are dropped here only). "
        f"Score difference = CT-team lead at live start, clipped to +/-4; the model is fitted on train only.")
    sc = fit_score(train_e, model["p0"])
    p_sc = predict_score(sc, test_e, "score")
    p_rsc = predict_score(sc, test_e, "round_score")
    p_fe = predict(model, test_e, "full")
    p_ce = np.full(len(test_e), model["p0"])
    ext = {
        "constant guess (train CT rate)": (metrics(yte, p_ce), p_ce),
        "map x round (primary, spec)": (metrics(yte, p_fe), p_fe),
        "score difference only (smoothed)": (metrics(yte, p_sc), p_sc),
        "round x score difference (smoothed)": (metrics(yte, p_rsc), p_rsc),
    }
    base_e = ext["constant guess (train CT rate)"][0]["ll"]
    rows_e = []
    for name, (m, _) in ext.items():
        rows_e.append([name, f"{m['ll']:.4f}", f"{m['brier']:.4f}", f"{m['auc']:.3f}",
                       f"{100 * (1 - m['ll'] / base_e):+.2f}%"])
    say("\n**Test set with a live-start score (same match split)**\n")
    say(md_table(["model", "test log loss", "test Brier", "test AUC", "log-loss skill vs constant"], rows_e))
    for name, p_model in (("score difference only", p_sc), ("round x score difference", p_rsc)):
        rng_e = np.random.default_rng(SPLIT_SEED)
        dl, da = boot_diff(yte, p_model, p_ce, mid_e, rng_e, BOOT)
        lo, hi = np.percentile(dl, [2.5, 97.5])
        alo, ahi = np.percentile(da, [2.5, 97.5])
        say(f"- {name} minus constant: log-loss difference {dl.mean():+.4f} (95% CI {lo:+.4f} to {hi:+.4f}); "
            f"AUC minus 0.5 {da.mean():+.4f} (95% CI {alo:+.4f} to {ahi:+.4f}).")
    p_sc_train = predict_score(sc, train_e, "score")
    say(f"- score-only model in-sample (train) log loss {metrics(np.array([r['y'] for r in train_e]), p_sc_train)['ll']:.4f}; "
        f"predicted CT probability on test: min {p_sc.min():.3f}, max {p_sc.max():.3f}.")
    return res


def main():
    hdr, rounds = load()
    say("# Task C: pre-round CT win priors (map, round, halves, pistol, score difference, platform, match_type)")
    say("")
    say("Source: 2026 tomes at G:\\pureskill\\csds-cs2\\tomes\\2026-10-01_to_08\\tome\\csds (header, round_end, round_state). "
        "No E: reads. Script: scripts\\eda\\priors_by_map_round.py. Run date 2026-10-13.")

    say("\n## 1. Scope")
    n_all = hdr.height
    wing = hdr.filter(pl.col("is_wingman")).height
    say(f"- header matches {n_all}; wingman {wing} (max_rounds 16); non-wingman {n_all - wing} (max_rounds 24).")
    total_rounds = rounds.height
    ntw = rounds.filter(pl.col("is_wingman")).height
    clean = clean_rounds(rounds)
    nonclean = rounds.filter((~pl.col("is_wingman")) & (pl.col("player_count") != 10))
    say(f"- round_end rows {total_rounds}; wingman rounds {ntw}; non-wingman rounds not 5v5 {nonclean.height} "
        f"(of which player_count 9: {nonclean.filter(pl.col('player_count') == 9).height}).")
    say(f"- clean 5v5 rounds {clean.height} in {clean['match_key'].n_unique()} matches.")
    nolive = clean.filter(pl.col("t_live").is_null()).height
    say(f"- clean rounds with no round_freeze_end row (score-based tables exclude them): {nolive}.")
    nb = clean.filter(pl.col("batch") == "new_1008").height
    ob = clean.filter(pl.col("batch") == "old_1001_1007").height
    od = clean.filter(pl.col("batch") == "other_dates").height
    say(f"- Batch: 2026-10-08 has {nb} clean rounds ({100 * nb / clean.height:.1f}% of clean rounds); "
        f"2026-10-01..10-07 has {ob}; other dates {od}.")

    say("\n## 2. Verified structure (from scripts\\eda\\priors_probe.py, priors_score_diag.py, priors_score_diag2.py, priors_playercount_check.py)")
    say("- Halves are 12 rounds, not 24. The phase label flips to 'second' at round 12 for 1,566 non-wingman matches (20 at round 13; wingman halves at round 8). The task text's '24-round halves' is contradicted by the data.")
    say("- round_end.winner_team_code is side-based (2 = T, 3 = CT), consistent with the win_reason pairing in schema_dictionary.md.")
    say("- round_state t_score/ct_score follow the TEAM across side swaps, not the side. Every r -> r+1 transition (34,200 pairs, wingman included) fits at least one of two hypotheses; none fits neither. At the halftime swap after round 12, the 1,329 decisive pairs fit only the swap form and 255 fit both (tied post-round scores, so uninformative); at round 13 and most other rounds the decisive pairs fit only the side-stay form. OT swap-only pairs occur after rounds 27 (163 matches) and 33 (37 matches). The schema_dictionary.md section 7 claim 'scores are cumulative per side' is contradicted here.")
    say("- score_update has non-unit deltas (3,044 rows, e.g. +8 or -8) at the same swaps, consistent with the team-based score.")
    say("- Round 1 live start is 0-0 in every match. Live start exists for 35,953 of 35,992 round_end rounds (39 missing).")
    say("- Overtime rounds exist: 1,165 non-wingman rounds in 163 matches (rounds 25 to 48).")
    say("- The 10-player filter drops 1,913 non-wingman rounds (5.6%): 1,380 are 9-player rounds, and the non-clean share is about 3% in rounds 1 to 9, 6% in round 12, 7% in round 15 and 21% in round 24 (matches that end early). Non-clean CT rates differ from clean ones in some rounds (for example R9: 61% versus 55%), but each round has only about 40 to 100 non-clean rounds, so the filter is a weak selection risk (scripts\\eda\\priors_playercount_check.py).")

    say("\n## 3. CT win rate tables (clean 5v5 rounds)")
    dim_table(clean, "rb", label_fn=lambda v: "OT R25+" if v == OT_BUCKET else f"R{v}",
              title="By round number (OT pooled)")
    say("Each match contributes at most one round to each regulation round-number cell (R1 to R24), so those cells are independent across matches. The pooled OT cell can hold several rounds from one match, so its test is optimistic.")
    dim_table(clean, "half", title="First half versus second half versus overtime")
    say("Caveat: a match contributes up to 12 rounds to each half, so the half tests ignore within-match correlation and are optimistic.")
    dim_table(clean, "pistol", title="Pistol rounds (R1, R13) versus other rounds; CT rate, T rate = 1 - CT")
    dim_table(clean, "map_name", title="By map_name")
    match_level_kw(clean, "map_name", "map_name")
    scored = clean.filter(pl.col("diff").is_not_null())
    dim_table(scored, "diff",
              label_fn=lambda v: "<= -4" if v == -4 else (">= +4" if v == 4 else (f"{int(v):+d}" if v else "0")),
              title="By score difference going into the round (ct_score - t_score at live start, team-based, clipped to +/-4)")
    say("Caveat: rounds within a match share the score history, so the omnibus chi-square is optimistic; the effect sizes are large enough that this should not change the reading.")
    dim_table(clean, "platform", title="By match platform (header)")
    match_level_kw(clean, "platform", "platform")
    dim_table(clean, "match_type", title="By match_type (header)")
    match_level_kw(clean, "match_type", "match_type")

    say("\n### Platform and match_type are confounded with batch")
    cross = (clean.group_by(["batch", "platform"])
             .agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches"),
                  pl.col("y").sum().alias("ct_wins"))
             .sort(["batch", "platform"]).to_dicts())
    say(md_table(["batch", "platform", "matches", "rounds", "CT wins", "CT win rate"],
                 [[r["batch"], r["platform"], r["matches"], r["rounds"], r["ct_wins"],
                   pct(r["ct_wins"] / r["rounds"])] for r in cross]))
    say("\nWithin 2026-10-08 only (steam versus unknown platform):")
    dim_table(clean.filter(pl.col("batch") == "new_1008"), "platform", title="2026-10-08 only, by platform")
    say("\nWithin 2026-10-01..10-07 only (steam versus faceit):")
    dim_table(clean.filter(pl.col("batch") == "old_1001_1007"), "platform", title="2026-10-01..10-07 only, by platform")

    say("\n## 4. Prior model: map and round number only")
    run_prior(clean)

    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(OUT) + "\n", encoding="utf-8")
    say(f"\nWrote {NOTE}")


if __name__ == "__main__":
    main()
