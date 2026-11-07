"""X13: molotov / incendiary fires on the planted bomb, 2026 tome (1,752 matches).

Primary set: clean 5v5 rounds (round_end.player_count == 10) with bomb_planted inside the live window
(round_freeze_end tick < plant tick < round_end tick) and a bomb-position proxy.
ASSUMPTIONS (printed in the log):
  bomb position = planter (x, y) at the last bomb_begin_plant at or before the bomb_planted tick (no bomb coordinate in data)
  fire centre = inferno_startburn (x_pos, y_pos); fire end = inferno_expire tick, else round_end tick
  thrower side = player_info.team_code for thrower_pid in the same round (2=T, 3=CT), else unknown
  defuse completion = bomb_defused by the same player within 12 s of the attempt and before that player's next attempt
  defuser hurt = player_hurt weapon_name 'inferno' or 'molotov' within 5 s after the attempt
All results are descriptive, not causal. Tick rate 64.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
DATE = "2026-09-18,2026-10-08"
TICK = 64
R_LIST = [100.0, 150.0, 200.0]
N_BOOT = 1000
SEED = 0
KEYS = ["match_key", "round"]
OUTCOMES = ["defuse", "explosion", "ct_elim", "t_elim", "other"]
LABELS = OUTCOMES + ["union(defuse|explosion|ct_elim)"]
MAP_MIN = 100
DEF_WIN = 12 * TICK
HURT_WIN = 5 * TICK
FIRE_WORDS = ("inferno", "molotov", "incend")


def P(*a):
    print(*a, flush=True)


def chan(ch):
    return pl.scan_parquet(os.path.join(ROOT, f"basic_{ch}.{DATE}", "dataframe_*"))


def sh(k, n):
    return f"{int(k)}/{int(n)} = {100.0 * k / n:.1f}%" if n else f"{int(k)}/0 = n/a"


def side_expr(col):
    return (pl.when(pl.col(col) == 2).then(pl.lit("T"))
            .when(pl.col(col) == 3).then(pl.lit("CT"))
            .otherwise(pl.lit("unknown")).alias("side"))


def add_union(D):
    return np.concatenate([D, D[..., :3].sum(axis=-1, keepdims=True)], axis=-1)


def boot_matrix(n_units, seed):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_units, size=(N_BOOT, n_units))
    W = np.zeros((N_BOOT, n_units), dtype=np.float64)
    np.add.at(W, (np.repeat(np.arange(N_BOOT), n_units), draws.ravel()), 1.0)
    return W


def crude(W, N_g, D_g):
    n1 = W @ N_g[1]
    n0 = W @ N_g[0]
    d1 = W @ D_g[1]
    d0 = W @ D_g[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = d1 / n1[:, None]
        r0 = d0 / n0[:, None]
    return r1, r0, r1 - r0


def adjusted(W, N_gb, D_gb):
    n1 = W @ N_gb[1]
    n0 = W @ N_gb[0]
    d1 = np.einsum("bm,mjk->bjk", W, D_gb[1])
    d0 = np.einsum("bm,mjk->bjk", W, D_gb[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = d1 / n1[:, :, None]
        r0 = d0 / n0[:, :, None]
        valid = (n1 > 0) & (n0 > 0)
        w = np.where(valid, n1 + n0, 0.0)
        diff = np.where(valid[:, :, None], r1 - r0, 0.0)
        tot = w.sum(axis=1)
        adj = (w[:, :, None] * diff).sum(axis=1) / tot[:, None]
    return adj


def ci(x):
    return np.nanpercentile(x, [2.5, 97.5], axis=0)


def fmt(v):
    return "n/a" if v is None or np.isnan(v) else f"{100 * v:.1f}%"


def fmt_ci(lohi):
    return f"[{fmt(lohi[0])}, {fmt(lohi[1])}]"


def main():
    P("== X13: molotov / incendiary fires on the planted bomb (2026 tome, 1,752 matches)")
    P("ASSUMPTIONS:")
    P("  bomb position = planter (x,y) at last bomb_begin_plant at/before bomb_planted tick (no bomb coordinate in data)")
    P("  fire centre = inferno_startburn (x_pos,y_pos); fire end = inferno_expire tick, else round_end tick")
    P("  thrower side = player_info.team_code for thrower_pid in same round (2=T,3=CT); else unknown")
    P("  defuse completion: bomb_defused by same player within 12 s of attempt, before that player's next attempt")
    P("  defuser hurt: player_hurt weapon_name inferno/molotov within 5 s after attempt; radii 100/150/200; main 150")

    # ---------- rounds, maps
    # header 'key' has the same format as round_end.match_key (csds/2026/10/01/<uuid>/csds); 'match_id' is the bare uuid
    hdr = (pl.scan_parquet(os.path.join(ROOT, "basic_round_end." + DATE, "header", "dataframe_*"))
           .select(["key", "map_name"]).collect().rename({"key": "match_key"}))
    rounds = (chan("round_end").select(KEYS + ["tick", "winner_team_code", "win_reason_code", "player_count"])
              .rename({"tick": "end_tick"}).collect())
    frz = (chan("round_state").filter(pl.col("event_type") == "round_freeze_end")
           .select(KEYS + [pl.col("tick").alias("freeze_tick")]).collect())
    rounds = (rounds.join(frz, on=KEYS, how="left")
              .join(hdr, on="match_key", how="left")
              .with_columns(pl.col("map_name").fill_null("unknown")))
    matches = sorted(rounds["match_key"].unique().to_list())
    M = len(matches)
    midx_map = {m: i for i, m in enumerate(matches)}
    P(f"header rows: {hdr.height}; matches in round_end: {M}; rounds: {rounds.height}")
    P(f"rounds with map unknown: {rounds.filter(pl.col('map_name') == 'unknown').height}; rounds without freeze_end: {rounds.filter(pl.col('freeze_tick').is_null()).height}")
    P("round_end (winner_team_code, win_reason_code, n):",
      rounds.group_by(["win_reason_code", "winner_team_code"]).agg(pl.len().alias("n")).sort(["win_reason_code", "winner_team_code"]).rows())
    P("round_end player_count distribution:",
      rounds.group_by("player_count").agg(pl.len().alias("n")).sort("player_count").rows())

    # ---------- plants
    planted = (chan("bomb_state").filter(pl.col("event_type") == "bomb_planted")
               .select(KEYS + [pl.col("tick").alias("plant_tick"), pl.col("player_id_fixed").alias("planter_pid")])
               .collect())
    pr = planted.join(rounds, on=KEYS, how="left")
    live = pr.filter((pl.col("plant_tick") > pl.col("freeze_tick")) & (pl.col("plant_tick") < pl.col("end_tick")))
    live5 = live.filter(pl.col("player_count") == 10)
    P("== STEP 1: planted rounds")
    P(f"bomb_planted events: {pr.height}; no round_end row: {pr.filter(pl.col('end_tick').is_null()).height}; "
      f"plant at/after round_end (excluded): {pr.filter(pl.col('plant_tick') >= pl.col('end_tick')).height}; "
      f"plant at/before freeze_end (excluded): {pr.filter(pl.col('plant_tick') <= pl.col('freeze_tick')).height}")
    P(f"planted inside live window, all player counts: {live.height} (matches {live['match_key'].n_unique()})")
    P(f"planted inside live window, clean 5v5 (player_count==10): {live5.height} (matches {live5['match_key'].n_unique()})")
    rp = rounds.join(planted.select(KEYS + ["plant_tick"]), on=KEYS, how="left").with_columns(
        (pl.col("plant_tick").is_not_null() & (pl.col("plant_tick") < pl.col("end_tick"))).alias("planted_before_end"))
    P("  win_reason x planted-before-end (reason, winner_team, planted_before_end, n) [2=T, 3=CT]:",
      rp.group_by(["win_reason_code", "winner_team_code", "planted_before_end"]).agg(pl.len().alias("n"))
      .sort(["win_reason_code", "planted_before_end"]).rows())
    P("  NOTE: brief labels 8='CTs eliminated' and 9='Ts eliminated' do not match winner_team_code; "
      "used 9 = CTs eliminated (T win) and 8 = Ts eliminated (CT win)")

    # ---------- bomb position proxy
    bp = (chan("bomb_action")
          .filter((pl.col("event_type") == "bomb_begin_plant") & pl.col("player_x_pos").is_not_null() & pl.col("player_y_pos").is_not_null())
          .select(KEYS + [pl.col("tick").alias("bp_tick"), pl.col("player_x_pos").alias("bp_x"),
                          pl.col("player_y_pos").alias("bp_y"), pl.col("player_id_fixed").cast(pl.Int64).alias("bp_pid")])
          .collect())

    # ---------- CT deaths (victim team from player_info, fallback player_death.player_team_code)
    pinfo = chan("player_info").select(KEYS + ["player_id_fixed", "team_code"]).collect()
    pdth = (chan("player_death").select(KEYS + ["tick", "player_id_fixed", "player_team_code"])
            .filter(pl.col("player_id_fixed").is_not_null())
            .with_columns(pl.col("player_id_fixed").cast(pl.Int64)).collect())
    pdth = (pdth.join(pinfo, on=KEYS + ["player_id_fixed"], how="left")
            .with_columns(pl.coalesce(pl.col("team_code"), pl.col("player_team_code")).alias("vteam")))
    CTD = pdth.filter(pl.col("vteam") == 3).select(KEYS + ["tick", "player_id_fixed"])

    # ---------- fires (inferno_startburn, paired to expire)
    mol = (chan("molotov_state").select(KEYS + ["tick", "event_type", "entity_id_fixed", "x_pos", "y_pos",
                                                "player_id_fixed", "player_team_code", "was_extinguished_by_smoke",
                                                "was_extinguished_by_thrown_smoke", "fraction_extinguished",
                                                "was_thrown_into_smoke"]).collect())
    sbr = mol.filter(pl.col("event_type") == "inferno_startburn")
    exr = mol.filter(pl.col("event_type") == "inferno_expire")
    P("== smoke-extinguish flags (startburn rows)")
    P(f"  startburn rows: {sbr.height}; expire rows: {exr.height}")
    P(f"  was_extinguished_by_smoke=True: {sbr.filter(pl.col('was_extinguished_by_smoke') == True).height}; "
      f"was_extinguished_by_thrown_smoke=True: {sbr.filter(pl.col('was_extinguished_by_thrown_smoke') == True).height}; "
      f"was_thrown_into_smoke=True: {sbr.filter(pl.col('was_thrown_into_smoke') == True).height}")
    P(f"  fraction_extinguished>0: {sbr.filter(pl.col('fraction_extinguished') > 0).height}; >=1: {sbr.filter(pl.col('fraction_extinguished') >= 1).height}")

    sb = sbr.select(KEYS + [pl.col("tick").alias("start_tick"), "entity_id_fixed", pl.col("x_pos").alias("fx"),
                            pl.col("y_pos").alias("fy"), pl.col("player_id_fixed").alias("thrower_pid"),
                            pl.col("player_team_code").alias("mol_team"), "was_extinguished_by_smoke",
                            "fraction_extinguished"])
    ex = exr.select(KEYS + ["entity_id_fixed", pl.col("tick").alias("expire_tick")])
    fires = (sb.join(ex, on=KEYS + ["entity_id_fixed"], how="left")
             .join(rounds.select(KEYS + ["end_tick"]), on=KEYS, how="left")
             .with_columns(pl.coalesce(pl.col("expire_tick"), pl.col("end_tick")).alias("fire_end"))
             .drop("end_tick"))
    pi_thr = pinfo.rename({"player_id_fixed": "thrower_pid", "team_code": "thrower_team"})
    fires = fires.join(pi_thr, on=KEYS + ["thrower_pid"], how="left").with_columns(side_expr("thrower_team"))
    P(f"  startburn with no expire row (fire_end = round end): {fires.filter(pl.col('expire_tick').is_null()).height}")
    P(f"  thrower_pid null: {fires.filter(pl.col('thrower_pid').is_null()).height}")
    P("  thrower side (player_info), all startburn rows:", fires.group_by("side").agg(pl.len().alias("n")).sort("side").rows())
    xk = fires.filter((pl.col("side") != "unknown") & pl.col("mol_team").is_not_null())
    P(f"  cross-check player_info side vs molotov_state.player_team_code (n={xk.height}):",
      xk.group_by(["side", "mol_team"]).agg(pl.len().alias("n")).sort(["side", "mol_team"]).rows())

    # ---------- build planted table with proxy, CT alive, fire counts
    def build(live_df):
        cand = (live_df.select(KEYS + ["plant_tick", "planter_pid"])
                .join(bp, on=KEYS, how="inner")
                .filter(pl.col("bp_tick") <= pl.col("plant_tick")))
        last = cand.group_by(KEYS).agg(pl.col("bp_tick").max().alias("bp_tick"))
        proxy = cand.join(last, on=KEYS + ["bp_tick"], how="inner").unique(subset=KEYS, keep="first")
        A = live_df.join(proxy.select(KEYS + ["bp_x", "bp_y", "bp_tick", "bp_pid"]), on=KEYS, how="inner")
        ctpre = (live_df.select(KEYS + ["plant_tick"]).join(CTD, on=KEYS, how="inner")
                 .filter(pl.col("tick") < pl.col("plant_tick"))
                 .group_by(KEYS).agg(pl.col("player_id_fixed").n_unique().alias("ct_dead")))
        A = (A.join(ctpre, on=KEYS, how="left")
             .with_columns(pl.col("ct_dead").fill_null(0))
             .with_columns((5 - pl.col("ct_dead")).clip(0, 5).alias("ct_alive"))
             # win_reason mapping checked against winner_team_code (see crosstab in STEP 1):
             # 7 defuse (CT win), 1 explosion (T win), 9 CTs eliminated (T win), 8 Ts eliminated (CT win, pre-plant only)
             .with_columns(pl.when(pl.col("win_reason_code") == 7).then(pl.lit("defuse"))
                           .when(pl.col("win_reason_code") == 1).then(pl.lit("explosion"))
                           .when(pl.col("win_reason_code") == 9).then(pl.lit("ct_elim"))
                           .when(pl.col("win_reason_code") == 8).then(pl.lit("t_elim"))
                           .otherwise(pl.lit("other")).alias("outcome")))
        fa = (fires.join(A.select(KEYS + ["plant_tick", "end_tick", "bp_x", "bp_y"]), on=KEYS, how="inner")
              .filter((pl.col("start_tick") > pl.col("plant_tick")) & (pl.col("start_tick") < pl.col("end_tick")))
              .with_columns(((pl.col("fx") - pl.col("bp_x")) ** 2 + (pl.col("fy") - pl.col("bp_y")) ** 2).sqrt().alias("dist")))
        for R in R_LIST:
            r = int(R)
            agg = fa.filter(pl.col("dist") <= R).group_by(KEYS).agg([
                (pl.col("side") == "T").sum().alias(f"nT{r}"),
                (pl.col("side") == "CT").sum().alias(f"nCT{r}"),
                (pl.col("side") == "unknown").sum().alias(f"nU{r}"),
                pl.col("start_tick").filter(pl.col("side") == "T").min().alias(f"firstT{r}"),
            ])
            A = A.join(agg, on=KEYS, how="left")
            A = A.with_columns([pl.col(f"nT{r}").fill_null(0), pl.col(f"nCT{r}").fill_null(0),
                                pl.col(f"nU{r}").fill_null(0)])
            A = A.with_columns((pl.col(f"nT{r}") > 0).alias(f"hasT{r}"))
        A = A.with_columns(((pl.col("firstT150") - pl.col("plant_tick")) / TICK).alias("sec_firstT150"))
        return A, fa

    A5, fa5 = build(live5)
    Aall, faall = build(live)
    P(f"clean 5v5 planted rounds with bomb proxy (analysis set A): {A5.height}; "
      f"without proxy (no bomb_begin_plant before plant): {live5.height - A5.height}")
    P(f"  proxy planter pid == bomb_planted planter pid: {A5.filter(pl.col('bp_pid') == pl.col('planter_pid')).height}/{A5.height}")
    P(f"  bomb_planted tick minus proxy begin_plant tick (ticks, median): {A5.select((pl.col('plant_tick') - pl.col('bp_tick')).median()).item()}")
    P(f"  A rounds by outcome: {A5.group_by('outcome').agg(pl.len().alias('n')).sort('outcome').rows()}")
    P(f"  CT alive at plant (5 - CT deaths before plant), counts: {A5.group_by('ct_alive').agg(pl.len().alias('n')).sort('ct_alive').rows()}")

    # ---------- STEP 2
    P("== STEP 2: fires starting after the plant and before round end, centre within R of proxy (set A)")
    P(f"  fires starting after plant and before round end, any distance: {fa5.height} in {A5.height} rounds")
    for R in R_LIST:
        sub = fa5.filter(pl.col("dist") <= R)
        cnt = {s: int(sub.filter(pl.col("side") == s).height) for s in ["T", "CT", "unknown"]}
        P(f"  R={int(R)}: total {sub.height} | T-thrown {cnt['T']} | CT-thrown {cnt['CT']} | unknown thrower {cnt['unknown']} | "
          f"rounds with >=1 fire {sub.select(KEYS).unique().height}")

    # ---------- STEP 3
    P("== STEP 3: share of planted rounds with >=1 T-thrown fire on the bomb")
    N5 = A5.height
    for R in R_LIST:
        P(f"  R={int(R)}: {sh(A5.filter(pl.col(f'hasT{int(R)}')).height, N5)}")
    P(f"  sensitivity (all live planted rounds, any player count, R=150): "
      f"{sh(Aall.filter(pl.col('hasT150')).height, Aall.height)} (n={Aall.height})")
    bym = (A5.group_by("map_name").agg([pl.len().alias("n"), pl.col("hasT150").sum().alias("nT")])
           .sort("n", descending=True))
    P("  by map (R=150; clean 5v5 planted):")
    small_n = 0
    small_k = 0
    for row in bym.iter_rows(named=True):
        if row["n"] >= MAP_MIN:
            P(f"    {row['map_name']:<14} {sh(row['nT'], row['n'])}  (n={row['n']})")
        else:
            small_n += row["n"]
            small_k += row["nT"]
    P(f"    maps with <{MAP_MIN} planted rounds pooled: {sh(small_k, small_n)} (n={small_n})")
    T5 = A5.filter(pl.col("hasT150"))
    P(f"  seconds from plant to first T-thrown fire (per round, R=150): n={T5.height}, median={T5['sec_firstT150'].median():.2f} s, "
      f"share <=10 s of those rounds={sh(T5.filter(pl.col('sec_firstT150') <= 10).height, T5.height)}, "
      f"share <=10 s of all planted={sh(A5.filter(pl.col('sec_firstT150') <= 10).height, N5)}")
    fT = (fa5.filter((pl.col("side") == "T") & (pl.col("dist") <= 150.0))
          .with_columns(((pl.col("start_tick") - pl.col("plant_tick")) / TICK).alias("sec")))
    P(f"  per T-thrown fire within 150 (after plant): n={fT.height}, median={fT['sec'].median():.2f} s, "
      f"q25={fT['sec'].quantile(0.25):.2f} s, q75={fT['sec'].quantile(0.75):.2f} s, "
      f"share <=10 s={sh(fT.filter(pl.col('sec') <= 10).height, fT.height)}")

    # ---------- STEP 4
    P("== STEP 4: outcomes, T-thrown fire on bomb (R=150) vs none; match-cluster bootstrap (B=1000, seed 0)")
    midx = np.array([midx_map[m] for m in A5["match_key"].to_list()], dtype=np.int64)
    g = A5["hasT150"].cast(pl.Int64).to_numpy().astype(np.int64)
    oc = np.array([OUTCOMES.index(o) for o in A5["outcome"].to_list()], dtype=np.int64)
    cb = A5["ct_alive"].cast(pl.Int64).to_numpy().astype(np.int64)
    N_g = np.zeros((2, M)); np.add.at(N_g, (g, midx), 1.0)
    D_g = np.zeros((2, M, len(OUTCOMES))); np.add.at(D_g, (g, midx, oc), 1.0)
    N_gb = np.zeros((2, M, 6)); np.add.at(N_gb, (g, midx, cb), 1.0)
    D_gb = np.zeros((2, M, 6, len(OUTCOMES))); np.add.at(D_gb, (g, midx, cb, oc), 1.0)
    D_g = add_union(D_g)
    D_gb = add_union(D_gb)
    W = boot_matrix(M, SEED)
    W1 = np.ones((1, M))
    P(f"  groups: with T-fire n={int(N_g[1].sum())}, without n={int(N_g[0].sum())}; matches resampled: {M}")
    r1p, r0p, dp = crude(W1, N_g, D_g)
    _, _, db = crude(W, N_g, D_g)
    dci = ci(db)
    r1b, r0b, _ = crude(W, N_g, D_g)
    r1ci = ci(r1b)
    r0ci = ci(r0b)
    P(f"  {'outcome':<34}{'with T-fire':>12}{'without':>12}{'diff (w-wo)':>13}  diff 95% CI")
    for k, lab in enumerate(LABELS):
        P(f"  {lab:<34}{fmt(r1p[0, k]):>12}{fmt(r0p[0, k]):>12}{fmt(dp[0, k]):>13}  {fmt_ci(dci[:, k])}")
    P("  95% CI for with-fire share: " + "; ".join(f"{LABELS[k]} {fmt_ci(r1ci[:, k])}" for k in range(len(LABELS))))
    P("  95% CI for without-fire share: " + "; ".join(f"{LABELS[k]} {fmt_ci(r0ci[:, k])}" for k in range(len(LABELS))))

    P("  by CT alive at plant (bins), rates shown as with/without:")
    P(f"  {'CT alive':<9}{'n_with':>8}{'n_wo':>8}{'defuse':>24}{'explosion':>24}{'ct_elim':>24}")
    for b in range(6):
        n1 = int(N_gb[1, :, b].sum())
        n0 = int(N_gb[0, :, b].sum())
        if n1 + n0 == 0:
            continue
        cells = []
        for k in range(3):
            r1 = D_gb[1, :, b, k].sum() / n1 if n1 else np.nan
            r0 = D_gb[0, :, b, k].sum() / n0 if n0 else np.nan
            cells.append(f"{fmt(r1)}/{fmt(r0)}")
        P(f"  {b:<9}{n1:>8}{n0:>8}" + "".join(f"{c:>24}" for c in cells))
    adj_p = adjusted(W1, N_gb, D_gb)
    adj_b = adjusted(W, N_gb, D_gb)
    adj_ci = ci(adj_b)
    P("  bin-adjusted difference (bins observed in both groups; weights = pooled bin size), with-minus-without:")
    for k, lab in enumerate(LABELS):
        P(f"    {lab:<34} {fmt(adj_p[0, k]):>8}  95% CI {fmt_ci(adj_ci[:, k])}")

    # ---------- STEP 5
    P("== STEP 5: defuse attempts (bomb_begin_defuse) in clean 5v5 planted rounds (set A)")
    bd = (chan("bomb_defuse").filter(pl.col("event_type") == "bomb_begin_defuse")
          .select(KEYS + ["tick", "player_id_fixed", "has_kit"]).collect())
    att = (bd.join(A5.select(KEYS + ["plant_tick", "end_tick", "bp_x", "bp_y", "map_name"]), on=KEYS, how="inner")
           .filter((pl.col("tick") > pl.col("plant_tick")) & (pl.col("tick") < pl.col("end_tick"))))
    att = att.with_columns(pl.Series("att_id", np.arange(att.height, dtype=np.int64)))
    P(f"  attempts (begin_defuse after plant, before round end): {att.height} in {att.select(KEYS).unique().height} rounds; "
      f"all begin_defuse rows in dataset: {bd.height}")
    fA = (fires.join(A5.select(KEYS + ["bp_x", "bp_y"]), on=KEYS, how="inner")
          .with_columns(((pl.col("fx") - pl.col("bp_x")) ** 2 + (pl.col("fy") - pl.col("bp_y")) ** 2).sqrt().alias("dist")))
    af = (att.select(["att_id"] + KEYS + ["tick"])
          .join(fA.select(KEYS + ["start_tick", "fire_end", "dist"]), on=KEYS, how="inner")
          .filter((pl.col("start_tick") <= pl.col("tick")) & (pl.col("tick") <= pl.col("fire_end"))))
    for R in R_LIST:
        flags = (af.filter(pl.col("dist") <= R).select("att_id").unique()
                 .with_columns(pl.lit(True).alias(f"fire{int(R)}")))
        att = att.join(flags, on="att_id", how="left").with_columns(pl.col(f"fire{int(R)}").fill_null(False))

    bdef = (chan("bomb_state").filter(pl.col("event_type") == "bomb_defused")
            .select(KEYS + [pl.col("tick").alias("def_tick"), "player_id_fixed"]).collect())
    att = att.sort(KEYS + ["player_id_fixed", "tick"]).with_columns(
        pl.col("tick").shift(-1).over(KEYS + ["player_id_fixed"]).alias("next_tick"))
    succ = (att.select(["att_id"] + KEYS + ["player_id_fixed", "tick", "next_tick"])
            .join(bdef, on=KEYS + ["player_id_fixed"], how="inner")
            .filter((pl.col("def_tick") >= pl.col("tick")) & (pl.col("def_tick") <= pl.col("tick") + DEF_WIN)
                    & (pl.col("next_tick").is_null() | (pl.col("def_tick") < pl.col("next_tick"))))
            .select("att_id").unique().with_columns(pl.lit(True).alias("success")))
    att = att.join(succ, on="att_id", how="left").with_columns(pl.col("success").fill_null(False))

    ph_all = chan("player_hurt").select(KEYS + ["tick", "player_id_fixed", "weapon_name"]).collect()
    wn = ph_all.group_by("weapon_name").agg(pl.len().alias("n")).sort("n", descending=True)
    fire_like = [(r["weapon_name"], r["n"]) for r in wn.iter_rows(named=True)
                 if r["weapon_name"] and any(w in r["weapon_name"].lower() for w in FIRE_WORDS)]
    P(f"  player_hurt weapon_name fire-like values: {fire_like}")
    ph_f = (ph_all.filter(pl.col("weapon_name").is_in(["inferno", "molotov"]))
            .select(KEYS + ["player_id_fixed", pl.col("tick").alias("hurt_tick")]))
    hurt = (att.select(["att_id"] + KEYS + ["player_id_fixed", "tick"])
            .join(ph_f, on=KEYS + ["player_id_fixed"], how="inner")
            .filter((pl.col("hurt_tick") >= pl.col("tick")) & (pl.col("hurt_tick") <= pl.col("tick") + HURT_WIN))
            .select("att_id").unique().with_columns(pl.lit(True).alias("hurt5")))
    att = att.join(hurt, on="att_id", how="left").with_columns(pl.col("hurt5").fill_null(False))
    P(f"  successful attempts (defuse completes, same player, before next attempt, <=12 s): {att.filter(pl.col('success')).height}")

    amid = np.array([midx_map[m] for m in att["match_key"].to_list()], dtype=np.int64)
    succ_a = att["success"].cast(pl.Int64).to_numpy().astype(np.float64)
    hurt_a = att["hurt5"].cast(pl.Int64).to_numpy().astype(np.float64)
    Wa = boot_matrix(M, SEED)
    for R in R_LIST:
        fl = att[f"fire{int(R)}"].cast(pl.Int64).to_numpy().astype(np.int64)
        Na = np.zeros((2, M)); np.add.at(Na, (fl, amid), 1.0)
        Da = np.zeros((2, M, 2))
        np.add.at(Da, (fl, amid, 0), succ_a)
        np.add.at(Da, (fl, amid, 1), hurt_a)
        n_w = int(fl.sum())
        n_wo = int(len(fl) - n_w)
        r1p, r0p, dp = crude(np.ones((1, M)), Na, Da)
        _, _, db = crude(Wa, Na, Da)
        dcs = ci(db)
        P(f"  R={int(R)}: attempts with fire burning within R: {n_w} ({sh(n_w, len(fl))} of attempts) | no fire: {n_wo}")
        P(f"     defuse-completed share: with-fire {fmt(r1p[0, 0])} vs no-fire {fmt(r0p[0, 0])}, diff {fmt(dp[0, 0])} 95% CI {fmt_ci(dcs[:, 0])}")
        P(f"     defuser hurt by inferno/molotov within 5 s: with-fire {fmt(r1p[0, 1])} vs no-fire {fmt(r0p[0, 1])}, diff {fmt(dp[0, 1])} 95% CI {fmt_ci(dcs[:, 1])}")
        if R == 150.0:
            P(f"     R=150 with-fire: success {att.filter(pl.col('fire150') & pl.col('success')).height}/{att.filter(pl.col('fire150')).height}; "
              f"hurt5 {att.filter(pl.col('fire150') & pl.col('hurt5')).height}/{att.filter(pl.col('fire150')).height}")
            P(f"     R=150 no-fire: success {att.filter(~pl.col('fire150') & pl.col('success')).height}/{att.filter(~pl.col('fire150')).height}; "
              f"hurt5 {att.filter(~pl.col('fire150') & pl.col('hurt5')).height}/{att.filter(~pl.col('fire150')).height}")

    # ---------- summary numbers
    P("== SUMMARY")
    P(f"  planted (clean 5v5, live, proxy) n={A5.height}; T-thrown fire within 150: {A5.filter(pl.col('hasT150')).height}")
    P(f"  fires after plant in clean planted rounds: {fa5.height}; within 150: {fa5.filter(pl.col('dist') <= 150.0).height}")
    P(f"  defuse attempts n={att.height}, with-fire(150) n={att.filter(pl.col('fire150')).height}")


if __name__ == "__main__":
    main()
