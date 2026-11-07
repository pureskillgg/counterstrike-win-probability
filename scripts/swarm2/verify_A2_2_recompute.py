# verify_A2_2_recompute.py
# Independent verifier for finding A2-2 (2024 round_start-family label offset).
# Does not read any A2_*.py output. Uses only tome parquet via polars lazy scans, one channel at a time.
# Tests:
#   A  label-agnostic position: for each round_start row, label of the latest round_end at or before its tick
#   B  label of the latest round_freeze_end at or before it, and gap in ticks
#   C  label of the earliest round_freeze_end at or after it, and gap in ticks (the round it precedes)
#   D  round_end rows: label of the latest round_start before them (d = k - s)
#   E  round_end labels with no preceding round_start (pre-start), and whether freeze_end(k) precedes them
#   F  simulation of the bake-off rst_keys filter: current key (label k) vs claimed fix (label k+1)
#   G  score test recomputed: freeze_end(k) -> freeze_end(k+1) score delta vs round_end label k and k+1
#   H  same-tick boundary events (round_officially_ended, round_prestart, round_poststart) vs round_start label
# Every threshold or window is an ASSUMPTION stated in the output.
import os, json
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
TOMES = {"2024": os.path.join(ROOT, "tomes", "2024-02-g", "tome", "csds"),
         "2026": os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")}
BUNDLE = ["round_officially_ended", "round_prestart", "round_poststart"]


def read(tome, base, cols, filt=None):
    names = [x for x in os.listdir(tome) if x.split(".", 1)[0] == base]
    assert len(names) == 1, (base, names)
    dp = os.path.join(tome, names[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = []
    for p in meta["pages"]:
        lf = pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols)
        if filt is not None:
            lf = lf.filter(filt)
        parts.append(lf.collect())
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]


def add_day(df):
    return df.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))


def group_map(days, tag):
    if tag == "2024":
        f = lambda d: "day01" if d == "01" else ("day02-07" if "02" <= d <= "07" else "day12-25")
    else:
        f = lambda d: "day" + d
    return pl.DataFrame({"day": days, "grp": [f(d) for d in days]}, schema={"day": pl.Utf8, "grp": pl.Utf8})


def bucket_expr():
    return (pl.when(pl.col("round") <= 12).then(pl.lit("r01-12"))
            .when(pl.col("round") <= 24).then(pl.lit("r13-24"))
            .otherwise(pl.lit("r25+")))


def quantiles(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return "none"
    def q(p):
        return v[min(len(v) - 1, int(p * (len(v) - 1)))]
    return f"n={len(v)} min={v[0]} p05={q(0.05)} median={q(0.5)} p95={q(0.95)} max={v[-1]}"


def load(tag):
    tome = TOMES[tag]
    H = read(tome, "header", ["key"])
    RE = read(tome, "basic_round_end", ["match_key", "round", "tick", "winner_team_code"])
    RS = read(tome, "basic_round_start", ["match_key", "round", "tick"])
    ST = read(tome, "basic_round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"],
              filt=pl.col("event_type").is_in(["round_freeze_end"] + BUNDLE))
    ints = lambda cols: [pl.col(c).cast(pl.Int64, strict=False) for c in cols]
    RE = RE.with_columns(ints(["round", "tick", "winner_team_code"]))
    RS = RS.with_columns(ints(["round", "tick"]))
    ST = ST.with_columns(ints(["round", "tick", "t_score", "ct_score"]))
    FE = ST.filter(pl.col("event_type") == "round_freeze_end").sort("tick").unique(
        ["match_key", "round"], keep="first", maintain_order=True)
    RE = add_day(RE)
    RS = add_day(RS)
    days = sorted(set(RE["day"].drop_nulls().to_list()) | set(RS["day"].drop_nulls().to_list()))
    gm = group_map(days, tag)
    return H, RE, RS, FE, ST, gm


def section_basic(tag, H, RE, RS, FE):
    print(f"\n== [{tag}] basic counts ==")
    print(f"header matches: {H.height}; round_end rows: {RE.height}; unique (match,label) round_end: "
          f"{RE.unique(['match_key', 'round']).height}; matches with round_end: {RE['match_key'].n_unique()}")
    print(f"round_start rows: {RS.height}; matches with round_start: {RS['match_key'].n_unique()}; "
          f"header matches with no round_start: {H.filter(~pl.col('key').is_in(RS['match_key'].unique().to_list())).height}")
    print(f"round_freeze_end unique (match,label): {FE.height}")


def section_position(tag, RE, RS, FE, gm):
    print(f"\n== [{tag}] position tests A-C (tick order; labels only from round_end / round_freeze_end) ==")
    rs_s = RS.select(["match_key", "round", "tick", "day"]).sort("tick")
    re_s = RE.select(["match_key", pl.col("tick").alias("te"), pl.col("round").alias("j")]).sort("te")
    fe_s = FE.select(["match_key", pl.col("tick").alias("ft"), pl.col("round").alias("g")]).sort("ft")
    A = rs_s.join_asof(re_s, left_on="tick", right_on="te", by="match_key", strategy="backward")
    A = A.with_columns((pl.col("round") - pl.col("j")).alias("dA"))
    print("A: round_start label minus label of latest round_end before it (dA=0 means round_start label equals the round that just ended):")
    print("  overall:", A.group_by("dA").len().sort("dA").rows())
    print("  by group:", A.join(gm, on="day").group_by(["grp", "dA"]).len().sort(["grp", "dA"]).rows())
    B = rs_s.join_asof(fe_s, left_on="tick", right_on="ft", by="match_key", strategy="backward")
    B = B.with_columns((pl.col("round") - pl.col("g")).alias("dB"), (pl.col("tick") - pl.col("ft")).alias("gapB"))
    print("B: round_start label minus label of latest round_freeze_end before it:")
    print("  overall:", B.group_by("dB").len().sort("dB").rows())
    print("  gap ticks since latest freeze_end:", quantiles(B["gapB"].to_list()))
    C = rs_s.join_asof(fe_s, left_on="tick", right_on="ft", by="match_key", strategy="forward")
    C = C.with_columns((pl.col("round") - pl.col("g")).alias("dC"), (pl.col("ft") - pl.col("tick")).alias("gapC"))
    print("C: round_start label minus label of earliest round_freeze_end after it (dC=0 means the same round starts):")
    print("  overall:", C.group_by("dC").len().sort("dC").rows())
    print("  gap ticks to next freeze_end:", quantiles(C["gapC"].to_list()))
    print("  by group gap to next freeze_end:")
    for g in sorted(set(gm["grp"].to_list())):
        sub = C.join(gm, on="day").filter(pl.col("grp") == g)
        print(f"    {g}: dC dist {sub.group_by('dC').len().sort('dC').rows()}  gap {quantiles(sub['gapC'].to_list())}")
    return A


def section_round_end(tag, RE, RS, gm):
    print(f"\n== [{tag}] round_end rows: label of latest round_start before them (E) ==")
    rs_s = RS.select(["match_key", pl.col("tick").alias("ts"), pl.col("round").alias("s")]).sort("ts")
    re_s = RE.sort("tick")
    E = re_s.join_asof(rs_s, left_on="tick", right_on="ts", by="match_key", strategy="backward")
    E = E.with_columns((pl.col("round") - pl.col("s")).alias("d"))
    print("  d = round_end label minus latest round_start label:", E.group_by("d", maintain_order=False).len().sort("d").rows())
    pre = E.filter(pl.col("s").is_null())
    print(f"  pre-start round_end rows (no round_start before): {pre.height}")
    print("  pre-start by label:", pre.group_by("round").len().sort("round").rows()[:8], "...")
    print("  pre-start by group:", pre.join(gm, on="day").group_by(["grp", "round"]).len().sort(["grp", "round"]).rows()[:12])
    # check pre-start round_end rows are real round ends: freeze_end of same label precedes them
    chk = pre.join(FE_GLOBAL.select(["match_key", "round", pl.col("tick").alias("ft")]), on=["match_key", "round"], how="left")
    chk = chk.with_columns((pl.col("ft") < pl.col("tick")).alias("fe_before_end"))
    print("  pre-start rows with a freeze_end of same label before them:",
          chk.filter(pl.col("fe_before_end") == True).height, "of", chk.height,
          "; with no freeze_end of that label:", chk.filter(pl.col("ft").is_null()).height)
    dup = RE.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
    print(f"  duplicate round_end (match,label) keys: {dup.height}")
    return E


def section_label0(tag, RE, RS, H, gm):
    print(f"\n== [{tag}] presence of round_start label 0 (start bundle of round 1) ==")
    m = RS.group_by("match_key").agg(pl.col("round").min().alias("rs_min"), pl.len().alias("n_rs"), pl.col("day").first())
    print(f"  matches with round_start: {m.height}; with min label 0: {m.filter(pl.col('rs_min') == 0).height}; "
          f"with min label 1: {m.filter(pl.col('rs_min') == 1).height}; other: "
          f"{m.filter(~pl.col('rs_min').is_in([0, 1])).height}")
    print("  by group (rs_min):", m.join(gm, on="day").group_by(["grp", "rs_min"]).len().sort(["grp", "rs_min"]).rows())
    maxlab = RS.group_by("match_key").agg(pl.col("round").max().alias("rs_max"))
    re_max = RE.group_by("match_key").agg(pl.col("round").max().alias("re_max"))
    cmp = re_max.join(maxlab, on="match_key", how="left")
    eq = cmp.filter(pl.col("rs_max") == pl.col("re_max")).height
    eq1 = cmp.filter(pl.col("rs_max") == pl.col("re_max") - 1).height
    print("  match-level: rs_max equals re_max:", eq, " rs_max == re_max - 1:", eq1,
          " other or no round_start:", cmp.height - eq - eq1)
    no_rs = H.filter(~pl.col("key").is_in(RS["match_key"].unique().to_list()))
    print(f"  header matches with no round_start rows: {no_rs.height}")


def section_filter_sim(tag, RE, RS, FE, gm):
    print(f"\n== [{tag}] simulation of the bake-off rst_keys filter (only this filter; freeze_end live check kept) ==")
    RE_k = RE.select(["match_key", "round"]).unique()
    cur = RS.select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("cur"))
    fix = RS.select(["match_key", (pl.col("round") + 1).alias("round")]).unique().with_columns(pl.lit(True).alias("fix"))
    fe = FE.select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("fe"))
    R = RE_k.join(cur, on=["match_key", "round"], how="left").join(fix, on=["match_key", "round"], how="left") \
            .join(fe, on=["match_key", "round"], how="left")
    R = R.with_columns([pl.col("cur").fill_null(False), pl.col("fix").fill_null(False), pl.col("fe").fill_null(False)])
    maxk = RE_k.group_by("match_key").agg(pl.col("round").max().alias("maxk"))
    R = R.join(maxk, on="match_key").with_columns(
        (pl.col("round") == 1).alias("is_r1"), (pl.col("round") == pl.col("maxk")).alias("is_last"))
    print(f"  round_end rounds: {R.height}")
    print("  current key (label k) present:", R.filter(pl.col("cur")).height,
          " missing:", R.filter(~pl.col("cur")).height,
          " of which round 1:", R.filter((~pl.col("cur")) & pl.col("is_r1")).height,
          " last round:", R.filter((~pl.col("cur")) & pl.col("is_last")).height)
    print("  claimed fix key (label k+1) present:", R.filter(pl.col("fix")).height,
          " missing:", R.filter(~pl.col("fix")).height,
          " of which round 1:", R.filter((~pl.col("fix")) & pl.col("is_r1")).height,
          " last round:", R.filter((~pl.col("fix")) & pl.col("is_last")).height)
    print("  rounds with a freeze_end (live start) row:", R.filter(pl.col("fe")).height, "of", R.height)
    print("  survivors of rst filter AND live check, current:", R.filter(pl.col("cur") & pl.col("fe")).height,
          " claimed fix:", R.filter(pl.col("fix") & pl.col("fe")).height,
          " live check only:", R.filter(pl.col("fe")).height)
    dd = add_day(R.select(["match_key", "round", "cur", "fix", "fe", "is_r1", "is_last"]))
    dd = dd.join(gm, on="day")
    print("  by group, claimed-fix drops of round 1:", dd.filter((~pl.col("fix")) & pl.col("is_r1")).group_by("grp").len().sort("grp").rows())
    print("  by group, current drops of last round:", dd.filter((~pl.col("cur")) & pl.col("is_last")).group_by("grp").len().sort("grp").rows())
    print("  by group, claimed-fix drops of last round:", dd.filter((~pl.col("fix")) & pl.col("is_last")).group_by("grp").len().sort("grp").rows())
    return R


def section_score(tag, FE, RE, gm):
    print(f"\n== [{tag}] score test (recomputed) ==")
    sumok = FE.filter((pl.col("t_score") + pl.col("ct_score")) == (pl.col("round") - 1)).height
    print(f"  freeze_end rows with t_score+ct_score == label-1: {sumok} of {FE.height}")
    fe = FE.select(["match_key", "round", "t_score", "ct_score"]).sort(["match_key", "round"])
    nxt = fe.with_columns([pl.col("round").shift(-1).over("match_key").alias("r2"),
                           pl.col("t_score").shift(-1).over("match_key").alias("t2"),
                           pl.col("ct_score").shift(-1).over("match_key").alias("c2")])
    nxt = nxt.filter(pl.col("r2") == pl.col("round") + 1)
    nxt = nxt.with_columns([(pl.col("t2") - pl.col("t_score")).alias("dT"),
                            (pl.col("c2") - pl.col("ct_score")).alias("dC")])
    nxt = nxt.with_columns(pl.when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(2)
                           .when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(3)
                           .otherwise(0).alias("ws"))
    print("  consecutive freeze_end pairs:", nxt.height, "; (dT,dC) combos:", nxt.group_by(["dT", "dC"]).len().sort(["dT", "dC"]).rows()[:10])
    rk = RE.select(["match_key", pl.col("round").alias("k"), pl.col("winner_team_code").alias("wk")])
    rk1 = RE.select(["match_key", pl.col("round").alias("k1"), pl.col("winner_team_code").alias("wk1")])
    nxt = nxt.join(rk, left_on=["match_key", "round"], right_on=["match_key", "k"], how="left")
    nxt = nxt.join(rk1, left_on=["match_key", "r2"], right_on=["match_key", "k1"], how="left")
    ok = nxt.filter(pl.col("ws") != 0)
    print(f"  pairs with score-derived winner: {ok.height}; agree with round_end label k: "
          f"{ok.filter(pl.col('ws') == pl.col('wk')).height} "
          f"({100 * ok.filter(pl.col('ws') == pl.col('wk')).height / max(ok.height, 1):.2f}%); "
          f"agree with label k+1: {ok.filter(pl.col('ws') == pl.col('wk1')).height} "
          f"({100 * ok.filter(pl.col('ws') == pl.col('wk1')).height / max(ok.height, 1):.2f}%)")
    ok = add_day(ok).join(gm, on="day").with_columns(bucket_expr().alias("bk"))
    tab = ok.group_by(["grp"]).agg(pl.len().alias("n"), (pl.col("ws") == pl.col("wk")).sum().alias("eq_k"),
                                   (pl.col("ws") == pl.col("wk1")).sum().alias("eq_k1")).sort("grp")
    print("  by group (n, eq_k, eq_k1):", tab.rows())
    tab2 = ok.group_by(["bk"]).agg(pl.len().alias("n"), (pl.col("ws") == pl.col("wk")).sum().alias("eq_k")).sort("bk")
    print("  by round bucket (n, eq_k):", tab2.rows())
    return ok


def section_bundle(tag, RS, ST):
    print(f"\n== [{tag}] same-tick boundary events vs round_start label (H) ==")
    for ev in BUNDLE:
        E = ST.filter(pl.col("event_type") == ev).select(["match_key", "tick", pl.col("round").alias("el")])
        E = E.unique(["match_key", "tick"], keep="first", maintain_order=True)
        J = RS.select(["match_key", "tick", "round"]).join(E, on=["match_key", "tick"], how="left")
        n_match = J.filter(pl.col("el").is_not_null()).height
        n_same = J.filter(pl.col("el") == pl.col("round")).height
        print(f"  {ev}: round_start rows with same-tick event {n_match} of {RS.height}; same label {n_same}")


if __name__ == "__main__":
    results = {}
    for tag in ["2024", "2026"]:
        H, RE, RS, FE, ST, gm = load(tag)
        FE_GLOBAL = FE
        section_basic(tag, H, RE, RS, FE)
        section_position(tag, RE, RS, FE, gm)
        section_round_end(tag, RE, RS, gm)
        section_label0(tag, RE, RS, H, gm)
        section_filter_sim(tag, RE, RS, FE, gm)
        section_score(tag, FE, RE, gm)
        section_bundle(tag, RS, ST)
    print("\nDONE")
