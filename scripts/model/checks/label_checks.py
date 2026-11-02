"""Label checks (C): the final round winner, the four vote signals and their reconciliation with the header score. No era, build or
map name is hard-coded: every rule is applied to every era|build found in the data."""
from checks.core import fail, notrun, ok, verdict, warn


def after_swap_expr(pl, rnd, mr):
    """First round of a half: round mr/2+1, and in overtime every 3rd round starting at mr+1 (independent copy of the rule)."""
    return (rnd == mr // 2 + 1) | ((rnd > mr) & (((rnd - mr - 1) % 3) == 0))


def swap_after_expr(pl, rnd, mr):
    """Sides swap after completed round `rnd`: halftime, and after the 3rd round of each overtime period."""
    return (rnd == mr // 2) | ((rnd > mr) & (((rnd - mr - 3) % 6) == 0))


def _key(pl):
    return (pl.col("era").cast(pl.String) + "|" + pl.col("build_num").cast(pl.String)).alias("key")


def vote_rule(pl, T, CT, bomb, rec_v, score, elim):
    """The documented final-label rule as an expression: the bomb outcome if present; else the majority of {rec (not suspect), score,
    elim} with at least two agreeing; else a single recorded signal; else unresolved (null)."""
    vt = sum([(c == T).cast(pl.Int8).fill_null(0) for c in (rec_v, score, elim)])
    vc = sum([(c == CT).cast(pl.Int8).fill_null(0) for c in (rec_v, score, elim)])
    return (pl.when(bomb.is_not_null()).then(bomb)
            .when((vt >= 2) & (vt > vc)).then(pl.lit(T))
            .when((vc >= 2) & (vc > vt)).then(pl.lit(CT))
            .when((vt == 1) & (vc == 0) & (rec_v == T)).then(pl.lit(T))
            .when((vc == 1) & (vt == 0) & (rec_v == CT)).then(pl.lit(CT))
            .otherwise(None))


def label_checks(ctx):
    R, P = ctx.R, ctx.P
    pl = ctx.pl
    if not ctx.want("C"):
        return
    T, CT = P.side_t, P.side_ct
    Rn, U = ctx.rounds, ctx.used

    def nviol(df, pred):
        return df.filter(pred.fill_null(True)).height

    def c1():
        ls = dict(U["label_source"].value_counts().iter_rows())
        return verdict(nviol(U, ~pl.col("final").is_in([T, CT])) == 0, f"{U.height} used rounds; label sources {ls}", n=U.height)
    R.run("C1", "final label in {2,3} for every used round (a null is a violation)", c1, min_n=P.min_used_rounds)

    def c2():
        by = U.group_by("era").agg(pl.len().alias("n"), (pl.col("label_source") == "recorded_only").sum().alias("ro"),
                                    ((pl.col("label_source") == "recorded_only") & pl.col("rec_suspect")).sum().alias("ro_susp")).sort("era")
        bad = [f"{r['era']}: {r['ro']}/{r['n']} ({r['ro'] / max(r['n'], 1):.2%})" for r in by.iter_rows(named=True)
               if r["ro"] / max(r["n"], 1) >= P.recorded_only_max_share or r["ro_susp"] > 0]
        return verdict(not bad, f"recorded_only share per era: {[(r['era'], r['ro'], r['n']) for r in by.iter_rows(named=True)]}; bound {P.recorded_only_max_share:.1%}"
                       + (f"; violations {bad}" if bad else ""), n=U.height)
    R.run("C2", "'recorded_only' labels are a small minority per era and never on suspect rounds", c2, min_n=P.min_used_rounds)

    def c3():
        X = U.with_columns(_key(pl), pl.col("final").ne_missing(pl.col("rec")).alias("diff"),
                           after_swap_expr(pl, pl.col("round"), pl.col("max_rounds_final")).alias("swap"))
        tab = X.group_by("key").agg(pl.len().alias("n"), pl.col("diff").sum().alias("n_diff"), (pl.col("diff") & ~pl.col("swap").fill_null(False)).sum().alias("n_out"),
                                    pl.col("swap").sum().alias("n_swap")).sort("key")
        out = int(tab["n_out"].sum())
        rows = []
        for r in tab.iter_rows(named=True):
            share = r["n_diff"] / max(r["n"], 1)
            ctx.record("relabel", r["key"], {"used_rounds": r["n"], "relabelled": r["n_diff"], "share": round(share, 5)})
            rows.append((r["key"], r["n_diff"], r["n"], share))
        msg = "relabelled/used by era|build: " + "; ".join(f"{k} {a}/{b} ({s:.2%})" for k, a, b, s in rows) + f"; relabelled outside a first-of-half round: {out}"
        worst = max((s for *_, s in rows), default=0.0)
        base_fail = []
        for k, a, b, s in rows:
            base = ctx.baseline_get("relabel", k)
            if base and s > base.get("share", 0.0) + P.completeness_baseline_tol:
                base_fail.append(f"{k} {s:.2%} worse than baseline {base['share']:.2%}")
        if out > P.relabel_outside_after_swap_max or worst > P.relabel_hard_share or base_fail:
            return fail(msg + (f"; {base_fail}" if base_fail else ""), n=U.height)
        if worst > P.relabel_warn_share:
            return warn(msg + " (known limitation: the recorded winner is inverted after a side swap in some builds; the vote repairs it)", n=U.height)
        return ok(msg, n=U.height)
    R.run("C3", "every relabelled round (final != recorded winner) is a first round of a half; relabel share bounded, per era|build", c3, min_n=P.min_used_rounds)

    def c4():
        exp = vote_rule(pl, T, CT, pl.col("bomb"), pl.col("rec_v"), pl.col("score"), pl.col("elim"))
        bad = Rn.filter(exp.ne_missing(pl.col("final"))).height
        return verdict(bad == 0, f"{Rn.height:,} rounds; final differs from the vote rule applied to the stored signals on {bad:,}", n=Rn.height)
    R.run("C4", "final = the documented vote rule (bomb, else majority of rec/score/elim) applied to the stored signals", c4, min_n=1)

    def c5():
        exp = pl.when(pl.col("defuse_tick").is_not_null()).then(pl.lit(CT)).when(pl.col("explode_tick").is_not_null()).then(pl.lit(T)).otherwise(None)
        bad = Rn.filter(exp.ne_missing(pl.col("bomb"))).height
        n_b = int(Rn["bomb"].is_not_null().sum())
        return verdict(bad == 0, f"{n_b:,} rounds with a bomb outcome; bomb signal differs from defuse/explode ticks on {bad}", n=n_b)
    R.run("C5", "bomb signal = CT if a defuse tick exists, T if an explosion tick exists", c5, min_n=1)

    def c6():
        if ctx.missing("deaths.parquet"):
            return notrun("deaths.parquet missing")
        d = pl.read_parquet(ctx.p("deaths.parquet")).filter(pl.col("side_ok"))
        k = d.with_columns(pl.col("tick").rank("ordinal").over(["match_key", "round", "side"]).alias("k")).filter(pl.col("k") == P.players_per_side)
        ct_w = k.filter(pl.col("side") == CT).select("match_key", "round", pl.col("tick").alias("ct_w"))
        t_w = k.filter(pl.col("side") == T).select("match_key", "round", pl.col("tick").alias("t_w"))
        x = (Rn.select("match_key", "round", "plant_tick", "elim", "ct_wipe_tick", "t_wipe_tick").join(ct_w, on=["match_key", "round"], how="left")
             .join(t_w, on=["match_key", "round"], how="left"))
        tdec = pl.when(pl.col("plant_tick").is_null() | (pl.col("plant_tick") > pl.col("t_w"))).then(pl.col("t_w")).otherwise(None)
        x = x.with_columns(pl.min_horizontal(pl.col("ct_w"), tdec).alias("wipe"))
        exp = pl.when(pl.col("wipe").is_null()).then(None).when(pl.col("ct_w") == pl.col("wipe")).then(pl.lit(T)).otherwise(pl.lit(CT))
        n_wipe = int(x["wipe"].is_not_null().sum())
        if n_wipe == 0:
            return notrun("no round with a wiped side in this store")
        bad_tick = x.filter(pl.col("ct_w").ne_missing(pl.col("ct_wipe_tick")) | pl.col("t_w").ne_missing(pl.col("t_wipe_tick"))).height
        bad_elim = x.filter(exp.ne_missing(pl.col("elim"))).height
        return verdict(bad_tick + bad_elim == 0, f"{n_wipe:,} rounds with a wiped side; wipe ticks differ from the 5th death in deaths.parquet on {bad_tick}; "
                                                  f"elimination vote differs on {bad_elim}", n=n_wipe)
    R.run("C6", "elimination vote re-derived from deaths.parquet (5th death per side; a T wipe after the plant does not decide)", c6, min_n=1)

    def c7():
        nxt = Rn.select("match_key", (pl.col("round") - 1).alias("round"), pl.col("t_sc").alias("t2"), pl.col("ct_sc").alias("c2"))
        x = Rn.select("match_key", "round", "t_sc", "ct_sc", "score", "swap_boundary", "max_rounds_final").join(nxt, on=["match_key", "round"], how="left")
        sb = swap_after_expr(pl, pl.col("round"), pl.col("max_rounds_final"))
        dt, dc = pl.col("t2") - pl.col("t_sc"), pl.col("c2") - pl.col("ct_sc")
        exp = (pl.when(sb | pl.col("t2").is_null() | pl.col("t_sc").is_null()).then(None)
               .when((dt == 1) & (dc == 0)).then(pl.lit(T)).when((dc == 1) & (dt == 0)).then(pl.lit(CT)).otherwise(None))
        n_s = int(x["score"].is_not_null().sum())
        bad = x.filter(exp.ne_missing(pl.col("score"))).height
        bad_sb = x.filter(sb.ne_missing(pl.col("swap_boundary"))).height
        return verdict(bad + bad_sb == 0, f"{n_s:,} rounds with a score signal; differs from the freeze-end score step on {bad}; swap-boundary flag differs on {bad_sb}", n=n_s)
    R.run("C7", "score signal = which side's freeze-end score rose, undefined on swap boundaries", c7, min_n=1)

    def c8():
        X = Rn.filter(pl.col("rec_suspect").fill_null(False))
        bad = X.filter(~after_swap_expr(pl, pl.col("round"), pl.col("max_rounds_final")).fill_null(False)).height
        return verdict(bad == 0, f"{X.height:,} suspect rounds, {bad} not a first round of a half", n=max(X.height, 1))
    R.run("C8", "recorded winners are only treated as suspect on first rounds of a half", c8, min_n=1)

    def c9():
        M = ctx.matches
        cnt = Rn.group_by("match_key").agg(pl.len().alias("n_rd"), pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"))
        X = (M.select("match_key", "era", "build_num", "t_starters_score_final", "ct_starters_score_final", "max_rounds_final").join(cnt, on="match_key", how="inner")
             .with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("tot")))
        comp = X.filter(pl.col("tot").is_not_null() & (pl.col("n_rd") == pl.col("tot")) & (pl.col("rmin") == 1) & (pl.col("rmax") == pl.col("tot")))
        rr = (Rn.select("match_key", "round", "final", "rec").join(comp.select("match_key", pl.col("max_rounds_final").alias("mr"), "t_starters_score_final",
                                                                                "ct_starters_score_final"), on="match_key", how="inner").sort(["match_key", "round"]))
        sw = swap_after_expr(pl, pl.col("round"), pl.col("mr")).cast(pl.Int32)
        rr = rr.with_columns((sw.cum_sum().over("match_key") - sw).alias("nsw"))

        def per_match(col):
            # the team that started on side T is on side T while an even number of swaps has happened
            won_by_t_starters = (pl.when(pl.col(col).is_null()).then(None)
                                 .when(((pl.col("nsw") % 2 == 0) & (pl.col(col) == T)) | ((pl.col("nsw") % 2 == 1) & (pl.col(col) == CT))).then(1).otherwise(0))
            g = rr.group_by("match_key").agg(won_by_t_starters.sum().alias("a"), pl.len().alias("n"), won_by_t_starters.null_count().alias("nn"),
                                             pl.col("t_starters_score_final").first().alias("tf"), pl.col("ct_starters_score_final").first().alias("cf"))
            # orientation-free: the header's "t_starters" / "ct_starters" naming is not consistent with who actually started on T
            same = ((pl.col("a") == pl.col("tf")) & (pl.col("n") - pl.col("a") == pl.col("cf"))) | ((pl.col("a") == pl.col("cf")) & (pl.col("n") - pl.col("a") == pl.col("tf")))
            return g.with_columns(same.alias("ok"))
        gf, gr = per_match("final"), per_match("rec")
        keyed = comp.select("match_key", _key(pl))
        gf = gf.join(keyed, on="match_key")
        gr = gr.join(keyed, on="match_key")
        used_by_key = U.select(_key(pl)).group_by("key").len().rename({"len": "used_rounds"})
        t_final = gf.filter(pl.col("nn") == 0).group_by("key").agg(pl.len().alias("n_checked"), pl.col("ok").sum().alias("n_ok")).sort("key")
        t_unres = gf.filter(pl.col("nn") > 0).group_by("key").len().rename({"len": "n_unresolvable"})
        t_rec = gr.filter(pl.col("nn") == 0).group_by("key").agg(pl.len().alias("rn"), pl.col("ok").sum().alias("rok"))
        allkeys = X.select(_key(pl)).unique().sort("key")
        tab = (allkeys.join(t_final, on="key", how="left").join(t_unres, on="key", how="left").join(t_rec, on="key", how="left")
               .join(used_by_key, on="key", how="left").fill_null(0))
        fails, notes, lines = [], [], []
        for r in tab.iter_rows(named=True):
            k, nc, nk = r["key"], r["n_checked"], r["n_ok"]
            lines.append(f"{k}: final {nk}/{nc}" + (f" (+{r['n_unresolvable']} with an unresolved round)" if r["n_unresolvable"] else "")
                         + (f", recorded winners {r['rok']}/{r['rn']}" if r["rn"] else ""))
            ctx.record("reconcile", k, {"checked": nc, "ok": nk})
            if nc >= P.reconcile_min_matches:
                if nk / nc < P.reconcile_min_share:
                    fails.append(f"{k} {nk}/{nc} reconcile")
            elif r["used_rounds"] > 0:
                notes.append(f"{k}: {r['used_rounds']:,} used rounds in a build with only {nc} checkable complete matches")
        total = int(tab["n_checked"].sum())
        msg = f"{total} complete matches reconciled; " + "; ".join(lines)
        if total == 0:
            return fail("no complete match to reconcile against the header score: " + msg, n=0)
        if fails:
            return fail(msg + "; BELOW " + f"{P.reconcile_min_share:.1%}: {fails}", n=total)
        if notes:
            return warn(msg + "; not checkable (known limitation): " + "; ".join(notes), n=total)
        return ok(msg, n=total)
    R.run("C9", "final labels reconcile with the header score pair on every complete match, per era|build (orientation-free)", c9, min_n=1)

    def c10():
        X = Rn.filter(pl.col("round_status") == "ok").with_columns(_key(pl))
        fails, warns, lines, pooled = [], [], [], 0
        for sig in ("rec_v", "score", "elim"):
            s = X.filter(pl.col("bomb").is_not_null() & pl.col(sig).is_not_null())
            tab = s.group_by("key").agg(pl.len().alias("n"), (pl.col(sig) == pl.col("bomb")).mean().alias("agree")).sort("key")
            n_all = int(tab["n"].sum()) if tab.height else 0
            pooled += n_all
            cells = []
            for r in tab.iter_rows(named=True):
                if r["n"] < P.signal_min_n:
                    continue
                cells.append(f"{r['key']} {r['agree']:.1%} (n={r['n']})")
                ctx.record("signal_agreement", f"{sig}|{r['key']}", {"n": r["n"], "agree": round(r["agree"], 4)})
                if r["agree"] < P.signal_fail_below:
                    fails.append(f"{sig} {r['key']} {r['agree']:.1%}")
                elif r["agree"] < P.signal_warn_below:
                    warns.append(f"{sig} {r['key']} {r['agree']:.1%}")
            lines.append(f"{sig}: " + (", ".join(cells) if cells else f"no cell with n >= {P.signal_min_n} (pooled n={n_all})"))
        if pooled < P.signal_min_n:
            return notrun(f"only {pooled} bomb-decided rounds carry a vote signal (need {P.signal_min_n}); " + "; ".join(lines))
        msg = "agreement of each vote signal with the bomb outcome on bomb-decided rounds: " + "; ".join(lines)
        if fails:
            return fail(msg + f"; BELOW {P.signal_fail_below:.0%}: {fails}", n=pooled)
        if warns:
            return warn(msg + f"; below {P.signal_warn_below:.0%} (known data defect): {warns}", n=pooled)
        return ok(msg, n=pooled)
    R.run("C10", "each vote signal (rec not suspect, score, elim) agrees with the bomb outcome on bomb-decided rounds", c10, min_n=1)
