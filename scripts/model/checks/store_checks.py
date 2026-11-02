"""Store-level checks: input manifest, store consistency (S), scope (A), round completeness (K), duplicates (B)."""
import glob
import os
import time

from checks import core
from checks.core import fail, notrun, ok, verdict, warn


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _ts(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) if t else "n/a"


def _top(d, k=6):
    return ", ".join(f"{a}={b}" for a, b in list(d.items())[:k]) + (" ..." if len(d) > k else "")


# ------------------------------------------------------------------ input manifest
def build_manifest(ctx):
    """What this run looked at: every prepared table, the model files, the split manifest and the tome store (names, sizes, mtimes, key
    counts; sha256 of the small files). Written into the report so a PASS can be tied to exact inputs."""
    P = ctx.P
    files = {}
    for n in ("matches.parquet", "rounds.parquet", "deaths.parquet", "grid.parquet", "prep_report.json"):
        files[n] = core.file_info(ctx.p(n), P.manifest_sha_max_mb, rows=n.endswith(".parquet"))
    for label, path in (("model", ctx.model_path), ("model_meta", ctx.meta_path), ("test_sample", ctx.sample_path)):
        files[label] = core.file_info(path, P.manifest_sha_max_mb, rows=path.endswith(".parquet"))
    smp = ctx.split_manifest_path()
    if smp:
        files["split_manifest"] = core.file_info(smp, P.manifest_sha_max_mb)
    tomes = {}
    if os.path.isdir(ctx.tomes_dir):
        for d in sorted(os.listdir(ctx.tomes_dir)):
            sub = os.path.join(ctx.tomes_dir, d)
            if not os.path.isdir(sub):
                continue
            pages = glob.glob(os.path.join(sub, "dataframe_*"))
            keysets = sorted(glob.glob(os.path.join(sub, "keyset_*")))
            tomes[d] = {"pages": len(pages), "bytes": sum(os.path.getsize(x) for x in pages),
                        "newest_page": _ts(max((_mtime(x) for x in pages), default=0.0)),
                        "keysets": [core.file_info(k, P.manifest_sha_max_mb, rows=True) for k in keysets]}
    st = sorted(glob.glob(os.path.join(ctx.status_dir, "*.parquet")))
    status = {"files": len(st), "bytes": sum(os.path.getsize(x) for x in st), "newest": _ts(max((_mtime(x) for x in st), default=0.0)),
              "first": os.path.basename(st[0]) if st else None, "last": os.path.basename(st[-1]) if st else None}
    return {"root": ctx.root, "store": ctx.store, "files": files, "tomes": tomes, "status_1hz": status}


def check_manifest(ctx):
    man = build_manifest(ctx)
    ctx.manifest = man
    need = ["matches.parquet", "rounds.parquet", "grid.parquet", "deaths.parquet"]
    miss = [n for n in need if not man["files"][n]["exists"]]
    if miss:
        return fail(f"prepared tables missing: {miss}")
    rows = {n: man["files"][n].get("rows") for n in need}
    return ok(f"rows {rows}; tome folders {len(man['tomes'])}; status files {man['status_1hz']['files']}", n=min(r or 0 for r in rows.values()))


# ------------------------------------------------------------------ S: store consistency
def _keys_of(ctx, d):
    pl = ctx.pl
    ks = sorted(glob.glob(os.path.join(d, "keyset_*")))
    if ks:
        return set(pl.concat([pl.read_parquet(f).to_series(0).cast(pl.String) for f in ks]).to_list()), "keyset"
    pages = sorted(glob.glob(os.path.join(d, "dataframe_*")))
    if not pages:
        return None, "no pages"
    col = "match_key"
    lf = pl.scan_parquet(pages)
    if col not in lf.collect_schema().names():
        col = "key"
    return set(lf.select(pl.col(col).cast(pl.String)).unique().collect().to_series().to_list()), "dataframe"


def store_consistency(ctx):
    R, P = ctx.R, ctx.P
    if not ctx.want("S"):
        return
    hd = ctx.header_dir()
    if hd is None:
        R.add("S1", "tome store: every channel's match keys exist in the header", notrun(f"no header folder under {ctx.tomes_dir}"))
        R.add("S2", "tome round_end rounds equal the prepared rounds table", notrun("no tome store"))
        R.add("S4", "prepared tables are not older than the tome files they were built from", notrun("no tome store"))
    else:
        hk, how = _keys_of(ctx, hd)
        ctx.header_keys = hk
        chans = [c.strip() for c in P.channels.split(",") if c.strip()]
        ctx.channel_keys = {}

        def s1():
            lines, bad, lack_tot = [], 0, 0
            for ch in chans:
                d = ctx.channel_dir(ch)
                if d is None:
                    lines.append(f"{ch}: folder missing")
                    bad += 1
                    continue
                k, _ = _keys_of(ctx, d)
                ctx.channel_keys[ch] = k
                orphan, lacking = len(k - hk), len(hk - k)
                bad += orphan > 0
                lack_tot += lacking
                lines.append(f"{ch}: {len(k):,} keys, {orphan:,} not in header, {lacking:,} header keys without it")
            msg = f"header {len(hk):,} keys ({how}); " + "; ".join(lines)
            if bad:
                return fail(msg, n=len(hk))
            if lack_tot:
                return warn(msg + " (matches lacking a channel are marked channels_complete=False and not used)", n=len(hk))
            return ok(msg, n=len(hk))
        R.run("S1", "tome store: every channel's match keys exist in the header (no orphan matches)", s1)

        def s2():
            pl = ctx.pl
            d = ctx.channel_dir("round_end")
            if d is None:
                return notrun("no round_end channel")
            pages = sorted(glob.glob(os.path.join(d, "dataframe_*")))
            re_ = pl.scan_parquet(pages).select("match_key", "round").unique().collect()
            hdr = pl.DataFrame({"match_key": sorted(hk)})
            in_hdr = re_.join(hdr, on="match_key", how="semi")
            orphan_rounds = re_.height - in_hdr.height
            prepared = ctx.rounds.select("match_key", "round")
            only_tome = in_hdr.join(prepared, on=["match_key", "round"], how="anti").height
            only_prep = prepared.join(in_hdr, on=["match_key", "round"], how="anti").height
            msg = (f"tome rounds {re_.height:,} (of which {orphan_rounds:,} for matches absent from the header); prepared {prepared.height:,}; "
                   f"header-match tome rounds missing from prepared {only_tome:,}, prepared rounds missing from tome {only_prep:,}")
            return verdict(orphan_rounds == 0 and only_tome == 0 and only_prep == 0, msg, n=in_hdr.height)
        R.run("S2", "tome round_end rounds equal the prepared rounds table", s2)

        def s4():
            newest = []
            for name in ["header"] + [f"basic_{c}" for c in chans]:
                sub = hd if name == "header" else ctx.channel_dir(name[len("basic_"):])
                if sub is None:
                    continue
                for f in glob.glob(os.path.join(sub, "dataframe_*")) + glob.glob(os.path.join(sub, "keyset_*")):
                    newest.append((_mtime(f), f))
            if not newest:
                return notrun("no tome files")
            t_tome = max(newest)
            t_tab = min(_mtime(ctx.p(n)) for n in ("matches.parquet", "rounds.parquet"))
            later = [x for x in newest if x[0] > t_tab + P.tome_newer_than_tables_slack_s]
            t_grid, t_rounds = _mtime(ctx.p("grid.parquet")), _mtime(ctx.p("rounds.parquet"))
            msg = (f"newest tome file {_ts(t_tome[0])} ({os.path.basename(os.path.dirname(t_tome[1]))}), matches/rounds built {_ts(t_tab)}, "
                   f"grid built {_ts(t_grid)}; {len(later):,} tome file(s) newer than the prepared tables")
            if t_grid + P.tome_newer_than_tables_slack_s < t_rounds:
                return fail(msg + "; grid is OLDER than rounds.parquet", n=len(newest))
            return verdict(not later, msg, n=len(newest))
        R.run("S4", "prepared tables are not older than the tome files they were built from (no stale snapshot)", s4)

    def s3():
        pl = ctx.pl
        M, Rn = ctx.matches, ctx.rounds
        mk = set(M["match_key"].to_list())
        parts = []
        if getattr(ctx, "header_keys", None) is not None:
            hk = ctx.header_keys
            parts.append(f"matches.parquet {len(mk):,} vs header {len(hk):,} (only in matches {len(mk - hk):,}, only in header {len(hk - mk):,})")
            bad = len(mk ^ hk)
        else:
            bad = 0
        r_keys = set(Rn["match_key"].unique().to_list())
        parts.append(f"rounds {Rn.height:,} rows / {len(r_keys):,} matches, {len(r_keys - mk):,} not in matches.parquet")
        bad += len(r_keys - mk)
        from checks import grid_checks
        g = grid_checks.scan_grid(ctx)[1].select("match_key", "round")  # one chunked pass over the grid (cached), not a 19 M-row unique()
        used = ctx.used.select("match_key", "round")
        extra = g.join(used, on=["match_key", "round"], how="anti").height
        lack = used.join(g, on=["match_key", "round"], how="anti").height
        parts.append(f"grid covers {g.height:,} rounds; used rounds {used.height:,}; grid rounds not used {extra:,}; used rounds without grid {lack:,}")
        bad += extra + lack
        dth = pl.scan_parquet(ctx.p("deaths.parquet")).select("match_key", "round").unique().collect()
        d_extra = dth.join(Rn.select("match_key", "round"), on=["match_key", "round"], how="anti").height
        parts.append(f"deaths rounds not in rounds {d_extra:,}")
        bad += d_extra
        return verdict(bad == 0, "; ".join(parts), n=used.height)
    if ctx.missing("matches.parquet", "rounds.parquet", "grid.parquet", "deaths.parquet"):
        R.add("S3", "matches / rounds / deaths / grid agree on the set of matches and rounds", notrun("prepared tables missing"))
    else:
        R.run("S3", "matches / rounds / deaths / grid agree on the set of matches and rounds", s3)


# ------------------------------------------------------------------ A: scope
def scope_checks(ctx):
    R, P = ctx.R, ctx.P
    pl = ctx.pl
    if not ctx.want("A"):
        return
    um, U, M = ctx.used_matches, ctx.used, ctx.matches
    maps = ctx.maps

    def nviol(df, pred):  # a NULL in the checked column counts as a violation (polars filter would drop it silently)
        return df.filter(pred.fill_null(True)).height

    def a1():
        if not maps:
            return notrun("no map list: neither the model meta (scope.maps / map_is_*) nor --maps is available")
        bad = nviol(um, ~pl.col("map_name").is_in(maps))
        return verdict(bad == 0, f"{um.height} used matches; maps {sorted(maps)}; {bad} outside", n=um.height)
    R.run("A1", "used matches are all on the model's maps (list from the model meta or --maps, not from prepare)", a1, min_n=P.min_used_matches)

    def a1b():
        bad_flag = nviol(M, pl.col("in_scope") != (pl.col("scope_reason") == "in_scope"))
        bad_used = nviol(U, ~pl.col("in_scope"))
        bad_um = nviol(um, pl.col("scope_reason") != "in_scope")
        return verdict(bad_flag + bad_used + bad_um == 0, f"in_scope != (scope_reason == in_scope) on {bad_flag} matches; used rounds not in scope {bad_used}; "
                                                            f"used matches with another scope_reason {bad_um}", n=M.height)
    R.run("A1b", "the in_scope flag follows scope_reason and no used round lies outside scope", a1b, min_n=1)

    def a2():
        bad = nviol(um, pl.col("is_wingman_final") | pl.col("is_hostage_map"))
        return verdict(bad == 0, f"{um.height} used matches, {bad} wingman or hostage", n=um.height)
    R.run("A2", "no wingman or hostage among used matches", a2, min_n=P.min_used_matches)

    def a3():
        bad = nviol(U, pl.col("player_count_norm") != P.players_per_round)
        side = nviol(U, ~(((pl.col("n_t") == P.players_per_side) & (pl.col("n_ct") == P.players_per_side)) | pl.col("missing_side").is_not_null()))
        return verdict(bad + side == 0, f"{U.height} used rounds; player_count != {P.players_per_round}: {bad}; roster not {P.players_per_side}v{P.players_per_side} "
                                         f"and without an inferred side: {side}", n=U.height)
    R.run("A3", "every used round is a complete 5v5 (player_count_norm == 10; 5 listed per side or one inferred)", a3, min_n=P.min_used_rounds)

    def a3b():
        n_inf = int(U["missing_side"].is_not_null().sum())
        share = n_inf / max(U.height, 1)
        if share > P.inferred_side_max_share:
            return fail(f"{n_inf:,} of {U.height:,} used rounds ({share:.2%}) have one player's side inferred; bound {P.inferred_side_max_share:.2%}", n=U.height)
        if n_inf:
            return warn(f"{n_inf:,} of {U.height:,} used rounds ({share:.2%}) have one player's side inferred (missing_side set); the inference disagreed "
                        f"with the event log in ~40% of checked cases (wp_common); bound {P.inferred_side_max_share:.2%}", n=U.height)
        return ok("no used round has an inferred side", n=U.height)
    R.run("A3b", "inferred-side rounds are a bounded minority (WARN: known limitation)", a3b, min_n=P.min_used_rounds)

    R.run("A4", "no used round has a disconnect or bot takeover",
          lambda: verdict(int(U["disc_or_bot"].fill_null(True).sum()) == 0, f"{U.height} used rounds", n=U.height), min_n=P.min_used_rounds)

    def a5():
        dis = M.filter(pl.col("wingman_flag_disagrees").fill_null(False))
        bad = U.join(dis.select("match_key"), on="match_key", how="semi").height
        share = bad / max(U.height, 1)
        msg = (f"{dis.height} matches where the header's wingman flag disagrees with the roster (prep_report wingman_flag_disagrees); {bad} of {U.height:,} used rounds "
               f"({share:.3%}) sit inside them; bound {P.wingman_disagree_max_share:.2%}")
        if share > P.wingman_disagree_max_share:
            return fail(msg, n=M.height)
        if bad:
            return warn(msg + " (known data defect: the header's wingman flag and the roster disagree; those few rounds are still used)", n=M.height)
        return ok(msg, n=M.height)
    R.run("A5", "header and roster agree on wingman for every match that contributes a used round (WARN within a bound)", a5, min_n=1)

    def a6():
        oop = M.filter(pl.col("scope_reason") == "map_out_of_pool")
        if oop.height == 0:
            return ok("no match was excluded as out-of-pool", n=M.height)
        names = dict(oop.group_by("map_name").len().sort("len", descending=True).iter_rows())
        return warn(f"{oop.height:,} matches excluded as out-of-pool maps: {_top(names, 8)}", n=M.height)
    R.run("A6", "excluded maps are reported by name (a new or misspelled map is not silently dropped)", a6, min_n=1)


# ------------------------------------------------------------------ K: completeness and numbering
def completeness_checks(ctx):
    R, P = ctx.R, ctx.P
    pl = ctx.pl
    if not ctx.want("K"):
        return

    def k1():
        M, Rn = ctx.matches, ctx.rounds
        cnt = Rn.group_by("match_key").agg(pl.len().alias("n_rd"), pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"),
                                           pl.col("round").n_unique().alias("nu"))
        X = (M.select("match_key", "era", "build_num", "t_starters_score_final", "ct_starters_score_final").join(cnt, on="match_key", how="inner")
             .with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("tot"),
                           (pl.col("era").cast(pl.String) + "|" + pl.col("build_num").cast(pl.String)).alias("key")))

        def table(by):
            return X.group_by(by).agg(
                pl.len().alias("n"), pl.col("tot").is_not_null().sum().alias("with_total"),
                (pl.col("n_rd") == pl.col("tot")).sum().alias("complete"), (pl.col("n_rd") < pl.col("tot")).sum().alias("fewer"),
                (pl.col("n_rd") > pl.col("tot")).sum().alias("more"), (pl.col("rmin") > 1).sum().alias("from_gt1"),
                ((pl.col("rmax") - pl.col("rmin") + 1) != pl.col("nu")).sum().alias("gaps")).sort(by)
        per_key, per_era = table("key"), table("era")
        failures, notes = [], []
        for row in per_key.iter_rows(named=True):
            k, n = row["key"], row["n"]
            inc = row["fewer"] / max(row["with_total"], 1)
            g1 = row["from_gt1"] / max(n, 1)
            ctx.record("completeness", k, {"matches": n, "incomplete_share": round(inc, 4), "from_gt1_share": round(g1, 4)})
            base = ctx.baseline_get("completeness", k)
            if base and n >= P.completeness_min_matches:
                for name, cur in (("incomplete_share", inc), ("from_gt1_share", g1)):
                    if cur > base.get(name, 0.0) + P.completeness_baseline_tol:
                        failures.append(f"{k} {name} {cur:.1%} worse than baseline {base.get(name, 0.0):.1%}")
            if row["more"] / max(n, 1) > P.more_rounds_than_total_max_share:
                failures.append(f"{k}: {row['more']} matches have MORE rounds than the header total")
            if row["fewer"] or row["from_gt1"] or row["gaps"]:
                notes.append(f"{k}: {row['fewer']}/{row['with_total']} incomplete ({inc:.0%}), {row['from_gt1']} numbered from round>1, {row['gaps']} with gaps")
        for row in per_era.iter_rows(named=True):
            inc = row["fewer"] / max(row["with_total"], 1)
            if inc > P.completeness_hard_bound:
                failures.append(f"era {row['era']}: incomplete share {inc:.1%} exceeds the hard bound {P.completeness_hard_bound:.0%}")
        tot_inc = int(per_era["fewer"].sum()) / max(int(per_era["with_total"].sum()), 1)
        head = f"{X.height:,} matches, {int(per_era['fewer'].sum()):,} with fewer rounds than the header total ({tot_inc:.0%} overall)"
        if failures:
            return fail(head + "; " + "; ".join(failures[:6]), n=X.height)
        if notes:
            return warn(head + "; known limitation: " + "; ".join(notes), n=X.height)
        return ok(head + ": every match is complete and numbered from 1 without gaps", n=X.height)
    R.run("K1", "round completeness by era|build: rounds present vs the header score total, numbering 1..n (WARN: documented limitation)", k1, min_n=1)


# ------------------------------------------------------------------ B: duplicates
def duplicate_checks(ctx):
    R, P = ctx.R, ctx.P
    pl = ctx.pl
    if not ctx.want("B"):
        return
    um, M, Rn = ctx.used_matches, ctx.matches, ctx.rounds

    def b1():
        kept = um.filter(pl.col("dup_group_size") > 1).group_by("sig_hash").len()
        bad = kept.filter(pl.col("len") > 1).height
        return verdict(bad == 0, f"{kept.height} duplicate groups with a used member; {bad} with two or more", n=um.height)
    R.run("B1", "at most one used match per exact-signature duplicate group", b1, min_n=P.min_used_matches)
    R.run("B2", "all used matches are duplicate keepers",
          lambda: verdict(um.filter(~pl.col("dup_keeper").fill_null(False)).height == 0, f"{um.height} used matches", n=um.height), min_n=P.min_used_matches)

    def b3():
        per = (Rn.sort(["match_key", "round"]).group_by("match_key", maintain_order=True)
               .agg(pl.len().alias("n"), pl.col("round").min().alias("r0"), pl.col("rec").cast(pl.String).fill_null("x").str.join("").alias("wseq"),
                    (pl.col("end") - pl.col("freeze").fill_null(pl.col("end"))).alias("lens"), pl.col("use").fill_null(False).any().alias("used"))
               .join(M.select("match_key", "map_name", "t_starters_score_final", "ct_starters_score_final"), on="match_key", how="inner")
               .filter(pl.col("n") >= P.dup_min_rounds))
        grp = (per.group_by(["map_name", "t_starters_score_final", "ct_starters_score_final", "n", "r0", "wseq"])
               .agg(pl.col("match_key"), pl.col("lens"), pl.col("used")).filter(pl.col("match_key").list.len() > 1))
        import numpy as np
        n_dup_clusters, n_coincidental, bad = 0, 0, []
        for row in grp.iter_rows(named=True):
            keys, lens, used = row["match_key"], [np.asarray(x, dtype=np.int64) for x in row["lens"]], row["used"]
            parent = list(range(len(keys)))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    if int(np.abs(lens[i] - lens[j]).max()) <= P.dup_tick_tol:
                        parent[find(i)] = find(j)
                    else:
                        n_coincidental += 1
            clusters = {}
            for i in range(len(keys)):
                clusters.setdefault(find(i), []).append(i)
            for members in clusters.values():
                if len(members) > 1:
                    n_dup_clusters += 1
                    n_used = sum(1 for i in members if used[i])
                    if n_used > 1:
                        bad.append([keys[i] for i in members if used[i]][:3])
        msg = (f"{per.height:,} matches with >= {P.dup_min_rounds} rounds; {grp.height} groups share map/score/round count/winner sequence; "
               f"{n_dup_clusters} timeline-identical duplicate clusters (round lengths within {P.dup_tick_tol} ticks), {n_coincidental} coincidental pairs separated; "
               f"{len(bad)} clusters with more than one used match")
        return verdict(not bad, msg + (f"; e.g. {bad[0]}" if bad else ""), n=per.height)
    R.run("B3", "duplicates re-derived independently of prepare (map, score, winner sequence + round-length timeline): at most one used match per cluster", b3, min_n=1)

    def b4():
        cand = M.filter(pl.col("n_rounds_re").fill_null(0) >= P.dup_min_rounds)
        nul = cand.filter(pl.any_horizontal(*[pl.col(c).is_null() for c in ("map_name", "r0", "n_rounds_re", "t_starters_score_final", "ct_starters_score_final")]))
        return verdict(nul.height == 0, f"{cand.height} matches with >= {P.dup_min_rounds} rounds, {nul.height} with a null in a field of the tolerant duplicate key", n=cand.height)
    R.run("B4", "the tolerant duplicate key has no null part (a null collapses unrelated matches into one group)", b4, min_n=1)
