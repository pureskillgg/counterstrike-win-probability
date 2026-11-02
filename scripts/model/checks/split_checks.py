"""Split checks (E): the train / validation / test assignment of matches, with and without the frozen split manifest."""
import glob
import hashlib
import json
import os

from checks import core
from checks.core import fail, notrun, ok, verdict, warn

SETS = ("train", "val", "test")


def manifest_digest(assign):
    """Same digest as scripts/model/split_manifest.py: sha256 over the sorted 'key=set' lines."""
    h = hashlib.sha256()
    for k in sorted(assign):
        h.update(f"{k}={assign[k]}\n".encode())
    return h.hexdigest()


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        man = json.load(f)
    if not isinstance(man.get("assign"), dict) or not man["assign"]:
        raise ValueError("split manifest has no assignments")
    return man


def provenance(ctx):
    """None if the model in meta.json was trained on the grid in this store (first-MiB sha1 recorded at export), else the reason."""
    meta = ctx.meta
    if meta is None:
        return f"no model meta at {ctx.meta_path}"
    want256, want = meta.get("grid_file_sha256"), meta.get("grid_file_sha1")
    if not (want256 or want):
        return "meta.json records neither grid_file_sha256 nor grid_file_sha1"
    if not os.path.exists(ctx.p("grid.parquet")):
        return "no grid.parquet"
    if want256:  # models exported since 2026-10-15 record the sha256 of the WHOLE grid file
        import hashlib
        h = hashlib.sha256()
        with open(ctx.p("grid.parquet"), "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        have = h.hexdigest()
        if have != want256:
            return f"the model was trained on another grid (meta grid_file_sha256 {want256[:8]}, this store's grid {have[:8]}): the model is stale for this store"
        return None
    have = core.sha1_first_mib(ctx.p("grid.parquet"))
    if have != want:
        return f"the model was trained on another grid (meta grid_file_sha1 {want[:8]}, this store's grid {have[:8]}): the model is stale for this store"
    return None


def split_checks(ctx):
    R, P = ctx.R, ctx.P
    pl = ctx.pl
    if not ctx.want("E"):
        return
    miss = ctx.missing("grid.parquet", "matches.parquet", "rounds.parquet")
    if miss:
        R.add("E*", "split checks", notrun(f"missing {miss}"))
        return
    W = ctx.wp
    from checks import grid_checks
    try:
        in_grid = grid_checks.scan_grid(ctx)[1].select("match_key").unique()  # matches that have grid rows (one chunked pass, cached)
    except Exception as exc:  # noqa: BLE001 - an empty or unreadable grid is a FAIL, not a crash
        R.add("E1", "train/val/test cover exactly the grid's matches", fail(f"the grid cannot be scanned ({type(exc).__name__}: {str(exc)[:200]})", n=0))
        return
    Gm = (in_grid.join(ctx.rounds.select("match_key", "era").unique("match_key"), on="match_key", how="left")
          .join(ctx.matches.select("match_key", "match_date"), on="match_key", how="left"))
    grid_keys = set(Gm["match_key"].to_list())
    man_path = ctx.split_manifest_path()
    man = None
    if man_path:
        try:
            man = load_manifest(man_path)
        except Exception as exc:  # noqa: BLE001
            R.add("E0", "split manifest loads", fail(f"{man_path}: {type(exc).__name__}: {exc}"))

    def pipeline_split(frame=None, name="random", seed=None):
        """The split exactly as wp_common.make_split would hand it to a training script: the manifest when there is one, else the legacy rule."""
        old = getattr(W, "SPLIT_MANIFEST", None)
        W.SPLIT_MANIFEST = man_path if (man_path and man) else "none"
        try:
            return W.make_split(name, Gm if frame is None else frame, seed=P.split_seed if seed is None else seed)
        finally:
            W.SPLIT_MANIFEST = old
    ctx._c["pipeline_split"] = pipeline_split

    def e1():
        sp = pipeline_split()
        a, b, c = sp["train"], sp["val"], sp["test"]
        overl = len(a & b) + len(a & c) + len(b & c)
        return verdict(overl == 0 and (a | b | c) == grid_keys,
                       f"{'manifest' if man else 'legacy rule'}: train {len(a):,} / val {len(b):,} / test {len(c):,} of {len(grid_keys):,} grid matches; overlaps {overl}; "
                       f"unassigned {len(grid_keys - (a | b | c))}, assigned but not in the grid {len((a | b | c) - grid_keys)}", n=len(grid_keys))
    R.run("E1", "train/val/test are disjoint and cover exactly the grid's matches (the split a training script would get)", e1, min_n=P.min_used_matches)

    def e2():
        used = set(ctx.used["match_key"].unique().to_list())
        sp = pipeline_split()
        allm = sp["train"] | sp["val"] | sp["test"]
        return verdict(allm == used, f"{len(used):,} matches with a used round; split holds {len(allm):,}; only in split {len(allm - used)}, only used {len(used - allm)}", n=len(used))
    R.run("E2", "the split covers every match with a used round (not only the grid's own match set)", e2, min_n=P.min_used_matches)

    def e3():
        keep = set(ctx.matches.filter(pl.col("dup_keeper").fill_null(False))["match_key"].to_list())
        sp = pipeline_split()
        allm = sp["train"] | sp["val"] | sp["test"]
        return verdict(allm <= keep, f"{len(allm):,} split matches, {len(allm - keep)} are not duplicate keepers", n=len(allm))
    R.run("E3", "only duplicate keepers are in the split", e3, min_n=P.min_used_matches)

    def e4():
        from checks import unit
        return unit.split_rule_result(W, Gm, P.test_frac, P.val_frac, P.split_seed)
    R.run("E4", "the legacy random split has the documented sizes per era: test = 20%, validation = 10% of the rest (val and test not swapped)", e4, min_n=P.min_used_matches)

    def data_keys():
        """The matches the model's own training data still holds. A snapshot model is fitted on the cache (make_snapshot_cache.py), which leaves out rounds
        with a roster defect or a table-vs-events mismatch: a match that loses ALL its rounds is not in the model's data, so its manifest assignment
        cannot be in the model's match counts (2026-10-15: 7 train + 1 val + 2 test matches). None when the model is not a snapshot model or there is no cache."""
        feats = (ctx.meta or {}).get("features") or []
        cache = ctx.p("cache_snapshot_2026.parquet")
        if not any(str(f).startswith("st_") for f in feats) or not os.path.exists(cache):
            return None
        return set(pl.scan_parquet(cache).select("match_key").unique().collect()["match_key"].to_list())

    # ---- the frozen manifest
    if man is not None:
        assign = man["assign"]

        def m1():
            d = manifest_digest(assign)
            return verdict(d == man.get("sha256"), f"stored {str(man.get('sha256'))[:12]} vs recomputed {d[:12]} over {len(assign):,} assignments", n=len(assign))
        R.run("E5", "split manifest: sha256 matches its content", m1, min_n=1)

        def m2():
            names = set(assign.values())
            unknown = names - set(SETS)
            missing = grid_keys - set(assign)
            return verdict(not unknown and not missing, f"{len(grid_keys):,} grid matches; not in the manifest {len(missing):,}; manifest sets {sorted(names)}"
                           f"{'; UNKNOWN set names ' + str(sorted(unknown)) if unknown else ''}", n=len(grid_keys))
        R.run("E6", "split manifest: covers every grid match and uses only train / val / test", m2, min_n=P.min_used_matches)

        def m3():
            prob = provenance(ctx)
            if prob:
                return notrun(prob)
            eras = list(((ctx.meta.get("training") or {}).get("eras") or {}).keys())
            keys = set(Gm.filter(pl.col("era").is_in(eras))["match_key"].to_list()) if eras else grid_keys
            dk = data_keys()
            note = ""
            if dk is not None:  # only matches that still have rows in the model's own training data (see data_keys)
                note = f" (restricted to the {len(keys & dk):,} of {len(keys):,} matches that still have rows in the snapshot cache)"
                keys = keys & dk
            tr = sum(1 for k in keys if assign.get(k) == "train")
            va = sum(1 for k in keys if assign.get(k) == "val")
            t = ctx.meta.get("training") or {}
            return verdict((tr, va) == (t.get("matches"), t.get("val_matches")),
                           f"eras {eras or 'all'}: manifest train/val {tr:,}/{va:,} vs the model's meta {t.get('matches')}/{t.get('val_matches')}{note}", n=len(keys))
        R.run("E7", "the model's training / validation match counts equal the manifest's counts for the matches it used", m3, min_n=1)

    def e8():
        sp = pipeline_split()
        arts = sorted(glob.glob(ctx.p("export_*_test_sample.parquet")) + glob.glob(ctx.p("export_*_test_predictions.parquet")))
        if not arts:
            return notrun("no export_*_test_sample / test_predictions artifact in the store")
        lines, bad, n_tot = [], [], 0
        for f in arts:
            mk = set(pl.scan_parquet(f).select("match_key").unique().collect()["match_key"].to_list())
            leak = len(mk & (sp["train"] | sp["val"]))
            out = len(mk - sp["test"])
            n_tot += len(mk)
            lines.append(f"{os.path.basename(f)}: {len(mk):,} matches, in train/val {leak}, outside test {out}")
            if leak or out:
                bad.append(os.path.basename(f))
        return verdict(not bad, "; ".join(lines) + (f"; LEAK: {bad}" if bad else ""), n=n_tot)
    R.run("E9", "no exported test sample / test prediction file contains a training or validation match (cross-artifact leakage)", e8, min_n=1)

    def e10():
        prob = provenance(ctx)
        if prob:
            return notrun(prob)
        t = ctx.meta.get("training") or {}
        eras = list((t.get("eras") or {}).keys())
        frame = Gm.filter(pl.col("era").is_in(eras)) if eras else Gm
        dk = data_keys()
        if dk is not None:  # the split a training script builds is over the matches its own data holds
            frame = frame.filter(pl.col("match_key").is_in(sorted(dk)))
        sp = pipeline_split(frame, name=t.get("split", "random"), seed=ctx.meta.get("seed", P.split_seed))
        got = (len(sp["train"]), len(sp["val"]))
        want = (t.get("matches"), t.get("val_matches"))
        msg = f"split '{t.get('split')}' seed {ctx.meta.get('seed')} eras {eras or 'all'}: train/val {got} vs meta {want}"
        if not os.path.exists(ctx.sample_path):
            return verdict(got == want, msg + "; no test sample to place", n=len(frame))
        smp = set(pl.scan_parquet(ctx.sample_path).select("match_key").unique().collect()["match_key"].to_list())
        in_test, in_other = len(smp & sp["test"]), len(smp - sp["test"])
        return verdict(got == want and in_other == 0, msg + f"; test-sample matches in test {in_test:,}, outside test {in_other}", n=len(smp))
    R.run("E10", "rebuilding the split from the model's own meta (split, seed, eras) reproduces its match counts and places every sample match in test", e10, min_n=1)
