"""Frozen, append-only train / validation / test assignment of matches (read the comment above `hash_assign` in wp_common.py for why).

  python -I -X utf8 scripts\\model\\split_manifest.py --freeze    # once: record the split the existing models used (legacy rule on the current grid)
  python -I -X utf8 scripts\\model\\split_manifest.py --extend    # after every data merge + make_grid: add the new matches (hash rule); old entries never change
  python -I -X utf8 scripts\\model\\split_manifest.py --verify    # counts per era and set, disjointness, every grid match assigned

Writes data/<store>/split_manifest.json (atomic temp + os.replace). Every script that calls wp_common.make_split("random") reads it.
--freeze refuses to overwrite an existing manifest: the whole point is that a held-out match stays held out.
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def grid_matches(store):
    d = os.path.join(W.ROOT, "data", store)
    g = pl.scan_parquet(os.path.join(d, "grid.parquet")).select("match_key", "era").unique("match_key").collect()
    return g


def write_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=0, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def digest(assign):
    h = hashlib.sha256()
    for k in sorted(assign):
        h.update(f"{k}={assign[k]}\n".encode())
    return h.hexdigest()


def report(man, g):
    era = dict(zip(g["match_key"].to_list(), g["era"].to_list()))
    counts = {}
    for k, s in man["assign"].items():
        if k in era:
            counts.setdefault(era[k], {"train": 0, "val": 0, "test": 0})[s] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--extend", action="store_true")
    ap.add_argument("--extend-train", action="store_true",
                    help="add new grid matches as train only; existing assignments, the frozen test and validation are never touched")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(W.ROOT, "data", a.store, "split_manifest.json")
    g = grid_matches(a.store)
    keys = g["match_key"].to_list()
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if a.freeze:
        if os.path.exists(out):
            raise SystemExit(f"{out} exists: a frozen split is never rewritten (use --extend to add matches)")
        G = W.load_grid(a.store, cols=["match_key", "era"])
        sp = W.make_split("random_legacy", G, seed=a.seed)
        assign = {k: s for s in ("train", "val", "test") for k in sp[s]}
        man = {"rule": "legacy: per-era shuffle (random.Random(seed)) of the sorted match keys, 20% test, 10% of the rest validation; "
                       "later matches: sha256(seed:match_key) -> test <0.20, val <0.28, else train",
               "seed": a.seed, "created_at": now, "frozen_from": {"store": a.store, "n_matches": len(keys)}, "history": [{"at": now, "event": "freeze", "added": len(assign)}],
               "assign": assign}
        man["sha256"] = digest(assign)
        write_atomic(out, man)
        print(f"froze {len(assign):,} matches -> {out}")
    elif a.extend or a.extend_train:
        if a.extend and a.extend_train:
            raise SystemExit("choose one of --extend (hash rule) or --extend-train (train only)")
        man = W.load_manifest(out)
        if man is None:
            raise SystemExit(f"{out} does not exist: run --freeze first")
        before = dict(man["assign"])
        new = [k for k in keys if k not in before]
        for k in new:
            man["assign"][k] = "train" if a.extend_train else W.hash_assign(k, man["seed"])
        assert all(man["assign"][k] == v for k, v in before.items()), "an existing assignment changed"
        event = "extend_train" if a.extend_train else "extend"
        man["history"].append({"at": now, "event": event, "added": len(new)})
        man["sha256"] = digest(man["assign"])
        write_atomic(out, man)
        print(f"added {len(new):,} matches (now {len(man['assign']):,}) -> {out}")
    else:
        man = W.load_manifest(out)
        if man is None:
            raise SystemExit(f"{out} does not exist")
    # verification (always)
    problems = []
    if man["sha256"] != digest(man["assign"]):
        problems.append("manifest sha256 does not match its assignments")
    if set(man["assign"].values()) - {"train", "val", "test"}:
        problems.append("unknown set name in the manifest")
    missing = [k for k in keys if k not in man["assign"]]
    if missing:
        problems.append(f"{len(missing):,} grid matches are not assigned (run --extend)")
    counts = report(man, g)
    for era, c in sorted(counts.items()):
        n = sum(c.values())
        print(f"  {era}: {n:,} matches  train {c['train']:,} ({c['train'] / n:.1%})  val {c['val']:,} ({c['val'] / n:.1%})  test {c['test']:,} ({c['test'] / n:.1%})")
    print(f"manifest {man['sha256'][:16]}  history: {[(h['event'], h['added']) for h in man['history']]}")
    if problems:
        print("PROBLEMS:", problems)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
