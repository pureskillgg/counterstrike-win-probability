"""Word2vec sentences built the way the published code builds them (codeAnalysis.py getSentencesForMatch, lines 120-167), de_mirage 2026.

Differences from w2v_gensim_embed.py, which includes the freeze time before anyone moves:
  - a round's sentence starts at its first movement second (published: ticks before firstMovementTick are skipped). Checked on
    2,439 rounds of 150 matches: the 1 Hz positions already show movement one second after the round's first row, so this drops
    at most the first sample of each round (there is no long freeze time in the positions)
  - movement is found from the 1 Hz positions, because they carry no velocity: a second counts as moving once any player's x, y or z
    differs from that player's first row of the round by more than 0.001. The published code uses velocity > 0 at 64 ticks/s, so this
    can start a second late
  - a round with no movement gives no sentence (published: an empty sentence)
Otherwise the same: one token every 3 seconds from round start, Word2Vec(vector_size=8, window=5, min_count=1, seed=0, epochs=5),
honest = train-split rounds only, leaky = all rounds.

Output: data/diagnostics/w2v_pubsent_{honest,leaky}.parquet, columns match_key, round, second, known, e0..e7 (same format as w2v_gensim_*).
Run in the navigation environment (it has gensim):
  C:/Users/13144/pureskill-nav312/Scripts/python.exe scripts/model/w2v_published_embed.py
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import polars as pl
from gensim.models import Word2Vec

ROOT = "G:/pureskill/csds-cs2"
DIAG = os.path.join(ROOT, "data", "diagnostics")
SKIP = 3
EPS = 1e-3


def first_movement(keys):
    """Per (match_key, round): the earliest second at which any player has left its first row of the round."""
    rows = []
    g = ["match_key", "round", "player_id"]
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "2026-*.parquet"))):
        p = pl.read_parquet(f, columns=["match_key", "round", "second", "player_id", "x", "y", "z"])
        p = p.filter(pl.col("match_key").is_in(keys))
        if p.height == 0:
            continue
        p = p.sort(g + ["second"]).with_columns(
            [pl.col(c).first().over(g).alias(f"{c}0") for c in ["x", "y", "z"]])
        disp = (pl.col("x") - pl.col("x0")).abs() + (pl.col("y") - pl.col("y0")).abs() + (pl.col("z") - pl.col("z0")).abs()
        moved = p.filter(disp > EPS)
        rows.append(moved.group_by(["match_key", "round"]).agg(pl.col("second").min().alias("first_move")))
    return pl.concat(rows).group_by(["match_key", "round"]).agg(pl.col("first_move").min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="w2v_pubsent")
    ap.add_argument("--sg", type=int, default=0, help="0 = CBOW (gensim default), 1 = skip-gram")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--sample", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    tok = pl.read_parquet(os.path.join(DIAG, "w2v_tokens_de_mirage.parquet"))
    man = json.load(open(os.path.join(ROOT, "data", "combined", "split_manifest.json"), encoding="utf-8"))["assign"]
    train_keys = [k for k, v in man.items() if v == "train"]
    keys = sorted(set(tok["match_key"].to_list()))
    n_rounds = tok.select(["match_key", "round"]).unique().height
    fm = first_movement(keys)
    print(f"rounds {n_rounds:,}; rounds with movement {fm.height:,}; {time.time() - t0:.0f} s", flush=True)

    s0 = tok.group_by(["match_key", "round"]).agg(pl.col("second").min().alias("s0"))
    samp = (tok.join(s0, on=["match_key", "round"])
               .join(fm, on=["match_key", "round"], how="inner")
               .filter(((pl.col("second") - pl.col("s0")) % SKIP == 0) & (pl.col("second") >= pl.col("first_move")))
               .sort(["match_key", "round", "second"]))

    KEY = ["match_key", "round", "second"]
    for version in ["honest", "leaky"]:
        d = samp if version == "leaky" else samp.filter(pl.col("match_key").is_in(train_keys))
        sents = d.group_by(["match_key", "round"], maintain_order=True).agg(pl.col("token")).get_column("token").to_list()
        print(f"{version}: {len(sents):,} sentences, {sum(len(s) for s in sents):,} words", flush=True)
        model = Word2Vec(sentences=sents, vector_size=8, window=args.window, min_count=1, workers=8, seed=0,
                         epochs=args.epochs, sg=args.sg, sample=args.sample)
        vocab = set(model.wv.key_to_index.keys())
        print(f"{version}: vocabulary {len(vocab):,}, training {time.time() - t0:.0f} s", flush=True)
        toks = tok.select(KEY + ["token"])
        tk = toks["token"].to_list()
        known = np.array([1 if t in vocab else 0 for t in tk], dtype=np.int64)
        vecs = np.zeros((toks.height, 8), dtype=np.float64)
        for i, t in enumerate(tk):
            if known[i]:
                vecs[i] = model.wv[t]
        out = toks.select(KEY).with_columns(pl.Series("known", known)).with_columns(
            [pl.Series(f"e{j}", vecs[:, j]) for j in range(8)])
        out.write_parquet(os.path.join(DIAG, f"{args.stem}_{version}.parquet"))
        print(f"{version}: unknown share over all seconds {1 - known.mean():.3f}; wrote", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
