"""Paper-scale word2vec embeddings: 20 random samples of 200 de_mirage matches, each embedding fitted on that sample only.

The Chahrour (2023) paper used 200 CS:GO games of de_mirage only. This reproduces that scale: a sample of 200 matches, the embedding
fitted on the sample's own sentences (the paper fits on all of its demos, which here means the sample), and the paper's settings
(Word2Vec vector_size 8, window 5, min_count 1, gensim defaults otherwise), with the movement-gated sentences of w2v_published_embed.py.

Writes data/diagnostics/w2v_paper_scale/sample{k}.parquet: match_key, round, second, known, e0..e7 for every token of the sample's matches.
Run in the navigation environment (gensim):
  C:/Users/13144/pureskill-nav312/Scripts/python.exe scripts/model/w2v_paper_scale_embed.py
"""
import glob
import os
import sys
import time

import numpy as np
import polars as pl
from gensim.models import Word2Vec

sys.path.insert(0, os.path.dirname(__file__))
from w2v_published_embed import first_movement  # noqa: E402

ROOT = "G:/pureskill/csds-cs2"
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUT = os.path.join(DIAG, "w2v_paper_scale")
N_SAMPLES = 20
N_MATCHES = 200
SKIP = 3
KEY = ["match_key", "round", "second"]


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    tok = pl.read_parquet(os.path.join(DIAG, "w2v_tokens_de_mirage.parquet"))
    matches = sorted(set(tok["match_key"].to_list()))
    print(f"de_mirage matches with tokens: {len(matches):,}", flush=True)
    fm = first_movement(matches)
    s0 = tok.group_by(["match_key", "round"]).agg(pl.col("second").min().alias("s0"))
    base = (tok.join(s0, on=["match_key", "round"]).join(fm, on=["match_key", "round"], how="inner")
               .filter(((pl.col("second") - pl.col("s0")) % SKIP == 0) & (pl.col("second") >= pl.col("first_move")))
               .sort(["match_key", "round", "second"]))
    print(f"sentence tokens: {base.height:,} ({time.time() - t0:.0f} s)", flush=True)
    for k in range(N_SAMPLES):
        rng = np.random.default_rng(1000 + k)
        sample = sorted(rng.choice(matches, size=N_MATCHES, replace=False).tolist())
        sub = base.filter(pl.col("match_key").is_in(sample))
        sents = sub.group_by(["match_key", "round"], maintain_order=True).agg(pl.col("token")).get_column("token").to_list()
        model = Word2Vec(sentences=sents, vector_size=8, window=5, min_count=1, workers=8, seed=k)
        vocab = set(model.wv.key_to_index.keys())
        toks = tok.filter(pl.col("match_key").is_in(sample)).select(KEY + ["token"])
        tk = toks["token"].to_list()
        known = np.array([1 if t in vocab else 0 for t in tk], dtype=np.int64)
        vecs = np.zeros((toks.height, 8), dtype=np.float64)
        for i, t in enumerate(tk):
            if known[i]:
                vecs[i] = model.wv[t]
        out = toks.select(KEY).with_columns(pl.Series("known", known)).with_columns(
            [pl.Series(f"e{j}", vecs[:, j]) for j in range(8)])
        out.write_parquet(os.path.join(OUT, f"sample{k}.parquet"))
        print(f"sample {k}: {len(sents):,} sentences, vocab {len(vocab):,}, known share {known.mean():.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
