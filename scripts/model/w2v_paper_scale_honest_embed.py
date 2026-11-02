"""Honest paper-scale word2vec embeddings: the embedding is fitted on the TRAINING matches of each 200-match sample only.

Same 20 samples as w2v_paper_scale_embed.py (sample k uses default_rng(1000 + k)). Within each sample, the matches are split by match
into 140 training and 60 held-out matches (default_rng(3000 + k) permutation of the sorted sample). Word2Vec (vector_size 8, window 5,
min_count 1, seed k) is fitted on the training matches' movement-gated sentences only, so held-out rows can only use tokens the training
matches produced.

Writes data/diagnostics/w2v_paper_scale_honest/sample{k}.parquet: match_key, round, second, known, e0..e7 for every token of the sample's
200 matches (known = 1 when the exact token is in the training vocabulary).
Run in the navigation environment (gensim):
  C:/Users/13144/pureskill-nav312/Scripts/python.exe scripts/model/w2v_paper_scale_honest_embed.py
"""
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
OUT = os.path.join(DIAG, "w2v_paper_scale_honest")
N_SAMPLES = 20
N_MATCHES = 200
N_TRAIN = 140
SKIP = 3
KEY = ["match_key", "round", "second"]


def split_sample(k):
    """The 200 sample matches of sample k and their training / held-out split."""
    matches = sorted(set(pl.read_parquet(os.path.join(DIAG, "w2v_tokens_de_mirage.parquet"), columns=["match_key"])["match_key"].to_list()))
    rng = np.random.default_rng(1000 + k)
    sample = sorted(rng.choice(matches, size=N_MATCHES, replace=False).tolist())
    perm = np.random.default_rng(3000 + k).permutation(len(sample))
    train = sorted(sample[i] for i in perm[:N_TRAIN])
    held = sorted(sample[i] for i in perm[N_TRAIN:])
    return sample, train, held


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    tok = pl.read_parquet(os.path.join(DIAG, "w2v_tokens_de_mirage.parquet"))
    matches = sorted(set(tok["match_key"].to_list()))
    fm = first_movement(matches)
    s0 = tok.group_by(["match_key", "round"]).agg(pl.col("second").min().alias("s0"))
    base = (tok.join(s0, on=["match_key", "round"]).join(fm, on=["match_key", "round"], how="inner")
               .filter(((pl.col("second") - pl.col("s0")) % SKIP == 0) & (pl.col("second") >= pl.col("first_move")))
               .sort(["match_key", "round", "second"]))
    print(f"sentence tokens {base.height:,} ({time.time() - t0:.0f} s)", flush=True)
    for k in range(N_SAMPLES):
        sample, train, held = split_sample(k)
        sub = base.filter(pl.col("match_key").is_in(train))
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
        out = toks.select(KEY).with_columns(pl.Series("known", known)).with_columns([pl.Series(f"e{j}", vecs[:, j]) for j in range(8)])
        out.write_parquet(os.path.join(OUT, f"sample{k}.parquet"))
        print(f"sample {k}: train matches {len(train)}, sentences {len(sents):,}, vocab {len(vocab):,}, "
              f"({time.time() - t0:.0f} s)", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
