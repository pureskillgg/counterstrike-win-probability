"""Train word2vec with the real gensim library, as the published code does, and write one embedding per second (de_mirage).

The published settings (codeAnalysis.py, line 200): gensim Word2Vec(sentences, vector_size=8, window=5, min_count=1), with the library
defaults for everything else (CBOW, negative sampling 5, alpha 0.025 to 0.0001, 5 epochs, frequent-token subsampling sample=1e-3).

Two versions, toggled in one place:
  honest  sentences from the TRAIN rounds of the frozen split only (what we did before)
  leaky   sentences from ALL rounds, as the published code does (train, validation and test)

Tokens are the per-second position strings from build_tokens (positions/256 rounded, one slot per player, '_' for a missing slot).
Sentences: one per round, a token every 3 seconds from the round start (the published interval).
Output: data/diagnostics/w2v_gensim_{honest,leaky}.parquet, one row per second with the 8-number vector of its token (zero if the
token is not in the vocabulary) and a 0/1 unknown flag.

Run in the navigation environment (it has gensim): C:/Users/<user>/pureskill-nav312/Scripts/python.exe
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
from gensim.models import Word2Vec

ROOT = "G:/pureskill/csds-cs2"
DIAG = os.path.join(ROOT, "data", "diagnostics")
SKIP = 3
KEY = ["match_key", "round", "second"]


def sentences_for(tok, round_keys):
    d = tok.filter(pl.col("match_key").is_in(list({k[0] for k in round_keys})))
    d = d.with_columns(pl.col("second").min().over(["match_key", "round"]).alias("s0")).filter(((pl.col("second") - pl.col("s0")) % SKIP) == 0)
    d = d.sort(["match_key", "round", "second"])
    out = []
    for (mk, rd), g in d.group_by(["match_key", "round"], maintain_order=True):
        if (mk, rd) in round_keys:
            out.append(g["token"].to_list())
    return out


def main():
    t0 = time.time()
    tok = pl.read_parquet(os.path.join(DIAG, "w2v_tokens_de_mirage.parquet"))
    man = json.load(open(os.path.join(ROOT, "data", "combined", "split_manifest.json"), encoding="utf-8"))["assign"]
    rounds = tok.select(["match_key", "round"]).unique().rows()
    all_rounds = set((mk, rd) for mk, rd in rounds)
    train_rounds = set((mk, rd) for mk, rd in rounds if man.get(mk) == "train")
    print(f"rounds: all {len(all_rounds):,}, train {len(train_rounds):,}", flush=True)

    for version, fit_rounds in [("honest", train_rounds), ("leaky", all_rounds)]:
        sents = sentences_for(tok, fit_rounds)
        print(f"{version}: {len(sents):,} sentences, {sum(len(s) for s in sents):,} words", flush=True)
        model = Word2Vec(sentences=sents, vector_size=8, window=5, min_count=1, workers=8, seed=0, epochs=5)
        vocab = set(model.wv.key_to_index.keys())
        print(f"{version}: vocabulary {len(vocab):,}, training {time.time() - t0:.0f} s", flush=True)
        toks = tok.select(KEY + ["token"]).with_columns(
            pl.col("token").map_elements(lambda t: 1 if t in vocab else 0, return_dtype=pl.Int64).alias("known"))
        vecs = np.zeros((toks.height, 8), dtype=np.float64)
        tk = toks["token"].to_list()
        for i, t in enumerate(tk):
            if t in vocab:
                vecs[i] = model.wv[t]
        out = toks.select(KEY + ["known"]).with_columns(
            [pl.Series(f"e{j}", vecs[:, j]) for j in range(8)])
        out.write_parquet(os.path.join(DIAG, f"w2v_gensim_{version}.parquet"))
        print(f"{version}: unknown share {1 - toks['known'].mean():.3f}; wrote", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
