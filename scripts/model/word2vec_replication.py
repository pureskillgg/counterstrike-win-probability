"""Word2vec-style position embedding, replicating kamalchah/CounterStrikeWinProb (codeAnalysis.py) on de_mirage, leakage removed.

The published method, as read from the repository source:
  - token = one string per tick for all ten players: each player's position divided by 256 and rounded, five T slots then five CT slots;
  - sentence = one round; words sampled every 3 seconds from the round start;
  - gensim Word2Vec(vector_size=8, window=5, min_count=1): CBOW (gensim default), negative sampling (5), 5 epochs, alpha 0.025 -> 0.0001.
Known leaks in the published evaluation (see notes/research): the embedding is fitted on every round before the split, the window
spans future ticks, the split is random by round. This module fixes the first and keeps the rest as the published design.

Modes (the same tokens and the same training code; only the rounds used for fitting differ):
  honest  fit on TRAIN rounds only (frozen manifest). Test and validation rounds never enter the vocabulary or the training.
  leaky   fit on ALL de_mirage rounds (train, validation and test), as the published evaluation effectively does.
  random  honest vocabulary, embeddings never trained (negative control: random 8-dimensional vectors per token).

Per-second features are a static lookup of the second's own token (causal: no future seconds at inference).
Tokens not in the fitted vocabulary get the zero vector (UNK).

  ...torch env python.exe -I -X utf8 -u scripts\\model\\word2vec_replication.py --mode honest
Writes data/diagnostics/w2v_de_mirage_<mode>.parquet (match_key, round, second, w2v0..w2v7).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

ROOT = r"G:\pureskill\csds-cs2"
DIAG = os.path.join(ROOT, "data", "diagnostics")
DIM, WINDOW, NEG, EPOCHS, ALPHA0, ALPHA1, BATCH = 8, 5, 5, 5, 0.025, 0.0001, 8192
SKIP = 3  # words every 3 seconds from the round start, as in the repository


def build_tokens(map_name="de_mirage"):
    cache = os.path.join(DIAG, "w2v_tokens_de_mirage.parquet")
    if os.path.exists(cache):
        return pl.read_parquet(cache)
    mk = pl.read_parquet(os.path.join(ROOT, "data", "combined", "matches.parquet"), columns=["match_key", "map_name"])
    keys = mk.filter(pl.col("map_name") == map_name)["match_key"].to_list()
    parts = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "2026-*.parquet"))):
        p = pl.read_parquet(f, columns=["match_key", "round", "second", "side", "player_id", "x", "y"]).filter(
            pl.col("match_key").is_in(keys) & pl.col("side").is_in([2, 3]))
        if p.height == 0:
            continue
        p = p.with_columns(pl.col("player_id").rank("ordinal").over(["match_key", "round", "second", "side"]).alias("rk")).filter(pl.col("rk") <= 5)
        p = p.with_columns(pl.when(pl.col("side") == 3).then(pl.col("rk") - 1).otherwise(pl.col("rk") + 4).alias("slot"))
        p = p.with_columns(pl.format("{},{}", (pl.col("x") / 256).round(0).cast(pl.Int32), (pl.col("y") / 256).round(0).cast(pl.Int32)).alias("cell"))
        parts.append(p.select("match_key", "round", "second", "slot", "cell"))
    P = pl.concat(parts)
    wide = P.pivot(on="slot", index=["match_key", "round", "second"], values="cell", aggregate_function="first")
    cols = [str(i) for i in range(10)]
    for c in cols:
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Utf8).alias(c))
    tok = wide.with_columns(pl.concat_str([pl.col(c).fill_null("_") for c in cols], separator="|").alias("token"))
    tok = tok.select("match_key", "round", "second", "token")
    tok.write_parquet(cache, compression="zstd")
    return tok


def frozen_sets(keys):
    man = json.load(open(os.path.join(ROOT, "data", "combined", "split_manifest.json"), encoding="utf-8"))["assign"]
    return {k: man.get(k) for k in keys}


def sentences(tok, round_keys):
    """One sentence per (match, round): tokens every SKIP seconds from the first second of the round."""
    d = tok.filter(pl.col("match_key").is_in(list({k[0] for k in round_keys})))
    d = d.with_columns(pl.col("second").min().over(["match_key", "round"]).alias("s0")).filter(((pl.col("second") - pl.col("s0")) % SKIP) == 0)
    d = d.sort(["match_key", "round", "second"])
    out = []
    for (mk, rd), g in d.group_by(["match_key", "round"], maintain_order=True):
        if (mk, rd) in round_keys:
            out.append(g["token"].to_list())
    return out


def cbow_pairs(sents, vocab, rng):
    """Dynamic window (1..WINDOW each side, as word2vec.c): centres and padded context lists."""
    centers, ctxs, masks = [], [], []
    for s in sents:
        ids = np.array([vocab.get(t, 0) for t in s], dtype=np.int64)
        n = len(ids)
        for i in range(n):
            w = rng.integers(1, WINDOW + 1)
            lo, hi = max(0, i - w), min(n, i + w + 1)
            c = [ids[j] for j in range(lo, hi) if j != i]
            if not c:
                continue
            row = np.zeros(2 * WINDOW, dtype=np.int64)
            m = np.zeros(2 * WINDOW, dtype=np.float32)
            row[:len(c)] = c
            m[:len(c)] = 1.0
            centers.append(ids[i]); ctxs.append(row); masks.append(m)
    return np.array(centers), np.stack(ctxs), np.stack(masks)


def train_cbow(sents, vocab_size, vocab, dev, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    # unigram^0.75 noise distribution from the training corpus
    counts = np.zeros(vocab_size, dtype=np.float64)
    for s in sents:
        for t in s:
            counts[vocab[t]] += 1
    noise = torch.as_tensor(counts ** 0.75, dtype=torch.float32, device=dev)
    W_in = torch.nn.Parameter((torch.rand(vocab_size, DIM, device=dev) - 0.5) / DIM)
    W_out = torch.nn.Parameter(torch.zeros(vocab_size, DIM, device=dev))
    W_in.data[0] = 0.0  # UNK stays zero
    opt = torch.optim.SGD([W_in, W_out], lr=ALPHA0)
    total_steps, step = None, 0
    for ep in range(EPOCHS):
        cen, ctx, msk = cbow_pairs(sents, vocab, rng)
        n = len(cen)
        if total_steps is None:
            total_steps = EPOCHS * ((n + BATCH - 1) // BATCH)
        perm = rng.permutation(n)
        loss_sum = 0.0
        for s0 in range(0, n, BATCH):
            idx = perm[s0:s0 + BATCH]
            c = torch.as_tensor(cen[idx], device=dev)
            x = torch.as_tensor(ctx[idx], device=dev)
            m = torch.as_tensor(msk[idx], device=dev)
            h = (W_in[x] * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
            neg = torch.multinomial(noise, len(idx) * NEG, replacement=True).view(len(idx), NEG)
            pos_score = (h * W_out[c]).sum(-1)
            neg_score = torch.bmm(W_out[neg], h[..., None]).squeeze(-1)
            # summed, not averaged: word2vec updates once per pair, and averaging over a batch shrinks each row's update to nothing
            loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1)).sum()
            lr = max(ALPHA1, ALPHA0 * (1 - step / total_steps))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                W_in[0] = 0.0
            step += 1
            loss_sum += float(loss.item())
        print(f"  epoch {ep + 1}/{EPOCHS}: pairs {n:,}, mean loss per pair {loss_sum / max(1, n):.4f}, lr {lr:.5f}", flush=True)
    return W_in.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["honest", "leaky", "random"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = build_tokens()
    print(f"tokens: {tok.height:,} de_mirage seconds ({time.time() - t0:.0f}s)", flush=True)
    keys = tok.select("match_key", "round").unique()
    sets = {}
    man = json.load(open(os.path.join(ROOT, "data", "combined", "split_manifest.json"), encoding="utf-8"))["assign"]
    for mk, rd in keys.iter_rows():
        sets[(mk, rd)] = man.get(mk)
    if a.mode == "leaky":
        fit_rounds = set(sets.keys())
    else:
        fit_rounds = {k for k, s in sets.items() if s == "train"}
    sents = sentences(tok, fit_rounds)
    vocab_tokens = sorted({t for s in sents for t in s})
    vocab = {t: i + 1 for i, t in enumerate(vocab_tokens)}  # 0 = UNK
    print(f"mode {a.mode}: fit rounds {len(fit_rounds):,}, sentences {len(sents):,}, vocabulary {len(vocab):,}", flush=True)
    if a.mode == "random":
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        W = ((torch.rand(len(vocab) + 1, DIM, generator=g) - 0.5) / DIM).numpy()
        W[0] = 0.0
    else:
        W = train_cbow(sents, len(vocab) + 1, vocab, dev, a.seed)
    vdf = pl.DataFrame({"token": list(vocab.keys()), "tid": list(vocab.values())}, schema={"token": pl.Utf8, "tid": pl.Int64})
    out_tok = tok.join(vdf, on="token", how="left").with_columns(pl.col("tid").fill_null(0))
    ids = out_tok["tid"].to_numpy()
    E = W[ids].astype(np.float32)
    res = out_tok.select("match_key", "round", "second").with_columns(
        [pl.Series(f"w2v{j}", E[:, j]) for j in range(DIM)])
    os.makedirs(DIAG, exist_ok=True)
    out = os.path.join(DIAG, f"w2v_de_mirage_{a.mode}.parquet")
    res.write_parquet(out, compression="zstd")
    unk = float((ids == 0).mean())
    print(json.dumps({"mode": a.mode, "rows": res.height, "unk_share": unk, "vocab": len(vocab), "seconds": round(time.time() - t0)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
