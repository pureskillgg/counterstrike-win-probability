"""Windows of previous seconds computed per batch, instead of a precomputed N x lookback table.

window(index, rows, L)[i, k] is the row index of the same match's same round at second (second_of_rows[i] - k), for k = 0..L-1
(k = 0 is the current second), or -1 when no such row exists. It returns exactly what
train_temporal_round_model.previous_rows builds for those rows, but the memory is O(N) rather than O(N x L), so a 60-second
window fits next to the feature table on a 20 GB card.
"""
import numpy as np
import torch


def build(match_idx, round_id, second, device):
    key = ((match_idx.astype(np.int64) * 128 + round_id.astype(np.int64)) * 1024) + second.astype(np.int64)
    assert round_id.max() < 128 and second.max() < 1024, "key packing assumes round < 128 and second < 1024"
    k = torch.as_tensor(key, device=device)
    order = torch.sort(k, stable=True).indices
    return {"key": k, "skey": k[order], "order": order}


def window(index, rows, L):
    base = index["key"][rows]
    ks = torch.arange(L, device=rows.device)
    target = base[:, None] - ks[None, :]
    n = index["skey"].numel()
    pos = torch.searchsorted(index["skey"], target)
    pos_c = pos.clamp(max=n - 1)
    found = (pos < n) & (index["skey"][pos_c] == target)
    return torch.where(found, index["order"][pos_c], torch.full_like(pos_c, -1))
