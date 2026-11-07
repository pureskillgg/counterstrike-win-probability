"""D3-4 verifier core (2024 tome, day 12..25). Independent rebuild of:
  (1) swap-signature counts under the claim rule and under a stricter rule (sum==1 and a negative component),
  (2) post-swap disagreement of round_end winner_team_code vs the freeze-score delta, by offset and absolute round,
  (3) bomb-event agreement (bomb_exploded/bomb_defused) vs winner_team_code and vs score delta,
  (4) header-final-score reconstruction of starters' score under candidate swap schedules.
Read-only. Lazy-ish: narrow column projection, filtered to day>=12 keys. Output to stdout; intermediate CSV to scratchpad.
ASSUMPTIONS: winner_team_code 2=T 3=CT (verified via win_reason_message table); freeze_end first-by-tick is the round start;
swap after round s means rounds >s are on swapped sides; header t_starters_* = score of team that started on T.
"""
import glob, sys, collections, itertools
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(240)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
def folder(ch):
    return TOME / "header" if ch == "header" else TOME / f"basic_{ch}{SUF}"
def read(ch, cols=None):
    fs = sorted(glob.glob(str(folder(ch) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")

H = read("header", ["key", "map_name", "t_starters_score_final", "ct_starters_score_final"]).with_columns(
    pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day"))
H = H.filter(pl.col("day") >= 12)
keys = H["key"].to_list(); keyset = set(keys)
hdr = {k: (mp, int(t), int(c), int(d)) for k, mp, t, c, d in H.select(["key", "map_name", "t_starters_score_final", "ct_starters_score_final", "day"]).iter_rows()}
print("day12-25 header matches:", len(keys))

RE = read("round_end", ["match_key", "round", "winner_team_code", "win_reason_message"]).unique(["match_key", "round"], keep="first")
RE = RE.filter(pl.col("match_key").is_in(keys))
RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
RS = RS.filter(pl.col("match_key").is_in(keys))
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick").group_by(["match_key", "round"])
      .agg(pl.col("t_score").first(), pl.col("ct_score").first(), pl.col("t_score_raw").first(), pl.col("ct_score_raw").first()))
BOMBX = set(RS.filter(pl.col("event_type") == "bomb_exploded").select(["match_key", "round"]).iter_rows())
BOMBD = set(RS.filter(pl.col("event_type") == "bomb_defused").select(["match_key", "round"]).iter_rows())
del RS
re = {(m, r): (w, msg) for m, r, w, msg in RE.select(["match_key", "round", "winner_team_code", "win_reason_message"]).iter_rows()}
fe = {(m, r): (t, c, tr, cr) for m, r, t, c, tr, cr in FE.select(["match_key", "round", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).iter_rows()}
print("round_end (m,r):", len(re), "| freeze_end (m,r):", len(fe), "| bomb_exploded (m,r):", len(BOMBX), "| bomb_defused (m,r):", len(BOMBD))
print("matches with any freeze row:", len({m for m, _ in fe}), "| with any round_end:", len({m for m, _ in re}))

def delta(m, r, raw=False):
    if (m, r) in fe and (m, r + 1) in fe:
        a, b = fe[(m, r)], fe[(m, r + 1)]
        return (b[2] - a[2], b[3] - a[3]) if raw else (b[0] - a[0], b[1] - a[1])
    return None
def claim_swap(dt, dc):
    if (dc == 1 and dt == 0) or (dt == 1 and dc == 0): return False
    if dt == 0 and dc == 0: return False
    return dt * dc <= 0
def strict_swap(dt, dc):
    return dt + dc == 1 and min(dt, dc) < 0
def cat(dt, dc):
    if (dt, dc) in ((1, 0), (0, 1)): return "clean"
    if strict_swap(dt, dc): return "strict_swap"
    if (dt, dc) == (0, 0): return "zero"
    return f"other_sum{dt + dc}"

mrounds = collections.defaultdict(set)
for (m, r) in re: mrounds[m].add(r)
for (m, r) in fe: mrounds[m].add(r)
per_match = {}
cat_counts = collections.Counter()
claim_cat = collections.Counter()
sig_claim = collections.Counter(); sig_strict = collections.Counter(); sig_strict_raw = collections.Counter()
for m in keys:
    rs = sorted(r for r in mrounds.get(m, ()) if (m, r) in fe)
    S_c, S_s, S_sr = [], [], []
    for r in rs:
        d = delta(m, r)
        if d is not None:
            dt, dc = d
            cat_counts[cat(dt, dc)] += 1
            if claim_swap(dt, dc):
                claim_cat[cat(dt, dc)] += 1
                S_c.append(r)
            if strict_swap(dt, dc): S_s.append(r)
        dr = delta(m, r, raw=True)
        if dr is not None and strict_swap(*dr): S_sr.append(r)
    per_match[m] = dict(S_claim=tuple(S_c), S_strict=tuple(S_s), S_strict_raw=tuple(S_sr))
    sig_claim[tuple(S_c)] += 1; sig_strict[tuple(S_s)] += 1; sig_strict_raw[tuple(S_sr)] += 1
print("\n[1] all consecutive freeze-delta categories (day12-25):", cat_counts.most_common())
print("[1b] categories among claim-rule swap transitions:", claim_cat.most_common())
print("\n[1c] swap signatures, CLAIM rule (dt*dc<=0, not (0,0)):", sig_claim.most_common(8))
print("[1d] swap signatures, STRICT rule (sum==1 & a negative component), t_score:", sig_strict.most_common(8))
print("[1e] swap signatures, STRICT rule on RAW score fields:", sig_strict_raw.most_common(8))
agree_sig = sum(1 for m in keys if per_match[m]["S_claim"] == per_match[m]["S_strict"])
print("matches where claim-rule signature == strict signature:", agree_sig, "of", len(keys))

# (2) post-swap agreement: winner_team_code vs delta side, for clean deltas only
def delta_side(d):
    if d == (0, 1): return 3
    if d == (1, 0): return 2
    return None
off_tab = collections.Counter(); off_dis = collections.Counter()
abs_tab = collections.defaultdict(lambda: [0, 0])  # (group_key, r) -> [agree, total] for clean rounds
for m in keys:
    S = per_match[m]["S_strict"]
    for r in sorted(r for r in mrounds.get(m, ()) if (m, r) in re):
        d = delta(m, r)
        if d is None: continue
        side = delta_side(d)
        if side is None: continue
        w = re[(m, r)][0]
        prev = [s for s in S if s < r]
        k = (r - max(prev)) if prev else 0
        off_tab[k] += 1
        if side != w: off_dis[k] += 1
        grp = str(S) if len(S) <= 2 else "many"
        abs_tab[(grp, r)][1] += 1
        if side == w: abs_tab[(grp, r)][0] += 1
print("\n[2] clean-delta rounds, disagreement rate by offset k = r - (last strict swap before r); k=0 means no prior swap:")
for k in sorted(off_tab):
    print(f"  k={k:>2}  rounds={off_tab[k]:>7}  disagree={off_dis[k]:>6}  rate={off_dis[k]/off_tab[k]:.4f}")
print("\n[2b] disagreement by group (strict swap tuple) and absolute round r (rounds 1..30), clean deltas:")
for grp in sorted({g for g, _ in abs_tab}, key=lambda s: (len(s), s)):
    if grp not in ("()", "(12,)", "(8,)", "(27,)", "(12, 27)"): continue
    row = []
    for r in range(1, 31):
        a, t = abs_tab.get((grp, r), [0, 0])
        if t: row.append(f"{r}:{t - a}/{t}")
    print(f"  {grp:<10} " + " ".join(row))
