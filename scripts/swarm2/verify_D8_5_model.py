"""verify_D8_5_model: test the alternative explanation for D8-5 (2024 tome).

Alternative (H-PH): the non-round-13 misfit is the same winner-code defect as the round-13 defect, located at the
first round after the halftime side swap. Under a swap after round h (h = detected D8 signature, default 12 when
no swap is detected), the first post-halftime round is h+1. H-PH says: flipping round_end's winner for round h+1
makes the header totals reconcile exactly. The claim's F-3 test only flipped round 13, so it could not see this.

Checks (all against the scratchpad rounds.parquet; appends to logs\\swarm2\\verify_D8_5.log):
  M-1 fit counts by class and signature: no flip / flip round h+1 / flip round 13 (MR12).
  M-2 other_misfit: H-PH fit rate by signature, and the () cases (35) examined individually.
  M-3 independent bomb check: round-h+1 bomb outcome vs round_end winner in detected-(8,) matches, with
      control rounds (round 13 in (12,) matches; round 9 in (12,) matches).
  M-4 round_state step census around the first post-halftime round (anomaly rate at step h+1).
  M-5 the five complete (8,) matches that fit under MR12 with no flip (is the (8,) detection real there?).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_model.py
"""
import collections
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
LOG = ROOT / "logs" / "swarm2" / "verify_D8_5.log"
OUT = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D8_5")
T, CT = 2, 3
FOUR = ["de_inferno", "de_vertigo", "de_overpass", "de_nuke"]
LOGF = open(LOG, "a", encoding="utf-8", newline="\n")


def say(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    LOGF.write(s + "\n")
    LOGF.flush()


R = pl.read_parquet(OUT / "rounds.parquet").sort(["match_key", "round"])
MAT = {}
for x in R.iter_rows(named=True):
    m = MAT.get(x["match_key"])
    if m is None:
        m = MAT[x["match_key"]] = {"key": x["match_key"], "map": x["map_name"], "tA": int(x["tA"]),
                                   "tB": int(x["tB"]), "day": int(x["day"]), "rd": []}
        MAT[x["match_key"]] = m
    m["rd"].append(x)
for m in MAT.values():
    rd = m["rd"]
    n = len(rd)
    m["n"] = n
    m["contig"] = [x["round"] for x in rd] == list(range(1, n + 1))
    m["complete"] = (n == m["tA"] + m["tB"]) and n <= 24 and m["contig"]
    m["dg"] = "d01-07" if m["day"] <= 7 else "d12-25"


def d8_sig(m):
    live = {x["round"]: (x["tb"], x["cb"]) for x in m["rd"] if x["tb"] is not None}
    sw = []
    for r in sorted(live):
        if r + 1 not in live:
            continue
        t1, c1 = live[r]
        t2, c2 = live[r + 1]
        dt, dc = t2 - t1, c2 - c1
        if (dc == 1 and dt == 0) or (dt == 1 and dc == 0) or (dt == 0 and dc == 0):
            continue
        if dt * dc <= 0:
            sw.append(r)
    return tuple(sw)


def d4_cls(m):
    rd, n, tA, tB = m["rd"], m["n"], m["tA"], m["tB"]
    mm13 = any(x["round"] == 13 and x["bside"] is not None and x["bside"] != x["winner"] for x in rd)
    wa = sum(1 for x in rd if x["winner"] == (T if x["round"] <= 12 else CT))
    misfit = (wa, n - wa) != (tA, tB)
    r13 = next((x for x in rd if x["round"] == 13), None)
    flip = False
    if misfit and r13 is not None:
        nwa = wa - 1 if r13["winner"] == CT else wa + 1
        flip = (nwa, n - nwa) == (tA, tB)
    if mm13:
        return "affected"
    if flip:
        return "r13_flip_misfit"
    if misfit:
        return "other_misfit"
    return "fits"


def wins_A(rd, swaps, flip_r=None):
    wa = 0
    for x in rd:
        k = sum(1 for s in swaps if s < x["round"])
        side_a = T if k % 2 == 0 else CT
        won = x["winner"] == side_a
        if flip_r is not None and x["round"] == flip_r:
            won = not won
        wa += int(won)
    return wa


def fit(m, swaps, flip_r=None):
    wa = wins_A(m["rd"], swaps, flip_r)
    return (wa, m["n"] - wa) == (m["tA"], m["tB"])


def steps(m):
    rd = m["rd"]
    out = []
    for x, y in zip(rd[:-1], rd[1:]):
        if y["round"] != x["round"] + 1:
            out.append((x["round"], "gap"))
            continue
        if None in (x["tb"], x["cb"], y["tb"], y["cb"]):
            out.append((x["round"], "missing"))
            continue
        tT = 1 if x["winner"] == T else 0
        tC = 1 - tT
        ns = (y["tb"] == x["tb"] + tT) and (y["cb"] == x["cb"] + tC)
        sw = (y["tb"] == x["cb"] + tC) and (y["cb"] == x["tb"] + tT)
        out.append((x["round"], "both" if (ns and sw) else "noswap" if ns else "swap" if sw else "anomaly"))
    return out


COMP = [m for m in MAT.values() if m["complete"]]
for m in MAT.values():
    m["sig"] = d8_sig(m)
for m in COMP:
    m["cls"] = d4_cls(m)
    m["h"] = m["sig"][0] if len(m["sig"]) == 1 else (12 if len(m["sig"]) == 0 else None)
    if m["h"] is not None:
        m["hph"] = fit(m, (m["h"],), flip_r=m["h"] + 1)
        m["hph0"] = fit(m, (m["h"],))
    else:
        m["hph"] = m["hph0"] = None
    m["mr12_f13"] = fit(m, (12,), flip_r=13) if any(x["round"] == 13 for x in m["rd"]) else False
    m["mr12_0"] = fit(m, (12,))

say("\n=== verify_D8_5 MODEL CHECKS (appended) ===")
say("H-PH: non-round-13 misfit = winner-code inversion at the first post-halftime round (h+1), h = D8 signature")
say("        (h = 12 for complete matches with no D8 swap, ASSUMPTION; one match-level h only)")

say("\n--- M-1 complete matches: fit counts by class and signature")
rows = [{"cls": m["cls"], "sig": str(m["sig"]), "mr12_0": m["mr12_0"], "mr12_flip13": m["mr12_f13"],
         "hph_noflip": m["hph0"], "hph_flip_h1": m["hph"], "map": m["map"]} for m in COMP]
Q = pl.DataFrame(rows)
say(Q.group_by(["cls", "sig"]).agg(pl.len().alias("n"), pl.col("mr12_0").sum().alias("MR12_fit"),
                                    pl.col("mr12_flip13").sum().alias("MR12+flip13"),
                                    pl.col("hph_noflip").sum().alias("h_fit_noflip"),
                                    pl.col("hph_flip_h1").sum().alias("h+flip_h+1")).sort(["cls", "sig"]))
say("H-PH fit (swap at h, flip round h+1) over all complete matches:",
    f"{sum(1 for m in COMP if m['hph'])} of {len(COMP)}")
say("H-PH fit over complete matches by class:",
    Q.group_by("cls").agg(pl.len().alias("n"), pl.col("hph_flip_h1").sum().alias("fit")).sort("cls").rows())

OM = [m for m in COMP if m["cls"] == "other_misfit"]
say(f"\n--- M-2 other_misfit ({len(OM)}): H-PH fit rate by signature and map")
say(collections.Counter((str(m["sig"]), m["hph"]) for m in OM))
say(collections.Counter((m["map"], str(m["sig"]), m["hph"]) for m in OM).most_common())
om0 = [m for m in OM if m["sig"] == ()]
say(f"\n    the {len(om0)} other_misfit with no D8 swap: anomaly rounds and cover by any single swap h with flip h+1:")
for m in om0:
    st = steps(m)
    anom = [r for r, t in st if t == "anomaly"]
    both = [r for r, t in st if t == "both"]
    swap = [r for r, t in st if t == "swap"]
    alt = [h for h in range(1, m["n"]) if fit(m, (h,), flip_r=h + 1)]
    m["alt_h"] = alt
    say(f"    {m['map']:<12} n={m['n']:>2} tA={m['tA']:>2} tB={m['tB']:>2} anomaly={anom} swap={swap} "
        f"both={len(both)} H-PH_any_h={alt}")
say(f"    () other_misfit with an H-PH fit for some h (flip h+1): {sum(1 for m in om0 if m['alt_h'])} of {len(om0)}")
say(f"    () other_misfit with an H-PH fit at h=12: {sum(1 for m in om0 if m['hph'])} of {len(om0)}")

# ----------------------------------------------------------------- M-3 bomb check, all matches (not only complete)
say("\n--- M-3 independent bomb-outcome check (rounds with a bomb outcome; mismatch = bside != round_end winner)")
D12 = [m for m in MAT.values() if m["dg"] == "d12-25"]
s8 = [m for m in D12 if m["sig"] == (8,)]
s12 = [m for m in D12 if m["sig"] == (12,)]
s0 = [m for m in D12 if m["sig"] == ()]


def bomb_tab(group, rnd, label):
    n_ = mm_ = 0
    for m in group:
        for x in m["rd"]:
            if x["round"] == rnd and x["bside"] is not None:
                n_ += 1
                mm_ += int(x["bside"] != x["winner"])
    say(f"  {label:<55} rows={n_:>4} mismatches={mm_:>4} rate={mm_ / n_ if n_ else float('nan'):.3f}")


bomb_tab(s8, 9, "(8,) matches, round 9 (first post-halftime round)")
bomb_tab(s8, 13, "(8,) matches, round 13 (control)")
bomb_tab(s8, 8, "(8,) matches, round 8 (last pre-halftime round, control)")
bomb_tab(s12, 13, "(12,) matches, round 13 (first post-halftime, MR12)")
bomb_tab(s12, 9, "(12,) matches, round 9 (control)")
bomb_tab(s0, 13, "() matches (no D8 swap), round 13")
bomb_tab(s0, 9, "() matches, round 9")
bomb_tab([m for m in D12 if m["map"] not in FOUR], 13, "other maps, all sig, round 13")
bomb_tab([m for m in D12 if m["map"] not in FOUR], 9, "other maps, all sig, round 9")

# ----------------------------------------------------------------- M-4 step census at first post-halftime round
say("\n--- M-4 round_state step census (winner-consistent), all matches days 12-25, step at h+1 vs control steps")


def step_counts(group, step_round, label):
    c = collections.Counter()
    for m in group:
        if not m["contig"]:
            continue
        for r, t in steps(m):
            if r == step_round:
                c[t] += 1
    tot = sum(c.values())
    say(f"  {label:<48} n={tot:>4} " + " ".join(f"{k}={v}" for k, v in sorted(c.items())) +
        (f"  anomaly rate={c['anomaly'] / tot:.3f}" if tot else ""))


step_counts(s8, 9, "(8,) matches, step after round 9")
step_counts(s8, 13, "(8,) matches, step after round 13 (control)")
step_counts(s12, 13, "(12,) matches, step after round 13")
step_counts(s12, 9, "(12,) matches, step after round 9 (control)")
step_counts(s0, 13, "() matches, step after round 13")

# ----------------------------------------------------------------- M-5 the five (8,) fits under MR12
say("\n--- M-5 complete (8,) matches that fit under MR12 with no flip (5 expected from F-3):")
five = [m for m in COMP if m["sig"] == (8,) and m["mr12_0"]]
for m in five:
    st = steps(m)
    say(f"  {m['key']} map={m['map']} n={m['n']} tA={m['tA']} tB={m['tB']} cls={m['cls']} "
        f"steps={[(r, t) for r, t in st if t != 'noswap']}")
say(f"  count={len(five)}; their H-PH fit: {[m['hph'] for m in five]}; MR12 fit (no flip): {[m['mr12_0'] for m in five]}")

say("\nM-DONE")
LOGF.close()
print("MODEL DONE")
