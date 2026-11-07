"""verify D5-9 main: independent recomputation of the round-9 (8->9 side swap, MR8-type halftime) claim.

Input: scratchpad tables written by verify_D5_9_build.py (C:, not the project).
Nothing is written to the project or to E:.

Conventions (checked, not assumed, where possible):
  winner_team_code: 2 = T, 3 = CT (bomb_exploded -> 2, bomb_defused -> 3).
  freeze-end t_score / ct_score: side-attached, so a side swap at boundary r->r+1 maps the fields crosswise.
  Boundary classification uses the D5 rule with winner w_r from round_end (ASSUMPTION: w_r is correct
  for the pre-halftime round; the claim itself makes that assumption).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_main.py
"""
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\VD59")

H = pl.read_parquet(SCR / "header.parquet")
FE = pl.read_parquet(SCR / "fe.parquet")
MK = pl.read_parquet(SCR / "marks.parquet")
REr = pl.read_parquet(SCR / "end_raw.parquet")
BS = pl.read_parquet(SCR / "bomb.parquet")
DE = pl.read_parquet(SCR / "deaths.parquet")

print("=== 0. inputs")
print("header matches:", H.height, " regimes/markers rows:", MK.height)

# ---------- per-match regime from round_state marker ----------
ann = MK.filter(pl.col("event_type") == "round_announce_match_start").select(["match_key", "n"])
print("matches with round_announce_match_start:", ann.height, " total marker rows:", int(ann["n"].sum()))
regA = set(ann["match_key"].to_list())

# ---------- round_end per (match, round): first row by tick; duplicates counted ----------
RE = (REr.sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"], maintain_order=True).agg(
          pl.col("winner_team_code").first().alias("w"),
          pl.col("winner_team_code").n_unique().alias("w_nd"),
          pl.col("tick").first().alias("end_tick"),
          pl.col("win_reason_code").first().alias("reason"),
          pl.col("player_count").first().alias("pc"),
          pl.len().alias("end_n")))
print("round_end (match,round) pairs:", RE.height, " with >1 row:", int((RE["end_n"] > 1).sum()),
      " with >1 distinct winner:", int((RE["w_nd"] > 1).sum()))
print("round_end player_count distribution:", RE.group_by("pc").len().sort("len", descending=True).head(8).to_dicts())

# ---------- bomb outcome per (match, round) ----------
BO_ev = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key", "round"]).agg(
    pl.col("event_type").unique().alias("evs"), pl.len().alias("n_bo"))
BO = BO_ev.with_columns(
    has_exp=pl.col("evs").list.contains("bomb_exploded"),
    has_def=pl.col("evs").list.contains("bomb_defused")).with_columns(
    bo_side=pl.when(pl.col("has_exp") & ~pl.col("has_def")).then(2)
    .when(pl.col("has_def") & ~pl.col("has_exp")).then(3).otherwise(None)).select(
    ["match_key", "round", "bo_side", "n_bo"])
PL = BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(
    pl.len().alias("n_pl"), pl.col("player_team_code").first().alias("pl_team"))
print("bomb planted-team codes:", PL.group_by("pl_team").len().to_dicts())
print("bomb outcome (side) distribution:", BO.group_by("bo_side").len().to_dicts(),
      " rounds with >1 outcome row:", int((BO["n_bo"] > 1).sum()))

# ---------- last victim per (match, round) ----------
DEs = DE.sort(["match_key", "round", "tick"])
LVt = DEs.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("lv_tick"))
LV = (DEs.join(LVt, on=["match_key", "round"])
      .filter(pl.col("tick") == pl.col("lv_tick"))
      .group_by(["match_key", "round"]).agg(
          pl.col("player_team_code").n_unique().alias("lv_nteam"),
          pl.col("player_team_code").max().alias("lv_victim"),
          pl.len().alias("lv_n")))

# ---------- freeze-end ----------
FEc = FE.select(["match_key", "round", "fe_t", "fe_ct", "fe_tick", "fe_n"])

# ---------- combined per-round table (full outer on freeze-end and round_end) ----------
R = (FEc.join(RE, on=["match_key", "round"], how="full", coalesce=True)
     .join(BO, on=["match_key", "round"], how="left")
     .join(PL, on=["match_key", "round"], how="left")
     .join(LV, on=["match_key", "round"], how="left"))
print("\ncombined round rows:", R.height)

# convert to dict[match][round]
rounds = defaultdict(dict)
for row in R.iter_rows(named=True):
    rounds[row["match_key"]][row["round"]] = row

# match-level markers
ann_last_half = MK.filter(pl.col("event_type") == "round_announce_last_round_half")
mp = MK.filter(pl.col("event_type") == "round_announce_match_point")
ann_last_half_lab = {r["match_key"]: r["min_round"] for r in ann_last_half.iter_rows(named=True)}
mp_rounds = {r["match_key"]: r["rounds"] for r in mp.iter_rows(named=True)}
tick_rate = {r["match_key"]: r["tick_rate"] for r in H.iter_rows(named=True)}
mapname = {r["match_key"]: r["map_name"] for r in H.iter_rows(named=True)}
day = {r["match_key"]: r["day"] for r in H.iter_rows(named=True)}
match_type = {r["match_key"]: r["match_type"] for r in H.iter_rows(named=True)}


def g(mk, r):
    return rounds.get(mk, {}).get(r)


def classify(t, ct, t2, ct2, w):
    """D5 rule. ns: fields follow team; sw: fields follow side (crosswise)."""
    if None in (t, ct, t2, ct2, w) or w not in (2, 3):
        return "na"
    a = 1 if w == 2 else 0
    b = 1 if w == 3 else 0
    ns = (t2 == t + a) and (ct2 == ct + b)
    sw = (t2 == ct + b) and (ct2 == t + a)
    if ns and sw:
        return "ambig"
    if ns:
        return "ns"
    if sw:
        return "sw"
    return "other"


def step_truth_ns(t, ct, t2, ct2):
    """No-swap step truth: T field +1 -> T (2); CT field +1 -> CT (3); otherwise None."""
    if None in (t, ct, t2, ct2):
        return None
    dt, dc = t2 - t, ct2 - ct
    if (dt, dc) == (1, 0):
        return 2
    if (dt, dc) == (0, 1):
        return 3
    return None


def boundary(mk, r):
    """Classification of boundary r->r+1 using round_end winner w_r."""
    a_, b_ = g(mk, r), g(mk, r + 1)
    if not a_ or not b_:
        return "absent"
    return classify(a_.get("fe_t"), a_.get("fe_ct"), b_.get("fe_t"), b_.get("fe_ct"), a_.get("w"))


# ---------- 1. swap classification at 8->9 and 12->13 ----------
print("\n=== 1. boundary classification (D5 rule, winner from round_end)")
cls8 = {mk: boundary(mk, 8) for mk in rounds}
cls12 = {mk: boundary(mk, 12) for mk in rounds}
for name, cl in [("8->9", cls8), ("12->13", cls12)]:
    for reg in ["A", "B"]:
        c = Counter(v for mk, v in cl.items() if ((mk in regA) == (reg == "A")))
        print(f"  boundary {name} regime {reg}: {dict(sorted(c.items()))}")

sw8 = {mk for mk, v in cls8.items() if v == "sw"}
sw12 = {mk for mk, v in cls12.items() if v == "sw"}
print("\nmatches with sw at 8->9 (all):", len(sw8), " regime A:", len(sw8 & regA), " regime B:", len(sw8 - regA))
print("matches with sw at 12->13 (all):", len(sw12), " regime A:", len(sw12 & regA), " regime B:", len(sw12 - regA))
# claim group: sw at 8->9 and not sw at 12->13, where 12->13 is observable
has13 = {mk for mk in rounds if g(mk, 13) is not None}
grp_claim = {mk for mk in sw8 if mk not in sw12}
print("claim group (sw 8->9, no sw 12->13): total", len(grp_claim),
      " regime A", len(grp_claim & regA), " regime B", len(grp_claim - regA))
print("   of which 12->13 observable (round 13 present):", len(grp_claim & has13),
      "  classified ns at 12->13:", sum(1 for mk in grp_claim if cls12[mk] == "ns"),
      "  absent:", sum(1 for mk in grp_claim if cls12[mk] == "absent"),
      "  other/ambig/na:", sum(1 for mk in grp_claim if cls12[mk] not in ("ns", "absent")))
print("   claim-group by map (top):", Counter(mapname[mk] for mk in grp_claim).most_common(10))
print("   claim-group by day:", sorted(Counter(day[mk] for mk in grp_claim).items()))
print("   claim-group by match_type:", Counter(match_type[mk] for mk in grp_claim).most_common())

# ---------- 2. format evidence ----------
print("\n=== 2. format evidence (per group)")


def last_label(mk):
    rs = [r for r, v in rounds[mk].items() if v.get("w") is not None or v.get("fe_t") is not None]
    return max(rs) if rs else None


def final_scores(mk):
    N = last_label(mk)
    if N is None:
        return None
    row = g(mk, N)
    if not row or row.get("fe_t") is None or row.get("w") not in (2, 3):
        return None
    t = row["fe_t"] + (1 if row["w"] == 2 else 0)
    ct = row["fe_ct"] + (1 if row["w"] == 3 else 0)
    return (N, t, ct)


def summarize(name, mks, use_final=True):
    labs, halfs, mps, maxs, sums_ok = [], [], [], [], []
    for mk in mks:
        N = last_label(mk)
        if N is not None:
            labs.append(N)
        if mk in ann_last_half_lab:
            halfs.append(ann_last_half_lab[mk])
        if mk in mp_rounds:
            mps.append(tuple(sorted(mp_rounds[mk])))
        if use_final:
            fs = final_scores(mk)
            if fs:
                maxs.append(max(fs[1], fs[2]))
    print(f"\n  [{name}] n={len(mks)}")
    if labs:
        print(f"    last round label: min {min(labs)} max {max(labs)} mean {st.mean(labs):.2f} median {st.median(labs)}")
    print("    last-round-half announcement label:", dict(Counter(halfs).most_common(6)),
          " (n with announcement:", len(halfs), ")")
    print("    match-point announcement labels (top):", Counter(mps).most_common(4),
          " (n with announcement:", len(mps), ")")
    if use_final and maxs:
        print("    max final team score from freeze-end + last winner:", dict(sorted(Counter(maxs).items())),
              " (n:", len(maxs), ")")


regB = set(rounds) - regA
claim_A = grp_claim & regA
claim_B = grp_claim - regA
others_A = regA - grp_claim
others_B = regB - grp_claim
summarize("claim group, regime A", claim_A)
summarize("claim group, regime B", claim_B)
summarize("other matches, regime A", others_A)
summarize("other matches, regime B", others_B)

# sum identity: freeze-end t+ct should equal r-1 (one point per completed round)
bad_sum = 0
tot_sum = 0
for mk, rr in rounds.items():
    for r, row in rr.items():
        if row.get("fe_t") is None:
            continue
        tot_sum += 1
        if row["fe_t"] + row["fe_ct"] != r - 1:
            bad_sum += 1
print(f"\n  freeze-end sum identity t+ct == r-1: violations {bad_sum} of {tot_sum} freeze-end rows")

# ---------- 3. round-9 inversion versus independent truths ----------
print("\n=== 3. round r winner (round_end) vs three truths")


def compare_round(mks, r, label):
    stat = Counter()
    for mk in mks:
        row = g(mk, r)
        nxt = g(mk, r + 1)
        if not row or row.get("w") not in (2, 3):
            stat["no_end"] += 1
            continue
        w = row["w"]
        # step truth (no-swap mapping on r->r+1)
        if nxt is not None:
            st_ = step_truth_ns(row.get("fe_t"), row.get("fe_ct"), nxt.get("fe_t"), nxt.get("fe_ct"))
            if st_ is None:
                stat["step_unusable"] += 1
            else:
                stat["step_n"] += 1
                stat["step_agree" if st_ == w else "step_disagree"] += 1
        else:
            stat["step_no_next"] += 1
        # bomb truth
        if row.get("bo_side") is not None:
            stat["bomb_n"] += 1
            stat["bomb_agree" if row["bo_side"] == w else "bomb_disagree"] += 1
        # last-victim truth on no-bomb rounds, elimination-type only (ties excluded)
        if row.get("pl_team") is None and row.get("lv_victim") in (2, 3) and row.get("lv_nteam") == 1:
            lv_truth = 5 - row["lv_victim"]
            stat["lv_n"] += 1
            stat["lv_agree" if lv_truth == w else "lv_disagree"] += 1
    n_step = stat["step_n"]
    n_bomb = stat["bomb_n"]
    n_lv = stat["lv_n"]
    def pct(a, b):
        return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else f"0/0"
    print(f"  {label}: round {r}")
    print(f"    score-step truth: disagree {pct(stat['step_disagree'], n_step)}  usable {n_step}  unusable {stat['step_unusable']}")
    print(f"    bomb truth:       disagree {pct(stat['bomb_disagree'], n_bomb)}  rounds with bomb outcome {n_bomb}")
    print(f"    last-victim truth (no bomb, single victim team): disagree {pct(stat['lv_disagree'], n_lv)}  n {n_lv}")
    return stat


# claim group (MR8-type halftime at 8)
print()
for name, mks in [("claim group regime A (sw 8->9)", claim_A), ("claim group regime B (sw 8->9)", claim_B)]:
    compare_round(mks, 9, name)

# control: MR12-type halftime (sw at 12->13, no sw at 8->9), round 13 is the first post-halftime round
ctrlA_12 = {mk for mk in sw12 if mk not in sw8 and mk in regA}
ctrlB_12 = {mk for mk in sw12 if mk not in sw8 and mk not in regA}
print()
compare_round(ctrlA_12, 13, "MR12 control regime A (sw 12->13)")
compare_round(ctrlB_12, 13, "MR12 control regime B (sw 12->13)")

# control: round 9 in regime A matches with NO halftime-type swap at 8 (should be clean if the defect is halftime-linked)
noswap_A = {mk for mk in regA if cls8.get(mk) == "ns" and cls12.get(mk) == "ns"}
noswap_B = {mk for mk in (set(rounds) - regA) if cls8.get(mk) == "ns" and cls12.get(mk) == "ns"}
print()
compare_round(noswap_A, 9, "no-swap control regime A (ns at 8 and 12)")
compare_round(noswap_B, 9, "no-swap control regime B (ns at 8 and 12)")

# ---------- 4. boundary 9->10 classification (claimed tie artifact) ----------
print("\n=== 4. boundary 9->10 classification in the claim group")
c910 = Counter()
for mk in claim_A | claim_B:
    c910[("A" if mk in regA else "B", boundary(mk, 9))] += 1
print("  boundary 9->10 classes:", dict(sorted(c910.items())))

# ---------- 5. timing: gap between round_end(r) and freeze_end(r+1) ----------
print("\n=== 5. gap round_end(r) -> freeze_end(r+1), seconds (ASSUMPTION: tick rate from header)")
def gaps(mks, r):
    out = []
    for mk in mks:
        a_, b_ = g(mk, r), g(mk, r + 1)
        if a_ and b_ and a_.get("end_tick") is not None and b_.get("fe_tick") is not None:
            tr = tick_rate.get(mk) or 64
            out.append((b_["fe_tick"] - a_["end_tick"]) / tr)
    if not out:
        return "n=0"
    out.sort()
    return f"n={len(out)} median {st.median(out):.1f}s p5 {out[int(0.05*len(out))]:.1f}s p95 {out[int(0.95*len(out))]:.1f}s min {out[0]:.1f}s max {out[-1]:.1f}s"

print("  claim group A, boundary 8->9:", gaps(claim_A, 8))
print("  claim group B, boundary 8->9:", gaps(claim_B, 8))
print("  MR12 control A, boundary 12->13:", gaps(ctrlA_12, 12))
print("  MR12 control B, boundary 12->13:", gaps(ctrlB_12, 12))
print("  no-swap regime A, boundary 8->9:", gaps(noswap_A, 8))
print("  no-swap regime A, boundary 9->10:", gaps(noswap_A, 9))
print("  claim group A, boundary 9->10:", gaps(claim_A, 9))

# ---------- 6. per-match bomb placement at label 9 (alignment of the bomb channel) ----------
print("\n=== 6. bomb rows inside the labelled round window [fe_tick(r), end_tick(r)]")
def bomb_window(mks, r):
    inside = outside = 0
    for mk in mks:
        row = g(mk, r)
        if not row or row.get("bo_side") is None:
            continue
        bt = None
        # bomb tick for this round
        bt = BS.filter((pl.col("match_key") == mk) & (pl.col("round") == r) &
                       pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        if bt.height == 0:
            continue
        t_b = bt["tick"][0]
        if row.get("fe_tick") is not None and row.get("end_tick") is not None and row["fe_tick"] <= t_b <= row["end_tick"]:
            inside += 1
        else:
            outside += 1
    return inside, outside

# (filter per match is slow; sample the claim group only)
sample = sorted(claim_A)[:400]
print("  claim group A round 9 (first 400 matches by key): inside/outside =", bomb_window(sample, 9))
