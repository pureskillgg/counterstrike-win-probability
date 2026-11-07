"""verify D7-6 step d: independent physical-side winner check on the 2024-02-g tome.

Sources (raw tome, read with lazy projection; no D7 intermediate used except the D7 ledger column for comparison):
  header            : build_num, match_date (UTC), map_name
  round_end         : winner_team_code (W), win_reason_code, tick, player_count
  round_state       : round_freeze_end t_score / ct_score
  player_info       : per-round player_id + team_code (2=T, 3=CT) -> physical side roster
  player_death      : per-round player_id of deaths
  bomb_state        : bomb_exploded (T wins) / bomb_defused (CT wins)

Derived per (match, round):
  oA_r   : team A (= roster on T in anchor round) is on T in round r (1) or CT (0), from rosters (no score data).
  PWs_r  : winner side from freeze_end score deltas between r and r+1, mapped through oA (side-attached scores).
  PWe_r  : winner side from elimination (one roster side has no deaths, the other has survivors).
  BS_r   : bomb side (2 if exploded, 3 if defused).
  L_r    : D7 DP ledger side (read from logs/swarm2/D7_rounds_with_ledger_dp.parquet, for comparison only).

ASSUMPTIONS (stated, not tuned):
  A1 roster side code 2 = T, 3 = CT, same encoding as winner_team_code (checked by pre-onset agreement).
  A2 freeze_end score fields are side-attached (score of the team currently on T is t_score); tested vs team-attached.
  A3 post period = match_date >= 2024-02-07 00:00:00 UTC (as in D7); onset is reported, not assumed.
  A4 elimination winner needs one roster side with zero deaths and the other with survivors; otherwise undecided.
Outputs: stdout only (captured into logs/swarm2/verify_D7_6.log by the caller).
"""
import collections
import datetime
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def scan(name, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{name}{SUF}" / "dataframe_*")))
    return pl.scan_parquet(fs).select(cols).collect()


# ---------- header
H = (pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num", "map_name"])
       .rename({"key": "match_key"})
       .with_columns(pl.col("match_date").str.slice(0, 19).str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False).alias("mdt")))
hdr = {k: (b, m, mp) for k, b, m, mp in H.select(["match_key", "build_num", "mdt", "map_name"]).iter_rows()}
print("header matches:", len(hdr), "| build counts:", H.group_by("build_num").len().sort("build_num").rows())

# ---------- round_end (W)
RE = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"]) \
    .unique(["match_key", "round"], keep="first")
re_ = {(k, r): (w, rs, t, pc) for k, r, t, w, rs, pc in
       RE.select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"]).iter_rows()}
print("round_end unique (match,round):", len(re_))

# ---------- freeze_end scores
fs = sorted(glob.glob(str(BASE / f"basic_round_state{SUF}" / "dataframe_*")))
FE = (pl.scan_parquet(fs).filter(pl.col("event_type") == "round_freeze_end")
        .select(["match_key", "round", "t_score", "ct_score"]).collect()
        .unique(["match_key", "round"], keep="first"))
fe = {(k, r): (t, c) for k, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
print("freeze_end unique (match,round):", len(fe))

# ---------- rosters: side codes per (match, round, player) from four sources in priority order
#   1 player_info.team_code, 2 player_spawn.player_team_code, 3 player_death victim team code,
#   4 player_death attacker_team_code (attacker_id). First assignment wins; codes 2 (T) / 3 (CT) only.
side = collections.defaultdict(dict)  # (k, r) -> {pid: code}
PI = scan("player_info", ["match_key", "round", "player_id", "team_code"])
for k, r, p, tc in PI.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
    if p is not None and tc in (2, 3):
        side[(k, r)].setdefault(p, tc)
n_pi = sum(len(v) for v in side.values())
PS = scan("player_spawn", ["match_key", "round", "player_id", "player_team_code"])
for k, r, p, tc in PS.select(["match_key", "round", "player_id", "player_team_code"]).iter_rows():
    if p is not None and tc in (2, 3):
        side[(k, r)].setdefault(p, tc)
n_ps = sum(len(v) for v in side.values())
PDX = scan("player_death", ["match_key", "round", "tick", "player_id", "player_team_code", "attacker_id", "attacker_team_code"])
dead = collections.defaultdict(set)
for k, r, t, p, tc, a, atc in PDX.select(["match_key", "round", "tick", "player_id", "player_team_code",
                                          "attacker_id", "attacker_team_code"]).iter_rows():
    if p is not None:
        dead[(k, r)].add(p)
    if p is not None and tc in (2, 3):
        side[(k, r)].setdefault(p, tc)
for k, r, t, p, tc, a, atc in PDX.select(["match_key", "round", "tick", "player_id", "player_team_code",
                                          "attacker_id", "attacker_team_code"]).iter_rows():
    if a is not None and atc in (2, 3):
        side[(k, r)].setdefault(a, atc)
ros = collections.defaultdict(lambda: (set(), set()))
for kr, mp_ in side.items():
    for p, tc in mp_.items():
        if tc == 2:
            ros[kr][0].add(p)
        else:
            ros[kr][1].add(p)
print("roster sources: player_info entries", n_pi, "| +spawn entries", n_ps, "| total entries", sum(len(v) for v in side.values()))
print("roster rounds:", len(ros), "| death rounds:", len(dead))

# ---------- bomb outcome
BSE = scan("bomb_state", ["match_key", "round", "event_type"])
BSE = BSE.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
bomb = collections.defaultdict(set)
for k, r, ev in BSE.select(["match_key", "round", "event_type"]).iter_rows():
    bomb[(k, r)].add(ev)

# ---------- D7 ledger (comparison only)
LD = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_with_ledger_dp.parquet"),
                     columns=["match_key", "round", "ledger_side"]).unique(["match_key", "round"], keep="first")
Ld = {(k, r): s for k, r, s in LD.iter_rows()}
print("D7 ledger rows (non-null):", sum(1 for v in Ld.values() if v is not None))

# ---------- per match orientation and winners
rounds_by_match = collections.defaultdict(set)
for (k, r) in set(fe) | set(ros) | set(re_):
    rounds_by_match[k].add(r)

rows = []
ori_fail = collections.Counter()
for k, rset in rounds_by_match.items():
    rs = sorted(rset)
    b, mdt, mp = hdr.get(k, (None, None, None))
    # anchor: first round with >=2 players on each side (2v2 matches exist in this tome; a >=3 rule
    # left them with no orientation and was a coverage artifact, see round-9 trace)
    A = None
    for r in rs:
        T, C = ros.get((k, r), (set(), set()))
        if len(T) >= 2 and len(C) >= 2:
            A = set(T)
            break
    oA = {}
    for r in rs:
        T, C = ros.get((k, r), (set(), set()))
        if A is None or len(T) == 0:
            oA[r] = None
            ori_fail["no_anchor_or_no_T"] += 1
            continue
        inter = len(T & A) / len(T)
        if inter > 0.5:
            oA[r] = 1
        elif inter < 0.5:
            oA[r] = 0
        else:
            oA[r] = None
            ori_fail["tie"] += 1
    for r in rs:
        if (k, r) not in re_:
            continue
        W, reason, tick, pc = re_[(k, r)]
        o0 = oA.get(r)
        prev = oA.get(r - 1)
        fas_phys = None
        if o0 is not None and prev is not None:
            fas_phys = (o0 != prev)
        # score-based winner (side-attached mapping) and team-attached diagnostic
        pws = None
        pws_status = "no_next_fe"
        alt = None
        if (k, r) in fe and (k, r + 1) in fe and o0 is not None and oA.get(r + 1) is not None:
            t0, c0 = fe[(k, r)]
            t1, c1 = fe[(k, r + 1)]
            o1 = oA[r + 1]
            sA0 = t0 if o0 == 1 else c0
            sB0 = c0 if o0 == 1 else t0
            sA1 = t1 if o1 == 1 else c1
            sB1 = c1 if o1 == 1 else t1
            dA, dB = sA1 - sA0, sB1 - sB0
            if dA == 1 and dB == 0:
                pws = 2 if o0 == 1 else 3
                pws_status = "ok"
            elif dB == 1 and dA == 0:
                pws = 3 if o0 == 1 else 2
                pws_status = "ok"
            else:
                pws_status = "bad_delta"
            # team-attached diagnostic: A's score is always t (no mapping change)
            tA = t1 - t0
            cB = c1 - c0
            alt = "ok" if ((tA == 1 and cB == 0) or (tA == 0 and cB == 1)) else "bad"
        # elimination winner from rosters + deaths
        pwe = None
        T, C = ros.get((k, r), (set(), set()))
        D = dead.get((k, r), set())
        if len(T) > 0 and len(C) > 0:
            aT = len(T - D)
            aC = len(C - D)
            if aT > 0 and aC == 0:
                pwe = 2
            elif aC > 0 and aT == 0:
                pwe = 3
        bev = bomb.get((k, r), set())
        bs = None
        if "bomb_exploded" in bev and "bomb_defused" not in bev:
            bs = 2
        elif "bomb_defused" in bev and "bomb_exploded" not in bev:
            bs = 3
        rows.append({
            "match_key": k, "round": r, "build": b, "mdt": mdt, "map": mp,
            "post": (mdt is not None and mdt >= datetime.datetime(2024, 2, 7, 0, 0, 0)),
            "W": W, "reason": reason, "pc": pc, "L": Ld.get((k, r)),
            "PWs": pws, "pws_status": pws_status, "alt_status": alt,
            "PWe": pwe, "BS": bs, "oA": o0, "oA_prev": prev, "oA_next": oA.get(r + 1),
            "fe_next": (k, r + 1) in fe, "fas_phys": fas_phys,
            "nT": len(T), "nC": len(C), "nD": len(D),
        })

R = pl.DataFrame(rows, infer_schema_length=None)
print("\nper-round rows:", R.height, "| orientation issues:", dict(ori_fail))
print("roster size (nT,nC) dist, top:", R.group_by(["nT", "nC"]).len().sort("len", descending=True).head(8).rows())
print("pws_status:", R.group_by("pws_status").len().sort("len", descending=True).rows())
print("A2 check, score mapping (rounds with a next freeze_end and known orientation):")
print("  side-attached 'exactly one team +1' rate vs team-attached, by period x build:")
print(R.filter(pl.col("alt_status").is_not_null()).group_by(["post", "build"])
       .agg(pl.len().alias("n"), (pl.col("pws_status") == "ok").sum().alias("ok_side"),
            (pl.col("alt_status") == "ok").sum().alias("ok_team")).sort(["post", "build"]))

# ---------- S1: validate D7 ledger L against physical PWs
S1 = R.filter(pl.col("L").is_not_null() & pl.col("PWs").is_not_null())
print("\nS1 D7 ledger L vs physical score winner PWs (rounds with both):")
print(S1.group_by(["post", "build"]).agg(pl.len().alias("n"), (pl.col("L") == pl.col("PWs")).sum().alias("L_eq_PWs"))
      .with_columns((pl.col("L_eq_PWs") / pl.col("n")).round(4).alias("rate")).sort(["post", "build"]))

# ---------- S2: winner field W vs PWs
S2 = R.filter(pl.col("PWs").is_not_null())
print("\nS2 round_end W vs physical score winner PWs (rounds with PWs):")
print(S2.group_by(["post", "build"]).agg(pl.len().alias("n"), (pl.col("W") == pl.col("PWs")).sum().alias("W_eq_PWs"))
      .with_columns((pl.col("W_eq_PWs") / pl.col("n")).round(4).alias("rate")).sort(["post", "build"]))

# ---------- S3: W vs PWe (elimination), and PWs vs PWe agreement
S3 = R.filter(pl.col("PWe").is_not_null())
print("\nS3 elimination winner PWe (rounds with PWe), by period x build:")
print(S3.group_by(["post", "build"]).agg(pl.len().alias("n"),
      (pl.col("W") == pl.col("PWe")).sum().alias("W_eq_PWe")).with_columns(
      (pl.col("W_eq_PWe") / pl.col("n")).round(4).alias("rate")).sort(["post", "build"]))
S3b = R.filter(pl.col("PWe").is_not_null() & pl.col("PWs").is_not_null())
print("  PWe vs PWs (both defined):", S3b.group_by(["post", "build"]).agg(pl.len().alias("n"),
      (pl.col("PWe") == pl.col("PWs")).sum().alias("eq")).sort(["post", "build"]).rows())

# ---------- S4: bomb outcome as an independent check
S4 = R.filter(pl.col("BS").is_not_null())
print("\nS4 bomb outcome BS vs W, PWs, L (rounds with a bomb outcome):")
print(S4.group_by(["post", "build"]).agg(
      pl.len().alias("n"),
      (pl.col("W") == pl.col("BS")).sum().alias("W_eq_BS"),
      (pl.col("PWs") == pl.col("BS")).sum().alias("PWs_eq_BS"),
      (pl.col("L") == pl.col("BS")).sum().alias("L_eq_BS")).sort(["post", "build"]))

# ---------- S5: flips W != PWs (post), split by bomb presence and round
F = S2.filter(pl.col("post") & (pl.col("W") != pl.col("PWs")))
print("\nS5 post flips W != PWs:", F.height, "of", S2.filter(pl.col("post")).height, "post rounds with PWs")
print("  by build x bomb outcome present:")
print(F.with_columns(pl.col("BS").is_not_null().alias("bomb")).group_by(["build", "bomb"]).len().sort(["build", "bomb"]).rows())
print("  by round number (top 12):", F.group_by("round").len().sort("len", descending=True).head(12).rows())
print("  by fas_phys (physical side change at this round vs previous):", F.group_by("fas_phys").len().rows())
print("  by map:", F.group_by("map").len().sort("len", descending=True).rows())

# ---------- S6: first-after-physical-swap rounds: W agreement, by build and round
S6 = S2.filter(pl.col("fas_phys") == True)
print("\nS6 rounds where roster side changes vs previous round (fas_phys=True), W==PWs by build x period x round:")
print(S6.group_by(["post", "build", "round"]).agg(pl.len().alias("n"), (pl.col("W") == pl.col("PWs")).sum().alias("W_eq_PWs"))
      .sort(["post", "build", "round"]).rows())
S6n = S2.filter(pl.col("fas_phys") == False)
print("S6 non-swap rounds, W!=PWs by build x period (count, flips):")
print(S6n.group_by(["post", "build"]).agg(pl.len().alias("n"), (pl.col("W") != pl.col("PWs")).sum().alias("flips"))
      .sort(["post", "build"]).rows())

# ---------- S7: matches, time vs build
M = (S2.filter(pl.col("post")).group_by(["match_key", "build"])
       .agg(pl.len().alias("n"), (pl.col("W") != pl.col("PWs")).sum().alias("flips")))
print("\nS7 post matches with >=1 physical flip, by build (matches, with flip, flip rate per round):")
print(M.group_by("build").agg(pl.len().alias("matches"), (pl.col("flips") > 0).sum().alias("with_flip"),
      (pl.col("flips").sum() / pl.col("n").sum()).round(4).alias("flip_rate")).sort("build").rows())
pre_M = (S2.filter(~pl.col("post")).group_by(["match_key", "build"])
           .agg(pl.len().alias("n"), (pl.col("W") != pl.col("PWs")).sum().alias("flips")))
print("S7 pre matches with >=1 physical flip, by build:")
print(pre_M.group_by("build").agg(pl.len().alias("matches"), (pl.col("flips") > 0).sum().alias("with_flip")).sort("build").rows())

# ======================= extended diagnostics =======================
print("\n================ EXTENDED DIAGNOSTICS (player_count == 20 subset, as in D7) ================")
R20 = R.filter(pl.col("pc") == 20)
print("rounds pc==20:", R20.height, "| pc dist (all rounds):", R.group_by("pc").len().sort("pc").rows())

# D7 universe: rows with a D7 ledger value and W
U = R20.filter(pl.col("L").is_not_null() & pl.col("W").is_not_null())
print("\nD7 universe (pc20, L and W known):", U.height)
print("  W != L by period x build:", U.group_by(["post", "build"]).agg(pl.len().alias("n"),
      (pl.col("W") != pl.col("L")).sum().alias("W_ne_L")).sort(["post", "build"]).rows())
U = U.with_columns(pl.when(pl.col("PWs").is_null())
                   .then(pl.when(pl.col("oA").is_null()).then(pl.lit("no_oA_r"))
                         .when(pl.col("oA_next").is_null()).then(pl.lit("no_oA_r1"))
                         .otherwise(pl.lit("no_score_pair")))
                   .otherwise(pl.lit("covered")).alias("cov"))
print("  coverage of physical check within D7 universe (by post x build):")
print(U.group_by(["post", "build", "cov"]).agg(pl.len().alias("n"), (pl.col("W") != pl.col("L")).sum().alias("W_ne_L"))
      .sort(["post", "build", "cov"]))
print("  W != L rows that are NOT covered by physical check, by round (top 12):")
print(U.filter((pl.col("W") != pl.col("L")) & (pl.col("cov") != "covered")).group_by(["post", "build", "round"])
      .len().sort("len", descending=True).head(12).rows())

# round 9 detail
r9 = R20.filter((pl.col("round") == 9) & pl.col("post"))
print("\nround 9 post (pc20): n=", r9.height, "| L!=W:", r9.filter(pl.col("L").is_not_null() & (pl.col("L") != pl.col("W"))).height)
print("  by build x cov:", r9.with_columns(pl.col("PWs").is_null().alias("pws_null")).group_by(["build", "pws_null", "oA_prev"]).len().sort(["build", "pws_null", "oA_prev"]).rows())
print("  fas_phys at round 9 (any build):", R20.filter(pl.col("round") == 9).group_by("fas_phys").len().rows())
print("  sample of L!=W round-9 rows (match, build, oA, oA_prev, oA_next, nT, nC, PWs, PWe, W, L):")
for row in r9.filter(pl.col("L").is_not_null() & (pl.col("L") != pl.col("W"))).head(6).iter_rows(named=True):
    print("   ", row["match_key"][:40], row["build"], row["oA"], row["oA_prev"], row["oA_next"], row["nT"], row["nC"],
          row["PWs"], row["PWe"], row["W"], row["L"], row["pws_status"])

# physical swap pattern per match
sw = collections.Counter()
for k, rl, fl, bl in R.group_by("match_key").agg(pl.col("round"), pl.col("fas_phys"), pl.col("build").first()).iter_rows():
    rr = sorted(zip(rl, fl))
    rounds_sw = tuple(r for r, f in rr if f is True)
    sw[(bl, rounds_sw)] += 1
print("\nphysical swap pattern (build, swap rounds where roster side changed) top 15:",
      sorted(sw.items(), key=lambda x: -x[1])[:15])

# W != PWs flips: PWe cross-tab (both defined)
FL = R20.filter(pl.col("post") & (pl.col("build") == 9961) & pl.col("PWs").is_not_null() & (pl.col("W") != pl.col("PWs")))
print("\nphysical flips (post 9961, pc20, W != PWs):", FL.height)
print("  PWe relation: PWe==PWs:", FL.filter(pl.col("PWe") == pl.col("PWs")).height,
      "| PWe==W:", FL.filter(pl.col("PWe") == pl.col("W")).height,
      "| PWe null:", FL.filter(pl.col("PWe").is_null()).height)
print("  flips by round x bomb:", FL.with_columns(pl.col("BS").is_not_null().alias("bomb")).group_by(["round", "bomb"]).len().sort(["round", "bomb"]).rows())
print("  flips with reason codes (W_reason, count top 8):", FL.group_by("reason").len().sort("len", descending=True).head(8).rows())

# bomb outcome on common rows
BC = R20.filter(pl.col("BS").is_not_null() & pl.col("PWs").is_not_null())
print("\nbomb rows with PWs known (pc20): n=", BC.height)
print(BC.group_by(["post", "build"]).agg(pl.len().alias("n"), (pl.col("W") == pl.col("BS")).sum().alias("W_eq_BS"),
      (pl.col("PWs") == pl.col("BS")).sum().alias("PWs_eq_BS")).sort(["post", "build"]).rows())
BCL = R20.filter(pl.col("BS").is_not_null() & pl.col("L").is_not_null())
print("bomb rows with L known (pc20):", BCL.group_by(["post", "build"]).agg(pl.len().alias("n"),
      (pl.col("L") == pl.col("BS")).sum().alias("L_eq_BS")).sort(["post", "build"]).rows())

# non-bomb physical flips by round
NB = R20.filter(pl.col("post") & pl.col("PWs").is_not_null() & pl.col("BS").is_null())
print("\nnon-bomb post rounds (pc20, PWs known):", NB.height, "| W != PWs:", NB.filter(pl.col("W") != pl.col("PWs")).height)
print("  by build x round (flip counts, top 10):", NB.filter(pl.col("W") != pl.col("PWs")).group_by(["build", "round"]).len()
      .sort("len", descending=True).head(10).rows())

# round 25 detail
r25 = R20.filter((pl.col("round") == 25) & pl.col("post") & (pl.col("build") == 9961))
print("\nround 25 (9961 post, pc20): n=", r25.height, "| PWs known:", r25.filter(pl.col("PWs").is_not_null()).height,
      "| W!=PWs:", r25.filter(pl.col("W") != pl.col("PWs")).height,
      "| fas_phys:", r25.group_by("fas_phys").len().rows(),
      "| PWe==PWs (PWs known & PWe known):", r25.filter(pl.col("PWe").is_not_null() & pl.col("PWs").is_not_null() & (pl.col("PWe") == pl.col("PWs"))).height,
      "| PWe known:", r25.filter(pl.col("PWe").is_not_null()).height)
print("  round 25 bomb outcome present:", r25.filter(pl.col("BS").is_not_null()).height,
      "| flips among bomb rows:", r25.filter((pl.col("BS").is_not_null()) & (pl.col("W") != pl.col("PWs"))).height)

# W vs elimination on physical flips
print("\nW vs PWe on all post 9961 pc20 rounds with PWe known: W==PWe", 
      R20.filter(pl.col("post") & (pl.col("build") == 9961) & pl.col("PWe").is_not_null()).select(
          (pl.col("W") == pl.col("PWe")).sum().alias("eq"), pl.len().alias("n")).rows())

# ======================= trace of round-9 mismatch matches =======================
print("\n================ TRACE: round 7-13 for matches with W != L at round 9 (post, pc20) ================")
bad9 = R20.filter((pl.col("round") == 9) & pl.col("post") & pl.col("L").is_not_null() & (pl.col("L") != pl.col("W")))
keys9 = bad9.select("match_key").unique().head(3)["match_key"].to_list()
for k in keys9:
    print("\nmatch", k[:45], "build", hdr[k][0], "map", hdr[k][2])
    print("  round | nT nC | oA | fe(t,c) | W | L | PWs | PWe | BS | reason | ndead | T-ids")
    for r in range(6, 15):
        T, C = ros.get((k, r), (set(), set()))
        row = R.filter((pl.col("match_key") == k) & (pl.col("round") == r))
        if row.height == 0:
            print("  ", r, "(no round_end row)", "fe" , fe.get((k, r)), "nT,nC", len(T), len(C))
            continue
        x = row.row(0, named=True)
        print("  ", r, "|", x["nT"], x["nC"], "|", x["oA"], "|", fe.get((k, r)), "|", x["W"], "|", x["L"], "|", x["PWs"], "|", x["PWe"],
              "|", x["BS"], "|", x["reason"], "|", x["nD"], "|", sorted(T)[:5])

# ======================= second pass: roster class, per-map, per-match, DP errors =======================
print("\n================ SECOND PASS (after anchor fix) ================")
# roster class per match: modal (nT, nC) over rounds with a roster
mc = (R.filter((pl.col("nT") > 0) & (pl.col("nC") > 0))
        .group_by(["match_key", "nT", "nC"]).len()
        .sort(["match_key", "len"], descending=[False, True])
        .group_by("match_key").agg(pl.col("nT").first().alias("mT"), pl.col("nC").first().alias("mC")))
R2 = R.join(mc, on="match_key", how="left").with_columns(
    pl.when((pl.col("mT") <= 2) & (pl.col("mC") <= 2)).then(pl.lit("2v2")).otherwise(pl.lit("5v5ish")).alias("mode"))
print("matches by mode x build x period:", R2.select(["match_key", "mode", "build", "post"]).unique()
      .group_by(["mode", "build", "post"]).len().sort(["mode", "build", "post"]).rows())
print("pc20 rounds by mode x build x post:", R2.filter(pl.col("pc") == 20).group_by(["mode", "build", "post"]).len()
      .sort(["mode", "build", "post"]).rows())
print("physical coverage (oA known) by mode x build x post:")
print(R2.group_by(["mode", "build", "post"]).agg(pl.len().alias("n"), pl.col("oA").is_not_null().sum().alias("oA_known"),
      pl.col("PWs").is_not_null().sum().alias("PWs_known")).sort(["mode", "build", "post"]).rows())

S2b = R2.filter(pl.col("pc") == 20 & pl.col("PWs").is_not_null() if False else (pl.col("pc") == 20) & pl.col("PWs").is_not_null())
print("\nW vs PWs (pc20, PWs known) by mode x build x post:")
print(S2b.group_by(["mode", "build", "post"]).agg(pl.len().alias("n"), (pl.col("W") != pl.col("PWs")).sum().alias("W_ne_PWs"))
      .sort(["mode", "build", "post"]).rows())
print("W vs PWs flips by round (pc20, post, all builds):", S2b.filter(pl.col("post") & (pl.col("W") != pl.col("PWs")))
      .group_by(["build", "mode", "round"]).len().sort("len", descending=True).head(15).rows())
print("fas_phys True rounds by mode x build x round (pc20):", S2b.filter(pl.col("fas_phys") == True)
      .group_by(["mode", "build", "round"]).agg(pl.len().alias("n"), (pl.col("W") != pl.col("PWs")).sum().alias("W_ne_PWs"))
      .sort(["mode", "build", "round"]).rows())

# DP ledger vs physical winner: where does the DP disagree (L != PWs)?
dpe = R2.filter((pl.col("pc") == 20) & pl.col("L").is_not_null() & pl.col("PWs").is_not_null() & (pl.col("L") != pl.col("PWs")))
print("\nDP ledger L != PWs (pc20, both known):", dpe.height, "| by build x round (top 12):",
      dpe.group_by(["build", "round"]).len().sort("len", descending=True).head(12).rows())
print("  DP L != PWs, by W agreement:", dpe.select((pl.col("W") == pl.col("PWs")).sum().alias("W_eq_PWs"), pl.len().alias("n")).rows())

# flips (W != PWs, pc20, post): bomb / elimination evidence
FLP = S2b.filter(pl.col("post") & (pl.col("W") != pl.col("PWs")))
print("\nflips W!=PWs (pc20, post):", FLP.height, "| with bomb outcome:", FLP.filter(pl.col("BS").is_not_null()).height)
print("  bomb flips: PWs==BS:", FLP.filter(pl.col("BS").is_not_null() & (pl.col("PWs") == pl.col("BS"))).height,
      "| W==BS:", FLP.filter(pl.col("BS").is_not_null() & (pl.col("W") == pl.col("BS"))).height)
NBF = FLP.filter(pl.col("BS").is_null())
print("  non-bomb flips:", NBF.height, "| PWe known:", NBF.filter(pl.col("PWe").is_not_null()).height,
      "| PWe==PWs:", NBF.filter(pl.col("PWe") == pl.col("PWs")).height, "| PWe==W:", NBF.filter(pl.col("PWe") == pl.col("W")).height)
print("  non-bomb flips by round:", NBF.group_by("round").len().sort("len", descending=True).head(8).rows())

# bomb rounds (pc20, PWs known) overall agreement with bomb outcome
BB = S2b.filter(pl.col("BS").is_not_null())
print("\nbomb rows (pc20, PWs known):", BB.height, "| PWs==BS:", BB.filter(pl.col("PWs") == pl.col("BS")).height,
      "| W==BS:", BB.filter(pl.col("W") == pl.col("BS")).height)

# per-map W!=PWs rate (post, 9961, pc20)
pm = S2b.filter(pl.col("post") & (pl.col("build") == 9961)).group_by("map").agg(
    pl.len().alias("n"), (pl.col("W") != pl.col("PWs")).sum().alias("flips"),
    ((pl.col("W") != pl.col("PWs")).mean()).round(4).alias("rate")).sort("rate", descending=True)
print("\nper-map post 9961 W!=PWs rate (pc20):", pm.rows())
pmp = S2b.filter(~pl.col("post")).group_by("map").agg(pl.len().alias("n"), ((pl.col("W") != pl.col("PWs")).mean()).round(4).alias("rate"))
print("per-map pre (all builds) rate:", pmp.rows())

# per-match: post matches with >=1 flip (pc20, PWs known), by build
MP = S2b.filter(pl.col("post")).group_by(["match_key", "build"]).agg((pl.col("W") != pl.col("PWs")).sum().alias("flips"), pl.len().alias("n"))
print("post matches with PWs rows (pc20):", MP.height, "| with >=1 flip:", MP.filter(pl.col("flips") > 0).height,
      "| by build:", MP.group_by("build").agg(pl.len().alias("matches"), (pl.col("flips") > 0).sum().alias("with_flip")).sort("build").rows())
n9842_post = sum(1 for k, (b, m, mp) in hdr.items() if b == 9842 and m is not None and m >= datetime.datetime(2024, 2, 7))
n9961_pre = sum(1 for k, (b, m, mp) in hdr.items() if b == 9961 and m is not None and m < datetime.datetime(2024, 2, 7))
print("header: 9842 matches on/after 2024-02-07 00:00 UTC:", n9842_post, "| 9961 matches before 2024-02-07 00:00 UTC:", n9961_pre)
print("header: 9842 matches after 2024-02-07 01:01:18 UTC:", sum(1 for k, (b, m, mp) in hdr.items() if b == 9842 and m is not None and m >= datetime.datetime(2024, 2, 7, 1, 1, 18)))

# ======================= third pass: pc filter sensitivity, reconciliation to D7 numbers =======================
print("\n================ THIRD PASS: pc sensitivity and reconciliation ================")
A3 = R2.filter(pl.col("PWs").is_not_null())
print("W vs PWs, ALL player_count values, by mode x build x post x pc (rows, flips):")
print(A3.group_by(["mode", "build", "post", "pc"]).agg(pl.len().alias("n"), (pl.col("W") != pl.col("PWs")).sum().alias("flips"))
      .sort(["mode", "build", "post", "pc"]).rows())
print("flip rounds (all pc, all builds):", A3.filter(pl.col("W") != pl.col("PWs")).group_by(["build", "mode", "round"]).len()
      .sort("len", descending=True).head(12).rows())
print("2v2 9842 pre matches: rounds with W!=PWs by round (all pc):",
      A3.filter((pl.col("mode") == "2v2") & (pl.col("build") == 9842) & (pl.col("W") != pl.col("PWs"))).group_by("round").len().rows())
print("2v2 9842 pre fas_phys True by round (all pc):", A3.filter((pl.col("mode") == "2v2") & (pl.col("build") == 9842) & (pl.col("fas_phys") == True))
      .group_by("round").len().sort("round").rows())

# reconciliation of D7 W!=L counts (pc20, post 9961) to physical truth
U2 = R2.filter((pl.col("pc") == 20) & pl.col("post") & pl.col("L").is_not_null() & pl.col("W").is_not_null())
print("\nreconciliation (pc20, post, L and W known):", U2.height)
print("  W!=L:", U2.filter(pl.col("W") != pl.col("L")).height,
      "| split bomb:", U2.filter((pl.col("W") != pl.col("L")) & pl.col("BS").is_not_null()).height,
      "| non-bomb:", U2.filter((pl.col("W") != pl.col("L")) & pl.col("BS").is_null()).height)
print("  W!=L & W==PWs (ledger error rows):", U2.filter((pl.col("W") != pl.col("L")) & (pl.col("W") == pl.col("PWs"))).height,
      "| by round:", U2.filter((pl.col("W") != pl.col("L")) & (pl.col("W") == pl.col("PWs"))).group_by("round").len().rows(),
      "| bomb:", U2.filter((pl.col("W") != pl.col("L")) & (pl.col("W") == pl.col("PWs")) & pl.col("BS").is_not_null()).height)
print("  W!=PWs (physical defect rows, PWs known):", U2.filter(pl.col("W") != pl.col("PWs")).height,
      "| with L==PWs:", U2.filter((pl.col("W") != pl.col("PWs")) & (pl.col("L") == pl.col("PWs"))).height)
print("  round 13 rows with L known:", U2.filter(pl.col("round") == 13).height,
      "| W!=L:", U2.filter((pl.col("round") == 13) & (pl.col("W") != pl.col("L"))).height,
      "| of W!=L at 13, mode:", U2.filter((pl.col("round") == 13) & (pl.col("W") != pl.col("L"))).group_by("mode").len().rows(),
      "| round-13 rows that agree, mode:", U2.filter((pl.col("round") == 13) & (pl.col("W") == pl.col("L"))).group_by("mode").len().rows())
print("  round 9 rows with L known:", U2.filter(pl.col("round") == 9).height,
      "| W!=L:", U2.filter((pl.col("round") == 9) & (pl.col("W") != pl.col("L"))).height)

# flips per match distribution (pc20, post, PWs known)
FPM = S2b.filter(pl.col("post")).group_by("match_key").agg((pl.col("W") != pl.col("PWs")).sum().alias("f"))
print("\nflips per post match (pc20, PWs known), distribution:", FPM.group_by("f").len().sort("f").rows())
print("flip round pattern per flipping match (top 8):", S2b.filter(pl.col("post") & (pl.col("W") != pl.col("PWs")))
      .group_by("match_key").agg(pl.col("round").sort()).group_by(pl.col("round").list.eval(pl.element().cast(pl.Utf8)).list.join("-"))
      .len().sort("len", descending=True).head(8).rows() if False else "skipped")

# ======================= fourth pass: off-by-one label shift alternative =======================
print("\n================ FOURTH PASS: off-by-one shift test (pc20, post, W and PWs known) ================")
prev_ = R.select(["match_key", "round", "PWs"]).with_columns((pl.col("round") + 1).alias("round")).rename({"PWs": "PWs_prev"})
next_ = R.select(["match_key", "round", "PWs"]).with_columns((pl.col("round") - 1).alias("round")).rename({"PWs": "PWs_next"})
SH = (R2.filter((pl.col("pc") == 20) & pl.col("post") & pl.col("PWs").is_not_null() & (pl.col("build") == 9961))
        .join(prev_, on=["match_key", "round"], how="left").join(next_, on=["match_key", "round"], how="left"))
FF = SH.filter(pl.col("W") != pl.col("PWs"))
print("9961 post flips:", FF.height)
print("  flips where W == PWs of round r-1:", FF.filter(pl.col("W") == pl.col("PWs_prev")).height,
      "| W == PWs of round r+1:", FF.filter(pl.col("W") == pl.col("PWs_next")).height,
      "| PWs_prev known:", FF.filter(pl.col("PWs_prev").is_not_null()).height)
NF = SH.filter(pl.col("W") == pl.col("PWs"))
print("9961 post non-flips:", NF.height, "| W == PWs of round r-1:", NF.filter(pl.col("W") == pl.col("PWs_prev")).height,
      "| W == PWs of round r+1:", NF.filter(pl.col("W") == pl.col("PWs_next")).height)
print("  flip rounds where W == PWs_prev (by round):", FF.filter(pl.col("W") == pl.col("PWs_prev")).group_by("round").len().rows())
