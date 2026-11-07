"""A5 audit: state construction in scripts/model/bakeoff.py.

Covers the round filters and duplicate-match removal that feed build_states, build_states itself
(live window, death handling, duplicate victims, null-side imputation, plant timing, seconds
features, is_overtime, sentinels), and swap_expected against the score data in every match.

Read-only. Nothing under the project is written. bakeoff.py is loaded by path and its main() is
not run. Output goes to stdout; the shell redirects it to logs/swarm2/A5_audit.log.
"""

import collections
import glob
import importlib.util
import time
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
_spec = importlib.util.spec_from_file_location("bakeoff", ROOT / "scripts" / "model" / "bakeoff.py")
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)
fv = bk.fv
T, CT, START, TICK = bk.T, bk.CT, bk.START, bk.TICK_RATE
WIN = fv.WIN_REASON
t0 = time.time()


def section(title):
    print(f"\n== {title} ==", flush=True)


def quant(vals):
    if not vals:
        return "n/a"
    s = sorted(vals)
    n = len(s)
    return [round(float(s[min(n - 1, int(round(q * (n - 1))))]), 3) for q in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)]


def bucket(g):
    if g <= 0.5:
        return "a<=0.5s"
    if g <= 1:
        return "b0.5-1s"
    if g <= 2:
        return "c1-2s"
    if g <= 5:
        return "d2-5s"
    if g <= 30:
        return "e5-30s"
    return "f>30s"


section("0. Constants and swap_expected spot checks")
print(f"T={T} CT={CT} START={START} TICK_RATE={TICK} SUFFIX={fv.SUFFIX}")
print(f"FEATS={bk.FEATS}")
print(f"MONO={dict(zip(bk.FEATS, bk.MONO))}")
print(f"WIN_REASON={WIN}")
checks = [(8, 16), (12, 24), (13, 24), (19, 16), (27, 24), (33, 24), (39, 24), (45, 24), (9, 24), (16, 16), (24, 24), (25, 24)]
print("swap_expected(r, max):", {f"{r},{mr}": bk.swap_expected(r, mr) for r, mr in checks})

section("1. Channel reads (projected columns)")
hdr = fv.read("header", ["key", "match_id", "match_date", "max_rounds", "tick_rate"])
H = {r["key"]: r for r in hdr.iter_rows(named=True)}
RE = fv.read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
RS = fv.read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
DE = fv.read("player_death", ["match_key", "round", "tick", "player_id", "player_id_fixed",
                              "player_team_code", "weapon_name", "attacker_id"])
PI = fv.read("player_info", ["match_key", "round", "player_id", "player_id_fixed", "team_code"])
BS = fv.read("bomb_state", ["match_key", "round", "tick", "event_type", "player_id"])
RST = fv.read("round_start", ["match_key", "round", "tick"])
DC = fv.read("player_disconnect", ["match_key", "round", "tick"])
BT = fv.read("bot_takeover", ["match_key", "round", "tick"])
TC = fv.read("team_change", ["match_key", "round", "tick"])
print(f"header matches {len(H)}; max_rounds {dict(collections.Counter(h['max_rounds'] for h in H.values()))}")
print(f"rows: RE {RE.height} RS {RS.height} DE {DE.height} PI {PI.height} BS {BS.height} "
      f"RST {RST.height} DC {DC.height} BT {BT.height} TC {TC.height}")
print(f"round_state event types: {sorted(RS['event_type'].unique().to_list())}")
print(f"bomb_state event types: {sorted(BS['event_type'].unique().to_list())}")

# ---------- 2. raw data checks (all matches, no filters)
section("2a. player_death ids and team codes (all matches)")
print(f"rows {DE.height}")
print(f"raw player_id: null {DE['player_id'].null_count()}, ==0 {int((DE['player_id'] == 0).sum())}, "
      f"==65535 {int((DE['player_id'] == 65535).sum())}")
print(f"player_id_fixed: null {DE['player_id_fixed'].null_count()}, ==0 {int((DE['player_id_fixed'] == 0).sum())}")
dis = DE.filter(pl.col("player_id").is_not_null() & pl.col("player_id_fixed").is_not_null()
                & (pl.col("player_id").cast(pl.Float64) != pl.col("player_id_fixed")))
print(f"rows where raw and fixed player_id differ (both non-null): {dis.height}")
z = DE.filter(pl.col("player_id") == 0)
print(f"raw player_id==0 death rows: {z.height}; weapons: {dict(collections.Counter(z['weapon_name'].to_list()).most_common(6))}")
print(f"player_team_code: null {DE['player_team_code'].null_count()}; values "
      f"{DE['player_team_code'].drop_nulls().unique().sort().to_list()}")
print(f"attacker_id: null {DE['attacker_id'].null_count()}; ==65535 {int((DE['attacker_id'] == 65535).sum())}")
dup_raw = (DE.filter(pl.col("player_id").is_not_null())
           .group_by(["match_key", "round", "player_id"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1))
print(f"(match, round, raw player_id) groups with >1 death row, all ticks: {dup_raw.height} groups, "
      f"{int(dup_raw['n'].sum()) if dup_raw.height else 0} rows; of which raw pid 0: "
      f"{int((dup_raw['player_id'] == 0).sum()) if dup_raw.height else 0}")
dup_fix = (DE.filter(pl.col("player_id_fixed").is_not_null())
           .group_by(["match_key", "round", "player_id_fixed"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1))
print(f"(match, round, player_id_fixed) groups with >1 death row, all ticks: {dup_fix.height}")

section("2b. player_info and freeze/end rows (all matches)")
print(f"player_info: raw player_id==0 rows {int((PI['player_id'] == 0).sum())}; raw vs fixed differ "
      f"{PI.filter(pl.col('player_id').cast(pl.Float64) != pl.col('player_id_fixed').cast(pl.Float64)).height}; "
      f"team_code values {PI['team_code'].drop_nulls().unique().sort().to_list()}; null team {PI['team_code'].null_count()}")
pi_per = PI.group_by(["match_key", "round"]).agg(pl.len().alias("n"))
print(f"player_info rows per (match, round): {dict(sorted(collections.Counter(pi_per['n'].to_list()).items()))}")
frz_rows = RS.filter(pl.col("event_type") == "round_freeze_end")
cnt = frz_rows.group_by(["match_key", "round"]).agg(pl.len().alias("n"))
print(f"round_freeze_end rows {frz_rows.height}; per (match, round): "
      f"{dict(sorted(collections.Counter(cnt['n'].to_list()).items()))}")
r1 = frz_rows.filter(pl.col("round") == 1)["tick"].to_list()
allf = frz_rows["tick"].to_list()
print(f"round 1 freeze-end tick: n={len(r1)} quantiles(0,5,25,50,75,95,100)={quant(r1)} ==66: "
      f"{sum(1 for t in r1 if t == 66)}")
print(f"all freeze-end ticks: ==66 {sum(1 for t in allf if t == 66)}; <=200 {sum(1 for t in allf if t <= 200)}; "
      f"quantiles {quant(allf)}")
rst1 = RST.filter(pl.col("round") == 1)["tick"].to_list()
print(f"round_start (round 1) ticks: ==66 {sum(1 for t in rst1 if t == 66)} of {len(rst1)}")
re_per = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n"))
print(f"round_end rows per (match, round): {dict(collections.Counter(re_per['n'].to_list()))}")

frzmap = {}
for m, r, t in frz_rows.select(["match_key", "round", "tick"]).iter_rows():
    frzmap.setdefault((m, int(r)), int(t))
endmap = {}
for m, r, t in RE.select(["match_key", "round", "tick"]).iter_rows():
    endmap.setdefault((m, int(r)), int(t))
rwin = {}
for m, r, w, rs in RE.select(["match_key", "round", "winner_team_code", "win_reason_code"]).iter_rows():
    rwin.setdefault((m, int(r)), (w, rs))

section("2c. player_death classification vs live window (all matches)")
cls = collections.Counter()
post_gap = []
post_rows = collections.defaultdict(list)
post_side = collections.Counter()
post_weapon = []
inwin_null = 0
pid_zero_inwin = 0
for m, r, t, pid, team, wpn in DE.select(["match_key", "round", "tick", "player_id", "player_team_code",
                                          "weapon_name"]).iter_rows():
    k = (m, int(r))
    if k not in frzmap or k not in endmap:
        cls["no freeze or no end row"] += 1
        continue
    f, e = frzmap[k], endmap[k]
    if t < f:
        cls["pre-freeze"] += 1
    elif t > e:
        cls["post-end"] += 1
        post_gap.append((t - e) / TICK)
        post_rows[k].append((t, team, pid))
        post_side[team] += 1
        post_weapon.append((k, wpn, (t - e) / TICK))
    else:
        cls["in window"] += 1
        if team is None:
            inwin_null += 1
        if pid == 0:
            pid_zero_inwin += 1
print(f"classification: {dict(cls)}")
print(f"post-end deaths: gap after round_end in seconds, buckets {dict(sorted(collections.Counter(bucket(g) for g in post_gap).items()))}; "
      f"quantiles {quant(post_gap)}")
print(f"post-end deaths by victim player_team_code: {dict(post_side)}")
print(f"in-window null player_team_code (all matches): {inwin_null}; in-window raw pid 0: {pid_zero_inwin}")

pi_team = {}
for m, r, pid, tc in PI.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
    pi_team[(m, int(r), pid)] = tc
agree = disagree = pid_missing = 0
ex_dis = []
for m, r, t, pid, team in DE.select(["match_key", "round", "tick", "player_id", "player_team_code"]).iter_rows():
    k = (m, int(r))
    if k not in frzmap or k not in endmap or not (frzmap[k] <= t <= endmap[k]) or team is None:
        continue
    pt = pi_team.get((m, int(r), pid))
    if pt is None:
        pid_missing += 1
    elif pt == team:
        agree += 1
    else:
        disagree += 1
        if len(ex_dis) < 5:
            ex_dis.append((m[:30], int(r), pid, team, pt))
print(f"in-window deaths with non-null team: death team == player_info team {agree}; differs {disagree}; "
      f"pid not in player_info {pid_missing}; examples {ex_dis}")

section("2c-bis. post-round_end deaths: weapon and outcome; explosions after round_end (all matches)")
print(f"post-end deaths by weapon (top 10): {collections.Counter(w for _, w, _ in post_weapon).most_common(10)}")
print(f"post-end deaths by round outcome: "
      f"{dict(collections.Counter(WIN.get(rwin.get(k, (None, None))[1], '?') for k, _, _ in post_weapon))}")
print(f"post-end deaths of weapon planted_c4 (bomb kill): "
      f"{sum(1 for _, w, _ in post_weapon if w == 'planted_c4')}; gap quantiles of those {quant([g for _, w, g in post_weapon if w == 'planted_c4'])}")
exp_gap = []
for m, r, t in BS.filter(pl.col("event_type") == "bomb_exploded").select(["match_key", "round", "tick"]).iter_rows():
    k = (m, int(r))
    if k in endmap and t > endmap[k]:
        exp_gap.append((t - endmap[k]) / TICK)
n_exp = BS.filter(pl.col("event_type") == "bomb_exploded").height
print(f"bomb_exploded rows {n_exp}; after round_end {len(exp_gap)}; gap quantiles {quant(exp_gap)}; "
      f"buckets {dict(sorted(collections.Counter(bucket(g) for g in exp_gap).items()))}")

section("2d. bomb_planted timing vs live window (all matches)")
bp = BS.filter(pl.col("event_type") == "bomb_planted")
pc = collections.Counter()
pgap = []
plant_by_round = collections.defaultdict(list)
after_end_rounds = set()
for m, r, t in bp.select(["match_key", "round", "tick"]).iter_rows():
    k = (m, int(r))
    plant_by_round[k].append(int(t))
    if k not in frzmap or k not in endmap:
        pc["no freeze or end row"] += 1
        continue
    f, e = frzmap[k], endmap[k]
    if t < f:
        pc["before freeze"] += 1
    elif t > e:
        pc["after round_end"] += 1
        pgap.append((t - e) / TICK)
        after_end_rounds.add(k)
    else:
        pc["in window"] += 1
print(f"bomb_planted rows {bp.height}: {dict(pc)}")
print(f"after round_end, gap after end in seconds: buckets {dict(sorted(collections.Counter(bucket(g) for g in pgap).items()))}; "
      f"quantiles {quant(pgap)}")
print(f"rounds with >1 bomb_planted row (all matches): {sum(1 for v in plant_by_round.values() if len(v) > 1)}")
exp_rounds = set()
def_rounds = set()
for m, r, ev in BS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).select(
        ["match_key", "round", "event_type"]).iter_rows():
    (exp_rounds if ev == "bomb_exploded" else def_rounds).add((m, int(r)))
oc = collections.Counter()
for k in after_end_rounds:
    w, rs = rwin.get(k, (None, None))
    oc[(w, WIN.get(rs, str(rs)), "exploded row" if k in exp_rounds else "no exploded row",
        "defused row" if k in def_rounds else "no defused row")] += 1
print("after-round_end plant rounds by (winner, reason, exploded row, defused row):")
for key, n in sorted(oc.items(), key=lambda kv: -kv[1]):
    print(f"   {key}: {n}")

# ---------- 3. reproduce bakeoff steps 1-2 (filters, dedupe) and call bk.build_states unchanged
section("3. Reproduce bakeoff filters and build_states")
all_matches = set(H)
frames = {"round_end": RE, "round_start": RST, "round_state": RS, "player_death": DE,
          "player_info": PI, "bomb_state": BS, "team_change": TC,
          "player_disconnect": DC, "bot_takeover": BT}
missing_by_match = collections.defaultdict(list)
for ch in bk.DENSE + bk.SPARSE:
    df = frames[ch]
    ks = bk.keyset_keys(ch)
    have = set(df["match_key"].unique().to_list())
    dense = ch in bk.DENSE
    miss = (all_matches - ks) | ((all_matches - have) if dense else set())
    for m in miss:
        missing_by_match[m].append(ch)
KEPT = all_matches - set(missing_by_match)
print(f"matches kept after channel check: {len(KEPT)} (bakeoff log: 1709)")

sig_parts = collections.defaultdict(list)
for x in RE.iter_rows(named=True):
    sig_parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"]))))
for x in DE.iter_rows(named=True):
    sig_parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
for x in PI.iter_rows(named=True):
    sig_parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
SIGS = {m: tuple(sorted(v)) for m, v in sig_parts.items()}
sig_groups = collections.defaultdict(list)
for m in sorted(KEPT):
    sig_groups[SIGS.get(m, ())].append(m)
DEDUP_GROUPS = [g for g in sig_groups.values() if len(g) > 1]
DUP_DROP = set()
for g in DEDUP_GROUPS:
    DUP_DROP.update(g[1:])
print(f"duplicate groups among kept matches: {len(DEDUP_GROUPS)} groups; matches to drop: {len(DUP_DROP)} (bakeoff log: 46 groups, 97 matches involved)")


def pipeline(pid_col, pi_col):
    rd = {}
    for x in RE.iter_rows(named=True):
        m = x["match_key"]
        if m in KEPT:
            rd[(m, int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                        "pc": x["player_count"]}
    rst_keys = {(x["match_key"], int(x["round"])) for x in RST.iter_rows(named=True) if x["match_key"] in KEPT}
    live = {}
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        m = x["match_key"]
        if m not in KEPT:
            continue
        k = (m, int(x["round"]))
        if k in live:
            continue
        live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        m = x["match_key"]
        if m in KEPT:
            players[(m, int(x["round"]))][x[pi_col]] = x["team_code"]
    disc_rounds = set()
    for df in (DC, BT):
        for x in df.iter_rows(named=True):
            if x["match_key"] in KEPT:
                disc_rounds.add((x["match_key"], int(x["round"])))
    plants = collections.defaultdict(list)
    for x in BS.filter(pl.col("event_type") == "bomb_planted").iter_rows(named=True):
        if x["match_key"] in KEPT:
            plants[(x["match_key"], int(x["round"]))].append(int(x["tick"]))
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        m = x["match_key"]
        if m in KEPT:
            deaths[(m, int(x["round"]))].append({"tick": int(x["tick"]), "pid": x[pid_col],
                                                 "team": x["player_team_code"],
                                                 "weapon": x["weapon_name"], "attacker": x["attacker_id"]})
    reasons = collections.Counter()
    survivors = []
    for k in sorted(rd):
        rr = rd[k]
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if rr["pc"] != 10:
            reasons["player_count != 10"] += 1
            continue
        if not (len(pl_) == 10 and n_t == START and n_ct == START):
            reasons["not 5v5 at start"] += 1
            continue
        if k in disc_rounds:
            reasons["disconnect or takeover"] += 1
            continue
        if k not in rst_keys:
            reasons["no round_start"] += 1
            continue
        if k not in live:
            reasons["no live start"] += 1
            continue
        if live[k]["t"] is None or live[k]["ct"] is None:
            reasons["null score at live start"] += 1
            continue
        survivors.append(k)
    survivors = [k for k in survivors if k[0] not in DUP_DROP]
    reasons["duplicate copy removed (rounds)"] = 0
    diag = collections.Counter()
    state_rows = []
    rounds_ok = []
    for k in survivors:
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        local = collections.Counter()
        out = bk.build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                              H[k[0]]["max_rounds"], local)
        if out is None:
            reasons["death with unresolved victim side"] += 1
            continue
        diag.update(local)
        state_rows.extend(out)
        rounds_ok.append(k)
    S = pl.DataFrame(state_rows)
    return {"S": S, "rounds_ok": rounds_ok, "reasons": reasons, "diag": diag, "rd": rd, "live": live,
            "players": players, "deaths": deaths, "plants": plants, "disc_rounds": disc_rounds}


P = pipeline("player_id", "player_id")
S = P["S"]
print(f"kept matches {len({k[0] for k in P['rounds_ok']})}; kept rounds {len(P['rounds_ok'])} "
      f"(log 1521 matches, 28683 rounds)")
print(f"states {S.height} (log 234265); live_start {int((S['kind'] == 'live_start').sum())} (log 28683); "
      f"death {int((S['kind'] == 'death').sum())} (log 205582)")
print(f"diag {dict(P['diag'])}")
print(f"reasons {dict(P['reasons'])}")
print(f"kept rounds by max_rounds: {dict(collections.Counter(H[k[0]]['max_rounds'] for k in P['rounds_ok']))}")
print(f"kept matches by max_rounds: {dict(collections.Counter(H[m]['max_rounds'] for m in {k[0] for k in P['rounds_ok']}))}")
print(f"kept rounds with round > max_rounds (overtime): {sum(1 for k in P['rounds_ok'] if k[1] > H[k[0]]['max_rounds'])}")

# ---------- 4. invariants and independent recomputation on the state table
section("4a. State table invariants")
print(f"alive_t<0 or alive_ct<0: {S.filter((pl.col('alive_t') < 0) | (pl.col('alive_ct') < 0)).height}")
print(f"alive_t>5 or alive_ct>5: {S.filter((pl.col('alive_t') > START) | (pl.col('alive_ct') > START)).height}")
print(f"live_start rows not 5v5: {S.filter((pl.col('kind') == 'live_start') & ((pl.col('alive_t') != START) | (pl.col('alive_ct') != START))).height}")
print(f"live_start rows with seconds_since_round_live != 0: {S.filter((pl.col('kind') == 'live_start') & (pl.col('seconds_since_round_live') != 0)).height}")
print(f"ct_alive_minus_t != alive_ct - alive_t: {S.filter(pl.col('ct_alive_minus_t') != (pl.col('alive_ct') - pl.col('alive_t'))).height}")
print(f"round_number != round: {S.filter(pl.col('round_number') != pl.col('round')).height}")
print(f"bomb_planted=1 but seconds_since_plant<0: {S.filter((pl.col('bomb_planted') == 1) & (pl.col('seconds_since_plant') < 0)).height}")
print(f"bomb_planted=0 but seconds_since_plant != -1: {S.filter((pl.col('bomb_planted') == 0) & (pl.col('seconds_since_plant') != -1.0)).height}")
print(f"seconds_since_round_live quantiles: {quant(S['seconds_since_round_live'].to_list())}; min<0: "
      f"{S.filter(pl.col('seconds_since_round_live') < 0).height}")
print(f"seconds_since_plant quantiles (planted states): "
      f"{quant(S.filter(pl.col('bomb_planted') == 1)['seconds_since_plant'].to_list())}")
print(f"planted share of states: {S['bomb_planted'].mean():.4f}; states with planted=1 and tick<freeze: 0 by construction")
print(f"is_overtime=1 states: {int(S['is_overtime'].sum())}; is_overtime rows with round<=max_rounds: "
      f"{S.filter((pl.col('is_overtime') == 1) & (pl.col('round_number') <= 16)).height} (16-round matches only)")
print(f"is_pistol_round=1 states by round number: "
      f"{dict(sorted(collections.Counter(S.filter(pl.col('is_pistol_round') == 1)['round_number'].to_list()).items()))}")
print(f"label_ct_win share of states: {S['label_ct_win'].mean():.4f}")

# per-round sequence invariant (insertion order): live=10, then each death -1
viol = 0
cur = None
prev = None
for m, r, kind, at, act in S.select(["match_key", "round", "kind", "alive_t", "alive_ct"]).iter_rows():
    key = (m, r)
    if key != cur:
        cur = key
        prev = None
    s = at + act
    if kind == "live_start":
        if s != 2 * START:
            viol += 1
    else:
        if prev is None or s != prev - 1:
            viol += 1
    prev = s
print(f"per-round sum sequence violations (live=10, each death -1, insertion order): {viol}")

meta = pl.DataFrame([{"match_key": k[0], "round": k[1], "freeze": P["live"][k]["tick"],
                      "end": P["rd"][k]["end"]} for k in P["rounds_ok"]],
                    schema={"match_key": pl.String, "round": pl.Int64, "freeze": pl.Int64, "end": pl.Int64})
section("4b. Independent recomputation of alive counts and planted flags (raw pid, as bakeoff)")
dw = (DE.select(["match_key", "round", "tick", "player_id", "player_team_code"])
      .join(meta, on=["match_key", "round"], how="inner")
      .filter((pl.col("tick") >= pl.col("freeze")) & (pl.col("tick") <= pl.col("end"))))
print(f"in-window death rows (before dedupe): {dw.height}; null team in window: "
      f"{dw.filter(pl.col('player_team_code').is_null()).height}")
dwu = dw.sort("tick").unique(subset=["match_key", "round", "player_id"], keep="first", maintain_order=True)
print(f"after first-death-per-pid dedupe: {dwu.height} (bakeoff deaths_in_window 205582)")
dwu = (dwu.sort(["match_key", "round", "tick"])
       .with_columns(
           (START - (pl.col("player_team_code") == T).cast(pl.Int64).cum_sum().over(["match_key", "round"])).alias("ind_alive_t"),
           (START - (pl.col("player_team_code") == CT).cast(pl.Int64).cum_sum().over(["match_key", "round"])).alias("ind_alive_ct")))
ind = collections.defaultdict(list)
for m, r, tk, at, act in dwu.select(["match_key", "round", "tick", "ind_alive_t", "ind_alive_ct"]).iter_rows():
    ind[(m, r)].append((tk, at, act))
sdeath = collections.defaultdict(list)
for m, r, tk, kind, at, act in S.select(["match_key", "round", "tick", "kind", "alive_t", "alive_ct"]).iter_rows():
    if kind == "death":
        sdeath[(m, r)].append((tk, at, act))
mism_rounds = 0
mism_ex = []
for k in P["rounds_ok"]:
    a = sorted(sdeath.get(k, []))
    b = sorted(ind.get(k, []))
    # order-independent: same ticks; same min alive per side (final counts)
    ta = [x[0] for x in a]
    tb = [x[0] for x in b]
    fa = (min((x[1] for x in a), default=START), min((x[2] for x in a), default=START))
    fb = (min((x[1] for x in b), default=START), min((x[2] for x in b), default=START))
    if ta != tb or fa != fb:
        mism_rounds += 1
        if len(mism_ex) < 3:
            mism_ex.append((k[0][:30], k[1], ta[:6], tb[:6], fa, fb))
print(f"rounds where S death ticks or final alive counts differ from independent recompute: {mism_rounds} "
      f"of {len(P['rounds_ok'])}; examples {mism_ex}")

pl_first = (BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round", "tick"])
            .join(meta, on=["match_key", "round"], how="inner")
            .filter((pl.col("tick") >= pl.col("freeze")) & (pl.col("tick") <= pl.col("end")))
            .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("plant_tick")))
SS = S.join(pl_first, on=["match_key", "round"], how="left").with_columns(
    ((pl.col("plant_tick").is_not_null()) & (pl.col("plant_tick") <= pl.col("tick"))).cast(pl.Int64).alias("exp_planted"),
    pl.when((pl.col("plant_tick").is_not_null()) & (pl.col("plant_tick") <= pl.col("tick")))
    .then((pl.col("tick") - pl.col("plant_tick")) / TICK).otherwise(-1.0).alias("exp_ssp"))
print(f"bomb_planted mismatches vs independent: {SS.filter(pl.col('exp_planted') != pl.col('bomb_planted')).height}")
print(f"seconds_since_plant mismatches vs independent: "
      f"{SS.filter((pl.col('exp_ssp') - pl.col('seconds_since_plant')).abs() > 1e-9).height}")
print(f"rounds with a plant in window: {pl_first.height} of {len(P['rounds_ok'])} kept rounds")

section("4c. Final state vs round outcome (kept rounds)")
fin = {}
for m, r, kind, at, act, pla, tk in S.select(["match_key", "round", "kind", "alive_t", "alive_ct",
                                              "bomb_planted", "tick"]).iter_rows():
    k = (m, r)
    s = at + act
    if k not in fin or s < fin[k][0]:
        fin[k] = (s, at, act, pla, tk, kind)
xt = collections.Counter()
viol_c = collections.Counter()
viol_post = collections.Counter()
anomalies = []
for k, (s, at, act, pla, tk, kind) in fin.items():
    w, rs = rwin[k]
    name = WIN.get(rs, str(rs))
    xt[(("T" if w == T else "CT"), name, f"alive_t0={at == 0}", f"alive_ct0={act == 0}", f"planted_at_final={pla}")] += 1
    in_win_plant = any(P["live"][k]["tick"] <= t <= P["rd"][k]["end"] for t in P["plants"].get(k, []))
    if w == T and rs == 9 and act != 0:
        viol_c["T wins reason 9 but CT alive at final state"] += 1
        if any(side == CT for _, side, _ in post_rows.get(k, [])):
            viol_post["  ...with a post-round_end CT death"] += 1
        if len(anomalies) < 8:
            anomalies.append(k)
    if w == CT and rs == 8 and pla == 1 and at != 0:
        viol_c["CT wins reason 8 with plant, T alive at final"] += 1
        if any(side == T for _, side, _ in post_rows.get(k, [])):
            viol_post["  ...with a post-round_end T death"] += 1
    if w == T and rs == 1 and not in_win_plant:
        viol_c["T wins reason 1 (bombed) but no in-window plant row"] += 1
    if w == CT and rs == 7 and not in_win_plant:
        viol_c["CT wins reason 7 (defused) but no in-window plant row"] += 1
    if rs in (1, 7):
        viol_c[f"reason {rs} kept rounds"] += 1
        viol_c[f"reason {rs} kept rounds with in-window plant row"] += int(in_win_plant)
    if rs == 9:
        viol_c["reason 9 kept rounds"] += 1
        viol_c["reason 9 kept rounds with a plant in window"] += int(in_win_plant)
        viol_c["reason 9 kept rounds with a plant after round_end"] += int(
            any(t > P["rd"][k]["end"] for t in P["plants"].get(k, [])))
print("final-state cross-tab (winner, reason, final alive_t==0, final alive_ct==0, planted at final state):")
for key, n in sorted(xt.items(), key=lambda kv: -kv[1]):
    print(f"   {n:>6}  {key}")
print(f"violations and counts: {dict(viol_c)}; with post-round_end death of the side that should be dead: {dict(viol_post)}")
for k in anomalies:
    print(f"   anomaly {k[0][:40]} round {k[1]} winner/reason {rwin[k]} freeze {P['live'][k]['tick']} "
          f"end {P['rd'][k]['end']} plants {P['plants'].get(k, [])}")
    for d in P["deaths"].get(k, []):
        print(f"      death tick {d['tick']} pid {d['pid']} team {d['team']} weapon {d['weapon']}")
    print(f"      post-end deaths {post_rows.get(k, [])}")

section("4d. Same-tick deaths (deduped victims, kept rounds)")
tie_rounds = 0
tie_mixed = 0
tie_states = 0
for k in P["rounds_ok"]:
    f_, e_ = P["live"][k]["tick"], P["rd"][k]["end"]
    seen = set()
    lst = []
    for d in P["deaths"].get(k, []):
        if not (f_ <= d["tick"] <= e_) or d["pid"] in seen:
            continue
        seen.add(d["pid"])
        lst.append(d)
    cnt_t = collections.Counter(d["tick"] for d in lst)
    dts = [t for t, c in cnt_t.items() if c > 1]
    if dts:
        tie_rounds += 1
    for t in dts:
        tie_states += cnt_t[t]
        if len({d["team"] for d in lst if d["tick"] == t}) > 1:
            tie_mixed += 1
print(f"kept rounds with >=2 deaths at one tick: {tie_rounds}; tick groups with both sides dying: {tie_mixed}; "
      f"death rows in such groups: {tie_states}")

section("4e. Raw vs fixed player id (player_id_fixed) in the same pipeline")
Fx = pipeline("player_id_fixed", "player_id_fixed")
SF = Fx["S"]
print(f"fixed-id pipeline: kept rounds {len(Fx['rounds_ok'])}; states {SF.height}; "
      f"live_start {int((SF['kind'] == 'live_start').sum())}; death {int((SF['kind'] == 'death').sum())}")
print(f"fixed-id diag: {dict(Fx['diag'])}")
print(f"fixed-id reasons: {dict(Fx['reasons'])}")
rawagg = {}
for m, r, kind, at, act in S.select(["match_key", "round", "kind", "alive_t", "alive_ct"]).iter_rows():
    k = (m, r)
    rawagg.setdefault(k, [0, START, START])
    rawagg[k][0] += 1
    rawagg[k][1] = min(rawagg[k][1], at)
    rawagg[k][2] = min(rawagg[k][2], act)
fixagg = {}
for m, r, kind, at, act in SF.select(["match_key", "round", "kind", "alive_t", "alive_ct"]).iter_rows():
    k = (m, r)
    fixagg.setdefault(k, [0, START, START])
    fixagg[k][0] += 1
    fixagg[k][1] = min(fixagg[k][1], at)
    fixagg[k][2] = min(fixagg[k][2], act)
diff_rounds = [k for k in set(rawagg) | set(fixagg) if rawagg.get(k) != fixagg.get(k)]
print(f"rounds whose (state count, min alive_t, min alive_ct) differ raw vs fixed: {len(diff_rounds)}")
for k in diff_rounds[:5]:
    print(f"   {k[0][:30]} r{k[1]} raw {rawagg.get(k)} fixed {fixagg.get(k)}")

section("5. Swap schedule vs score transitions (all matches, all rounds with a next freeze row)")
fs = {}
for m, r, t, ct in frz_rows.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    if t is None or ct is None:
        continue
    fs.setdefault((m, int(r)), (int(t), int(ct)))
tab = collections.defaultdict(collections.Counter)
mismatch_total = 0
disc_total = 0
examples = []
for (m, r), (t, ct) in fs.items():
    nxt = fs.get((m, r + 1))
    w = rwin.get((m, r), (None, None))[0]
    if nxt is None or w is None:
        continue
    mr = H[m]["max_rounds"]
    tw, cw = int(w == T), int(w == CT)
    t_post, ct_post = t + tw, ct + cw
    no_ok = nxt == (t_post, ct_post)
    sw_ok = nxt == (ct_post, t_post)
    if no_ok and sw_ok:
        cat = "ambig"
    elif no_ok:
        cat = "noswap"
        disc_total += 1
    elif sw_ok:
        cat = "swap"
        disc_total += 1
    else:
        cat = "neither"
        disc_total += 1
    sched = bk.swap_expected(r, mr)
    row = tab[(mr, r)]
    row[cat] += 1
    row["sched_swap" if sched else "sched_noswap"] += 1
    if cat in ("noswap", "swap", "neither") and (cat == "swap") != sched:
        row["MISMATCH"] += 1
        mismatch_total += 1
        if len(examples) < 6:
            examples.append((m[:30], r, mr, cat, sched, (t, ct), w, nxt))
print(f"discriminative transitions {disc_total}; schedule-vs-observed mismatches (incl. neither): {mismatch_total}")
print("examples:", examples)
for (mr, r) in sorted(tab):
    if mr == 24 and r > 34:
        continue
    if mr == 16 and r > 20:
        continue
    row = tab[(mr, r)]
    print(f"   max{mr:>2} r{r:>2}: noswap {row['noswap']:>5} swap {row['swap']:>5} ambig {row['ambig']:>5} "
          f"neither {row['neither']:>3} | schedule {'SWAP' if row['sched_swap'] else 'none'} "
          f"| mismatches {row['MISMATCH']}")

section("Done")
print(f"run time {time.time() - t0:.1f}s")
