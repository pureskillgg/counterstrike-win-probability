"""VERIFIER for the GRENADE feasibility report (TASK B). Read-only on the tome and data\\state_sample.

Independent recomputation in plain Python (does not reuse feas_grenade_events.py logic):
  S0 win-reason mapping sources, S1 full-tome schema counts, S2 round windows and labels,
  S3 smoke list, S4 HE impacts, S5 molotov impacts, S6 bomb events vs smoke,
  S7 seeded (random.Random(0)) recompute of 5 items, S8 impossible-value sweep.
Stdout only. The caller redirects it to logs\\feas\\verify_grenade.txt.
Run from repo root: pixi run python -I -X utf8 scripts\\eda\\verify_feas_grenade.py
"""
import ast
import csv
import glob
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
FEAS = ROOT / "logs" / "feas"
TICK = 64
T, CT = 2, 3
HE_WIN = 1 * TICK          # report ASSUMPTION: 1 s after HE detonation
FIRE_WIN = 3 * TICK        # report ASSUMPTION: 3 s after fire start
FIRE_LONG = 7 * TICK       # report sensitivity
HE_LEAD = 16               # report sensitivity: 0.25 s before detonation
SMOKE_FB = 18 * TICK       # report ASSUMPTION (task spec)
OBS_LIFE = 1412            # observed smoke life in report (ticks)
RADIUS = 150.0             # report ASSUMPTION: 2D radius, game units
HE_W = "hegrenade"
FIRE_W = {"inferno", "molotov", "incendiary"}

RESULTS = []
BUCKETS = {}
RESET = {}   # (match, round) -> tick of round_state round_officially_ended (same round), set in main


def check(label, report_val, got):
    ok = report_val == got
    RESULTS.append((label, ok))
    print(f"  {'MATCH   ' if ok else 'MISMATCH'} {label} | report={report_val} | recomputed={got}")
    return ok


def note(label, ok, detail=""):
    RESULTS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else 'FLAG'} {label}" + (f" | {detail}" if detail else ""))
    return ok


def folder(ch):
    hits = sorted(p for p in TOME.iterdir() if p.is_dir() and p.name.startswith(f"basic_{ch}."))
    if len(hits) != 1:
        raise SystemExit(f"{ch}: expected one folder, got {[h.name for h in hits]}")
    return hits[0]


def load(ch, cols):
    files = sorted(glob.glob(str(folder(ch) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def to_int(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return int(x)


def pct(vals, q):
    s = sorted(vals)
    if not s:
        return float("nan")
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def buckets(vals):
    c = Counter()
    for v in vals:
        c["0" if v == 0 else "1" if v == 1 else "2" if v == 2 else ">2"] += 1
    return (c["0"], c["1"], c["2"], c[">2"])


def literal_wr(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "WIN_REASON"
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    return None


# ---------------------------------------------------------------- S0 / S1
def s0_win_reason():
    print("\n== S0 win-reason mapping source")
    a = literal_wr(ROOT / "scripts" / "eda" / "batch_effect.py")
    b = literal_wr(ROOT / "scripts" / "model" / "feature_validation.py")
    print(f"  batch_effect.py WIN_REASON = {a}")
    note("WIN_REASON identical in scripts\\eda\\batch_effect.py and scripts\\model\\feature_validation.py", a == b)
    src = (ROOT / "scripts" / "eda" / "feas_grenade_events.py").read_text(encoding="utf-8")
    uses = [ln.strip() for ln in src.splitlines() if "win_reason" in ln.lower() or "WIN_REASON" in ln]
    print(f"  feas_grenade_events.py lines mentioning win_reason/WIN_REASON: {uses}")
    note("grenade report script uses no win-reason mapping (mapping not exercised)", len(uses) == 0)


def s1_full_tome():
    print("\n== S1 full-tome counts (report section 1)")
    gs = load("grenade_state", ["event_type", "grenade_id", "tick_throw", "player_x_pos"])
    check("grenade_state rows", 531031, gs.height)
    cnt = dict(gs.group_by("event_type").len().iter_rows())
    for k, v in {"flashbang_detonate": 138846, "smokegrenade_detonate": 138489,
                 "smokegrenade_expired": 126806, "hegrenade_detonate": 119201,
                 "decoy_started": 3920, "decoy_detonate": 3769}.items():
        check(f"grenade_state[{k}]", v, cnt.get(k, 0))
    check("grenade_state grenade_id null % (all)", "14.0",
          f"{100 * gs['grenade_id'].null_count() / gs.height:.1f}")
    for et, rep in [("smokegrenade_detonate", "13.5"), ("hegrenade_detonate", "14.3")]:
        sub = gs.filter(pl.col("event_type") == et)
        check(f"grenade_id null % [{et}]", rep, f"{100 * sub['grenade_id'].null_count() / sub.height:.1f}")
    det = gs.filter(pl.col("event_type").str.ends_with("_detonate"))
    sh = det.filter(pl.col("event_type").is_in(["smokegrenade_detonate", "hegrenade_detonate"]))
    check("tick_throw null % on smoke+HE detonate rows", "0.0",
          f"{100 * sh['tick_throw'].null_count() / sh.height:.1f}")
    check("tick_throw null % overall", "21.2", f"{100 * gs['tick_throw'].null_count() / gs.height:.1f}")
    check("player_x_pos null % overall", "25.4", f"{100 * gs['player_x_pos'].null_count() / gs.height:.1f}")

    ms = load("molotov_state", ["event_type", "burn_duration", "match_key", "round", "entity_id_fixed"])
    check("molotov_state rows", 208908, ms.height)
    mc = dict(ms.group_by("event_type").len().iter_rows())
    check("molotov_state[inferno_startburn]", 104603, mc.get("inferno_startburn", 0))
    check("molotov_state[inferno_expire]", 104305, mc.get("inferno_expire", 0))
    bd = [v for v in ms["burn_duration"].to_list() if v is not None]
    neg = sum(1 for v in bd if v < 0)
    check("burn_duration non-null rows, FULL TOME (report section 1 label)", 9764, len(bd))
    print(f"  burn_duration: n={len(bd)} negative={neg} exactly -1.0={sum(1 for v in bd if v == -1.0)} "
          f"p05={pct(bd, .05)} p50={pct(bd, .5)} p95={pct(bd, .95)}; "
          f"p05 among >=0 only={pct([v for v in bd if v >= 0], .05)}")
    note("burn_duration contains negative sentinel-like values (report lists p05=-1.0 without flagging)",
         neg == 0, f"negative rows={neg}")
    bd_types = ms.filter(pl.col("burn_duration").is_not_null()).group_by("event_type").len().rows()
    print(f"  burn_duration non-null by event_type: {bd_types}")

    mf = load("molotov_fire", ["match_key", "round", "entity_id_fixed"])
    check("molotov_fire rows", 3278394, mf.height)
    mf_k = mf.unique()
    sb_k = ms.filter(pl.col("event_type") == "inferno_startburn").select(
        ["match_key", "round", "entity_id_fixed"]).unique()
    check("molotov_fire distinct keys", 104608, mf_k.height)
    check("startburn distinct keys", 104603, sb_k.height)
    print(f"  molotov_fire keys not in startburn={mf_k.join(sb_k, on=['match_key', 'round', 'entity_id_fixed'], how='anti').height}; "
          f"startburn keys not in molotov_fire={sb_k.join(mf_k, on=['match_key', 'round', 'entity_id_fixed'], how='anti').height}")

    hurt_n = load("player_hurt", ["match_key"]).height
    blind_n = load("player_blind", ["match_key"]).height
    check("player_hurt rows", 1007846, hurt_n)
    check("player_blind rows", 244469, blind_n)
    bs = dict(load("bomb_state", ["event_type"]).group_by("event_type").len().iter_rows())
    ba = dict(load("bomb_action", ["event_type"]).group_by("event_type").len().iter_rows())
    check("bomb_state rows", 25733, sum(bs.values()))
    check("bomb_state bomb_planted", 18306, bs.get("bomb_planted", 0))
    check("bomb_action rows", 163451, sum(ba.values()))
    check("bomb_action bomb_begin_plant", 20948, ba.get("bomb_begin_plant", 0))
    check("bomb_defuse rows", 7148, load("bomb_defuse", ["event_type"]).height)


# ---------------------------------------------------------------- S2
def s2_rounds():
    print("\n== S2 sample rounds and window definitions")
    rounds = pl.read_parquet(ROOT / "data" / "state_sample" / "rounds.parquet")
    check("rounds rows", 2673, rounds.height)
    check("matches", 150, rounds["match_key"].n_unique())
    check("unique (match,round)", 2673, rounds.select(["match_key", "round"]).unique().height)
    note("no null live_start_tick/round_end_tick", rounds.filter(pl.col("live_start_tick").is_null()
                                                               | pl.col("round_end_tick").is_null()).height == 0)
    note("live_start_tick < round_end_tick in every round",
         rounds.filter(pl.col("live_start_tick") >= pl.col("round_end_tick")).height == 0)
    rk_df = rounds.select(["match_key", "round", "live_start_tick", "round_end_tick", "plant_tick", "ct_win"])
    RK = {(r["match_key"], r["round"]): (r["live_start_tick"], r["round_end_tick"], r["plant_tick"], r["ct_win"])
          for r in rk_df.to_dicts()}
    keys = rounds.select(["match_key", "round"]).unique()
    re_ = load("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).join(
        keys, on=["match_key", "round"], how="semi")
    global WR
    WR = {(r["match_key"], r["round"]): r["win_reason_code"] for r in re_.to_dicts()}
    check("round_end rows in sample (one per round)", 2673, re_.height)
    end_mis = sum(1 for r in re_.to_dicts() if RK[(r["match_key"], r["round"])][1] != r["tick"])
    note("rounds.round_end_tick equals round_end.tick", end_mis == 0, f"mismatches={end_mis}")
    ct_mis = sum(1 for r in re_.to_dicts()
                 if RK[(r["match_key"], r["round"])][3] != (1 if r["winner_team_code"] == CT else 0))
    note("rounds.ct_win equals (round_end.winner_team_code == CT)", ct_mis == 0, f"mismatches={ct_mis}")
    print(f"  win_reason_code distribution in sample: "
          f"{sorted(Counter(re_['win_reason_code'].to_list()).items(), key=lambda kv: str(kv[0]))}")
    rs = load("round_state", ["match_key", "round", "tick", "event_type"]).filter(
        pl.col("event_type") == "round_freeze_end").join(keys, on=["match_key", "round"], how="semi")
    fz = {}
    for r in rs.to_dicts():
        k = (r["match_key"], r["round"])
        fz[k] = min(fz.get(k, r["tick"]), r["tick"])
    fz_mis = sum(1 for k, v in RK.items() if fz.get(k) != v[0])
    note("live_start_tick equals first round_freeze_end tick", fz_mis == 0,
         f"mismatches={fz_mis} rounds_with_freeze_end={len(fz)}")
    return RK, keys


# ---------------------------------------------------------------- side map (own team first, then map)
def build_side_map(*row_sets):
    cands = defaultdict(Counter)
    for rows, pairs in row_sets:
        for x in rows:
            for pk, tk in pairs:
                p, t = to_int(x[pk]), to_int(x[tk])
                if p is None or t not in (T, CT):
                    continue
                cands[(x["match_key"], x["round"], p)][t] += 1
    tmap = {k: next(iter(c)) for k, c in cands.items() if len(c) == 1}
    conflicts = sum(1 for c in cands.values() if len(c) > 1)
    return tmap, len(cands), conflicts


def main():
    print("VERIFIER: feas_grenade_events report (TASK B). Independent recompute, read-only.")
    print(f"constants used for recompute: TICK={TICK} HE_WIN={HE_WIN} FIRE_WIN={FIRE_WIN} "
          f"SMOKE_FB={SMOKE_FB} RADIUS={RADIUS} (all ASSUMPTIONS in the report)")
    s0_win_reason()
    s1_full_tome()
    RK, keys = s2_rounds()
    global RESET
    for r in load("round_state", ["match_key", "round", "tick", "event_type"]).join(
            keys, on=["match_key", "round"], how="semi").filter(
            pl.col("event_type") == "round_officially_ended").to_dicts():
        kk = (r["match_key"], r["round"])
        RESET[kk] = min(RESET.get(kk, r["tick"]), r["tick"])
    print(f"  round_officially_ended tick found for {len(RESET)} of 2673 sample rounds")

    def samp(df):
        return df.join(keys, on=["match_key", "round"], how="semi")

    gs_df = samp(load("grenade_state", ["match_key", "round", "tick", "event_type", "entity_id_fixed",
                                        "grenade_id", "x_pos", "y_pos", "z_pos", "player_id_fixed",
                                        "player_team_code", "tick_throw", "second"])).with_columns(
        pl.col("player_id_fixed").cast(pl.Int64, strict=False))
    gs = gs_df.to_dicts()
    ms = samp(load("molotov_state", ["match_key", "round", "tick", "event_type", "entity_id_fixed",
                                     "player_id_fixed", "player_team_code", "was_extinguished_by_smoke",
                                     "burn_duration"])
              ).with_columns(pl.col("player_id_fixed").cast(pl.Int64, strict=False)).to_dicts()
    hurt_raw = samp(load("player_hurt", ["match_key", "round", "tick", "weapon_name", "player_id_fixed",
                                         "player_team_code", "attacker_id_fixed", "attacker_team_code"])).select([
        "match_key", "round", "tick", pl.col("weapon_name").alias("weapon"),
        pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("victim"),
        pl.col("player_team_code").cast(pl.Int64, strict=False).alias("vteam"),
        pl.col("attacker_id_fixed").cast(pl.Int64, strict=False).alias("attacker"),
        pl.col("attacker_team_code").cast(pl.Int64, strict=False).alias("ateam")]).to_dicts()
    blind_raw = samp(load("player_blind", ["match_key", "round", "player_id_fixed", "player_team_code",
                                           "attacker_id_fixed", "attacker_team_code"])).select([
        "match_key", "round",
        pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("victim"),
        pl.col("player_team_code").cast(pl.Int64, strict=False).alias("vteam"),
        pl.col("attacker_id_fixed").cast(pl.Int64, strict=False).alias("attacker"),
        pl.col("attacker_team_code").cast(pl.Int64, strict=False).alias("ateam")]).to_dicts()
    bom = {}
    for ch in ["bomb_action", "bomb_defuse", "bomb_state"]:
        bom[ch] = samp(load(ch, ["match_key", "round", "tick", "event_type", "player_id_fixed",
                                 "player_team_code", "player_x_pos", "player_y_pos", "player_z_pos"])
                       ).with_columns(pl.col("player_id_fixed").cast(pl.Int64, strict=False)).to_dicts()

    tmap, n_keys, n_conf = build_side_map(
        (hurt_raw, [("victim", "vteam"), ("attacker", "ateam")]),
        (blind_raw, [("victim", "vteam"), ("attacker", "ateam")]),
        (gs, [("player_id_fixed", "player_team_code")]),
        (ms, [("player_id_fixed", "player_team_code")]),
        (bom["bomb_action"], [("player_id_fixed", "player_team_code")]),
        (bom["bomb_defuse"], [("player_id_fixed", "player_team_code")]),
        (bom["bomb_state"], [("player_id_fixed", "player_team_code")]))
    print("\n== side map (independent)")
    check("player-round keys with a side candidate", 26416, n_keys)
    check("conflicting player-round keys (excluded)", 57, n_conf)

    def side_of(m, r, pid, own):
        if own in (T, CT):
            return own
        if pid is None:
            return None
        return tmap.get((m, r, pid))

    for h in hurt_raw:
        h["vside"] = side_of(h["match_key"], h["round"], h["victim"], h["vteam"])
        h["aside"] = side_of(h["match_key"], h["round"], h["attacker"], h["ateam"])
    print(f"  hurt rows with own victim team null (fallback map used): "
          f"{sum(1 for h in hurt_raw if h['vteam'] not in (T, CT))} of {len(hurt_raw)}")

    s3_smoke(gs, RK, side_of)
    s4_he(gs, hurt_raw, RK, side_of)
    s5_molotov(ms, hurt_raw, RK, side_of)
    s6_bomb(gs, bom, RK)
    s7_seeded(gs, ms, hurt_raw, RK, side_of)
    s9_tail(gs, ms, RK, keys)
    s8_impossible(RK)

    print("\n== SUMMARY")
    mism = [l for l, ok in RESULTS if not ok]
    print(f"  checks={len(RESULTS)} not-matching={len(mism)}")
    for l in mism:
        print(f"    - {l}")


# ---------------------------------------------------------------- S3 smoke
def s3_smoke(gs, RK, side_of):
    print("\n== S3 smoke list (report section 2)")
    det = [r for r in gs if r["event_type"] == "smokegrenade_detonate"]
    exp = [r for r in gs if r["event_type"] == "smokegrenade_expired"]
    check("smoke detonations", 8289, len(det))
    check("smoke expired rows", 7574, len(exp))
    check("smoke detonate null grenade_id", 5189, sum(r["grenade_id"] is None for r in det))
    check("smoke expired rows with grenade_id", 2835, sum(r["grenade_id"] is not None for r in exp))

    def key(r):
        return (r["match_key"], r["round"], r["entity_id_fixed"])

    dk = Counter(key(r) for r in det)
    ek = Counter(key(r) for r in exp)
    note("smoke detonate keys unique (match,round,entity)", all(v == 1 for v in dk.values()),
         f"dups={sum(v - 1 for v in dk.values() if v > 1)}")
    note("smoke expired keys unique (match,round,entity)", all(v == 1 for v in ek.values()),
         f"dups={sum(v - 1 for v in ek.values() if v > 1)}")
    orphan = sum(1 for k in ek if k not in dk)
    note("expired rows with no matching detonate (entity)", orphan == 0, f"orphans={orphan}")
    exp_by = {}
    for r in exp:
        exp_by.setdefault(key(r), r)
    links = [(d, exp_by.get(key(d))) for d in det]
    linked = [(d, e) for d, e in links if e is not None]
    fb = [d for d, e in links if e is None]
    fb_keys = {key(d) for d in fb}
    life = Counter(e["tick"] - d["tick"] for d, e in linked)
    print(f"  observed life (exp-det, ticks) counter: {dict(life)}")
    BUCKETS["smoke_linked"] = len(linked)
    BUCKETS["life_min"] = min(life)
    note("observed smoke life never negative", min(life) >= 0, f"min={min(life)}")
    check("observed smoke life distribution (ticks)", {OBS_LIFE: 7574}, dict(life))
    check("smoke with entity-linked expiry", 7574, len(linked))
    check("smoke fallback (no expiry row)", 715, len(fb))
    gexp = {}
    for r in exp:
        if r["grenade_id"] is not None:
            gexp.setdefault((r["match_key"], r["round"], r["grenade_id"]), r["tick"])
    both = agree = 0
    for d, e in linked:
        if d["grenade_id"] is None:
            continue
        t = gexp.get((d["match_key"], d["round"], d["grenade_id"]))
        if t is not None:
            both += 1
            agree += int(t == e["tick"])
    check("entity and grenade_id expiry both present", 2835, both)
    check("entity and grenade_id expiry agree", 2835, agree)
    sec = Counter(round(e["second"] - d["second"], 2) for d, e in linked
                  if d["second"] is not None and e["second"] is not None)
    print(f"  game-clock second diff (expired - detonate), top: {sec.most_common(3)}; distinct={len(sec)}")
    fl = [(d["tick"] - d["tick_throw"]) / TICK for d in det if d["tick_throw"] is not None]
    print(f"  flight time detonate - tick_throw (s): n={len(fl)} min={min(fl):.3f} p50={pct(fl, .5):.3f} "
          f"max={max(fl):.3f}; null tick_throw={len(det) - len(fl)}")
    note("no negative flight times", min(fl) >= 0)
    before = [d for d in det if d["tick"] < RK[(d["match_key"], d["round"])][0]]
    after = [d for d in det if d["tick"] > RK[(d["match_key"], d["round"])][1]]
    out_n = len(before) + len(after)
    check("smoke detonations outside [live_start, round_end]", 54, out_n)
    print(f"  of which before live={len(before)} (fallback={sum(key(d) in fb_keys for d in before)}); "
          f"after round_end={len(after)} (fallback={sum(key(d) in fb_keys for d in after)})")
    note("report states how the 54 outside-window smokes are treated (included in counts)", False,
         "report says only '54 detonations fall outside'; they are kept in the 8,289 and in the bomb check")
    fb_late = [(RK[(d["match_key"], d["round"])][1] - d["tick"]) / TICK for d in fb]
    check("fallback smokes with det+18 s after round_end", 715,
          sum(1 for d in fb if d["tick"] + SMOKE_FB > RK[(d["match_key"], d["round"])][1]))
    check("fallback smokes with det+22.06 s after round_end", 715,
          sum(1 for d in fb if d["tick"] + OBS_LIFE > RK[(d["match_key"], d["round"])][1]))
    print(f"  fallback: seconds from detonation to round_end min={min(fb_late):.2f} p50={pct(fb_late, .5):.2f} "
          f"max={max(fb_late):.2f}")
    # folder claim: null grenade_id per match, sample vs full tome
    per_m = defaultdict(lambda: [0, 0])
    for d in det:
        per_m[d["match_key"]][0] += 1
        per_m[d["match_key"]][1] += int(d["grenade_id"] is None)
    rates_s = [n / t for t, n in [(v[0], v[1]) for v in per_m.values()]]
    full = load("grenade_state", ["match_key", "event_type", "grenade_id"]).filter(
        pl.col("event_type") == "smokegrenade_detonate").group_by("match_key").agg(
        pl.len().alias("n"), pl.col("grenade_id").is_null().sum().alias("nn")).to_dicts()
    rates_f = [r["nn"] / r["n"] for r in full if r["n"]]
    def summ(rs):
        return (f"matches={len(rs)} all-null={sum(1 for x in rs if x == 1)} none-null={sum(1 for x in rs if x == 0)} "
                f"mixed={sum(1 for x in rs if 0 < x < 1)} share>=0.5={sum(1 for x in rs if x >= 0.5) / len(rs):.3f}")
    print(f"  grenade_id null rate per match, SAMPLE: {summ(rates_s)}")
    print(f"  grenade_id null rate per match, FULL TOME: {summ(rates_f)}")
    print("  (informational) compare the two share>=0.5 values above against the report's 'folders' claim")
    global SMOKE_ROWS
    SMOKE_ROWS = {"det": det, "links": links, "fb_keys": fb_keys}


# ---------------------------------------------------------------- S4 HE
def enemy_counts(dets, hurt_idx, side_of, win, att_only=False, lead=0):
    out = []
    for d in dets:
        m, r = d["match_key"], d["round"]
        ts = side_of(m, r, d["thrower"], d["tteam"])
        victims = set()
        for h in hurt_idx.get((m, r), []):
            if d["tick"] - lead <= h["tick"] <= d["tick"] + win and ts is not None and h["vside"] is not None:
                if h["vside"] != ts and (not att_only or h["attacker"] == d["thrower"]):
                    victims.add(h["victim"])
        out.append(len(victims))
    return out


def s4_he(gs, hurt_raw, RK, side_of):
    print("\n== S4 HE impacts (report section 3)")
    det = [r for r in gs if r["event_type"] == "hegrenade_detonate"]
    dets = [{"match_key": r["match_key"], "round": r["round"], "tick": r["tick"],
             "thrower": to_int(r["player_id_fixed"]), "tteam": r["player_team_code"],
             "grenade_id": r["grenade_id"], "entity": r["entity_id_fixed"]} for r in det]
    check("HE detonations", 7775, len(dets))
    check("HE detonate null grenade_id", 4854, sum(d["grenade_id"] is None for d in dets))
    check("HE thrower null", 0, sum(d["thrower"] is None for d in dets))
    thr_unknown = sum(1 for d in dets if side_of(d["match_key"], d["round"], d["thrower"], d["tteam"]) is None)
    check("HE thrower side unknown", 0, thr_unknown)
    before = sum(1 for d in dets if d["tick"] < RK[(d["match_key"], d["round"])][0])
    after = sum(1 for d in dets if d["tick"] > RK[(d["match_key"], d["round"])][1])
    check("HE detonations outside live window", 67, before + after)
    dk = Counter((d["match_key"], d["round"], d["entity"]) for d in dets)
    dup = sum(v - 1 for v in dk.values() if v > 1)
    note("HE detonate keys unique (match,round,entity); report never checks this", dup == 0, f"dups={dup}")

    he_h = [h for h in hurt_raw if h["weapon"] == HE_W and h["victim"] is not None]
    check("hegrenade hurt rows", 4419, len(he_h))
    hi = defaultdict(list)
    for h in he_h:
        hi[(h["match_key"], h["round"])].append(h)
    di = defaultdict(list)
    for d in dets:
        di[(d["match_key"], d["round"])].append(d)

    matched = att_eq = multi = 0
    unmatched = []
    for k, hs in hi.items():
        ds = di.get(k, [])
        for h in hs:
            hits = [d for d in ds if d["tick"] <= h["tick"] <= d["tick"] + HE_WIN]
            if hits:
                matched += 1
                multi += int(len(hits) > 1)
                att_eq += int(any(d["thrower"] == h["attacker"] for d in hits))
            else:
                unmatched.append((h, ds))
    check("hegrenade hurt matched to HE det within 1 s", 4226, matched)
    check("matched hurt where attacker == thrower", 4206, att_eq)
    check("hegrenade hurt unmatched", 193, len(he_h) - matched)
    print(f"  hurt rows matched to >1 detonation (attribution ambiguity, not in report): {multi}")
    bins = Counter()
    before_n = 0
    for h, ds in unmatched:
        if not ds:
            bins["no det in round"] += 1
            continue
        near = min(ds, key=lambda d: abs(h["tick"] - d["tick"]))
        diff = h["tick"] - near["tick"]
        if abs(diff) <= 64:
            bins["<=1 s"] += 1
        elif abs(diff) <= 192:
            bins["1-3 s"] += 1
        else:
            bins[">3 s"] += 1
        if -64 <= diff < 0:
            before_n += 1
    print(f"  unmatched nearest-gap bins: {dict(bins)}; hurt 0-1 s BEFORE nearest det: {before_n}")
    check("unmatched: hurt 0-1 s before nearest det", 178, before_n)
    pre = sorted(min(ds, key=lambda d: abs(h["tick"] - d["tick"]))["tick"] - h["tick"]
                 for h, ds in unmatched if ds and -64 <= h["tick"] - min(ds, key=lambda d: abs(h["tick"] - d["tick"]))["tick"] < 0)
    if pre:
        print(f"  ticks (det - hurt) for those {len(pre)} rows: min={pre[0]} p50={pct(pre, .5)} "
              f"p95={pct(pre, .95)} max={pre[-1]}; within 16 ticks (0.25 s)={sum(p <= 16 for p in pre)}")
    BUCKETS["he_pre_within16"] = sum(p <= 16 for p in pre)

    b1 = buckets(enemy_counts(dets, hi, side_of, HE_WIN))
    BUCKETS["he"] = b1
    BUCKETS["he_matched"] = matched
    check("HE enemy-hurt buckets (0,1,2,>2) window 1 s", (4884, 2051, 642, 198), b1)
    print(f"  shares: " + ", ".join(f"{x / len(dets):.4f}" for x in b1))
    b2 = buckets(enemy_counts(dets, hi, side_of, HE_WIN, att_only=True))
    check("HE attacker==thrower buckets", (4973, 2028, 600, 174), b2)
    b3 = buckets(enemy_counts(dets, hi, side_of, HE_WIN, lead=HE_LEAD))
    check("HE sensitivity lead 0.25 s buckets", (4868, 2054, 650, 203), b3)
    b4 = buckets(enemy_counts(dets, hi, side_of, HE_WIN, lead=64))
    print(f"  HE sensitivity lead 1.0 s (not in report; tests the 178 pre-detonation hurts): "
          f"buckets={b4} n={sum(b4)}; share 0={b4[0] / sum(b4):.4f}")
    BUCKETS["he_lead64"] = b4
    s = sum(b1)
    note("HE buckets sum to detonation count", s == len(dets), f"sum={s}")
    # report CSV: keep the per-detonation sample for S7
    global HE_DETS, HE_HI
    HE_DETS, HE_HI = dets, hi


# ---------------------------------------------------------------- S5 molotov
def s5_molotov(ms, hurt_raw, RK, side_of):
    print("\n== S5 molotov / incendiary impacts (report section 4)")
    sb = [r for r in ms if r["event_type"] == "inferno_startburn"]
    mf_s = load("molotov_fire", ["match_key", "round", "entity_id_fixed"]).filter(
        pl.col("match_key").is_in(sorted({k[0] for k in RK}))).to_dicts()
    mf_keys = {(r["match_key"], r["round"], r["entity_id_fixed"]) for r in mf_s if (r["match_key"], r["round"]) in RK}
    starts = [{"match_key": r["match_key"], "round": r["round"], "tick": r["tick"],
               "entity": r["entity_id_fixed"], "thrower": to_int(r["player_id_fixed"]),
               "tteam": r["player_team_code"]} for r in sb]
    check("molotov fire starts", 5868, len(starts))
    sb_keys = {(s["match_key"], s["round"], s["entity"]) for s in starts}
    check("starts with a molotov_fire row (sample)", 5868, len(sb_keys & mf_keys))
    check("molotov_fire keys without a startburn (sample)", 0, len(mf_keys - sb_keys))
    sk = Counter((s["match_key"], s["round"], s["entity"]) for s in starts)
    note("molotov start keys unique", all(v == 1 for v in sk.values()), f"dups={sum(v - 1 for v in sk.values() if v > 1)}")
    check("thrower null", 172, sum(s["thrower"] is None for s in starts))
    side_unknown = sum(1 for s in starts if s["thrower"] is not None
                       and side_of(s["match_key"], s["round"], s["thrower"], s["tteam"]) is None)
    check("thrower id known, side unknown", 416, side_unknown)
    known = [s for s in starts if side_of(s["match_key"], s["round"], s["thrower"], s["tteam"]) is not None]
    check("starts with known thrower side", 5280, len(known))
    before = sum(1 for s in starts if s["tick"] < RK[(s["match_key"], s["round"])][0])
    after = sum(1 for s in starts if s["tick"] > RK[(s["match_key"], s["round"])][1])
    check("starts outside live window", 31, before + after)
    print(f"  startburn rows with was_extinguished_by_smoke=True: "
          f"{sum(1 for r in sb if r['was_extinguished_by_smoke'] is True)} (report does not address)")

    fh = [h for h in hurt_raw if h["weapon"] in FIRE_W and h["victim"] is not None]
    check("fire-weapon hurt rows", 5647, len(fh))
    fi = defaultdict(list)
    for h in fh:
        fi[(h["match_key"], h["round"])].append(h)
    si = defaultdict(list)
    for s in starts:
        si[(s["match_key"], s["round"])].append(s)
    matched = att_eq = 0
    gaps = Counter()
    late_sign = Counter()
    for k, hs in fi.items():
        ss = si.get(k, [])
        for h in hs:
            hits = [s for s in ss if s["tick"] <= h["tick"] <= s["tick"] + FIRE_WIN]
            if hits:
                matched += 1
                att_eq += int(any(s["thrower"] == h["attacker"] for s in hits))
            else:
                if ss:
                    near_s = min(ss, key=lambda s: abs(h["tick"] - s["tick"]))
                    near = abs(h["tick"] - near_s["tick"])
                    if near > FIRE_WIN:
                        gaps[">3 s"] += 1
                        late_sign["after start" if h["tick"] > near_s["tick"] else "before start"] += 1
                    else:
                        gaps["<=3 s"] += 1
                else:
                    gaps["no start in round"] += 1
    check("fire hurt matched within 3 s", 4686, matched)
    check("fire hurt unmatched", 961, len(fh) - matched)
    check("fire matched where attacker == thrower", 3979, att_eq)
    print(f"  fire unmatched gap bins: {dict(gaps)}; sign of >3 s rows: {dict(late_sign)}")
    check("fire unmatched more than 3 s from nearest start", 775, gaps.get(">3 s", 0))
    kn = [s for s in starts if side_of(s["match_key"], s["round"], s["thrower"], s["tteam"]) is not None]
    b1 = buckets(enemy_counts(kn, fi, side_of, FIRE_WIN))
    BUCKETS["mo"] = b1
    BUCKETS["fire_matched"] = matched
    check("molotov enemy buckets, 3 s", (4442, 733, 83, 22), b1)
    b2 = buckets(enemy_counts(kn, fi, side_of, FIRE_WIN, att_only=True))
    check("molotov attacker==thrower buckets", (4567, 635, 64, 14), b2)
    b3 = buckets(enemy_counts(kn, fi, side_of, FIRE_LONG))
    check("molotov 0-7 s buckets", (4290, 854, 102, 34), b3)
    print(f"  shares 3 s: " + ", ".join(f"{x / len(kn):.4f}" for x in b1))


# ---------------------------------------------------------------- S6 bomb
def s6_bomb(gs, bom, RK):
    print("\n== S6 bomb events vs active smoke (report section 5)")
    plants = [{"match_key": r["match_key"], "round": r["round"], "tick": r["tick"], "ex": r["player_x_pos"],
               "ey": r["player_y_pos"], "ez": r["player_z_pos"], "type": "plant"}
              for r in bom["bomb_action"] if r["event_type"] == "bomb_begin_plant"]
    defs = [{"match_key": r["match_key"], "round": r["round"], "tick": r["tick"], "ex": r["player_x_pos"],
             "ey": r["player_y_pos"], "ez": r["player_z_pos"], "type": "defuse"}
            for r in bom["bomb_defuse"] if r["event_type"] == "bomb_begin_defuse"]
    check("plant events (bomb_begin_plant)", 1568, len(plants))
    check("defuse events (bomb_begin_defuse)", 529, len(defs))
    evs = plants + defs
    note("no null acting-player position", all(e["ex"] is not None for e in evs),
         f"null={sum(e['ex'] is None for e in evs)}")
    after = [e for e in evs if e["tick"] > RK[(e["match_key"], e["round"])][1]]
    before = [e for e in evs if e["tick"] < RK[(e["match_key"], e["round"])][0]]
    check("plant events after round_end_tick", 118, sum(1 for e in after if e["type"] == "plant"))
    check("defuse events after round_end_tick", 0, sum(1 for e in after if e["type"] == "defuse"))
    check("events before live_start_tick", 0, len(before))
    gaps = sorted((e["tick"] - RK[(e["match_key"], e["round"])][1]) / TICK for e in after)
    if gaps:
        print(f"  plant after round_end: seconds past round_end min={gaps[0]:.2f} p50={pct(gaps, .5):.2f} "
              f"p95={pct(gaps, .95):.2f} max={gaps[-1]:.2f}; within 5 s={sum(g <= 5 for g in gaps)}; "
              f"within 30 s={sum(g <= 30 for g in gaps)}")
    links = SMOKE_ROWS["links"]
    by_round = defaultdict(list)
    for d, e in links:
        by_round[(d["match_key"], d["round"])].append((d["tick"], e["tick"] if e else None,
                                                       d["x_pos"], d["y_pos"], d["z_pos"]))

    def run(mode):
        def fb_end(dt, k):
            if mode == "primary":
                return dt + SMOKE_FB
            if mode == "obs":
                return dt + OBS_LIFE
            if mode == "reset":
                return RESET.get(k, RK[k][1])
            return RK[k][1]
        res = {"defuse": Counter(), "plant": Counter()}
        per_ev = []
        for e in evs:
            k = (e["match_key"], e["round"])
            act = n2 = n3 = nx = False
            for dt, et, sx, sy, sz in by_round.get(k, []):
                exact = et is not None
                ex_t = et if exact else fb_end(dt, k)
                if dt <= e["tick"] <= ex_t:
                    act = True
                    d2 = math.hypot(e["ex"] - sx, e["ey"] - sy)
                    d3 = math.sqrt((e["ex"] - sx) ** 2 + (e["ey"] - sy) ** 2 + (e["ez"] - sz) ** 2)
                    n2 = n2 or d2 <= RADIUS
                    n3 = n3 or d3 <= RADIUS
                    nx = nx or (d2 <= RADIUS and exact)
            per_ev.append((e, act, n2, n3, nx))
            res[e["type"]]["active"] += int(act)
            res[e["type"]]["2d"] += int(n2)
            res[e["type"]]["3d"] += int(n3)
            res[e["type"]]["exact2d"] += int(nx)
        after_hit = sum(1 for e, a, n2, n3, nx in per_ev if n2 and e["tick"] > RK[(e["match_key"], e["round"])][1])
        return res, after_hit, per_ev

    res, after_hit, per_ev = run("primary")
    d, p = res["defuse"], res["plant"]
    check("PRIMARY defuse active (time only)", 162, d["active"])
    check("PRIMARY plant active (time only)", 951, p["active"])
    check("PRIMARY total active (time only)", 1113, d["active"] + p["active"])
    check("PRIMARY defuse 2D<=150", 30, d["2d"])
    check("PRIMARY plant 2D<=150", 56, p["2d"])
    check("PRIMARY total 2D<=150", 86, d["2d"] + p["2d"])
    check("PRIMARY total 3D<=150", 81, d["3d"] + p["3d"])
    check("PRIMARY total exact-expiry 2D", 63, d["exact2d"] + p["exact2d"])
    check("PRIMARY events 2D hit with tick after round_end", 2, after_hit)
    r_obs, a_obs, _ = run("obs")
    check("SENS observed-life: plant active", 952, r_obs["plant"]["active"])
    check("SENS observed-life: total 2D", 86, r_obs["defuse"]["2d"] + r_obs["plant"]["2d"])
    r_end, a_end, _ = run("end")
    check("SENS round_end: plant active", 918, r_end["plant"]["active"])
    check("SENS round_end: total 2D", 85, r_end["defuse"]["2d"] + r_end["plant"]["2d"])
    check("SENS round_end: 2D hit after round_end", 1, a_end)
    r_rst, a_rst, _ = run("reset")
    print(f"  SENS reset (fallback end = same-round round_officially_ended): defuse active={r_rst['defuse']['active']} "
          f"plant active={r_rst['plant']['active']} total active={r_rst['defuse']['active'] + r_rst['plant']['active']}; "
          f"total 2D={r_rst['defuse']['2d'] + r_rst['plant']['2d']}; 2D hit after round_end={a_rst}")
    post_reset = sum(1 for e in evs if e["tick"] > RESET.get((e["match_key"], e["round"]), 10**12))
    note("bomb events after same-round round_officially_ended reset", post_reset == 0, f"n={post_reset}")

    # timing of the logged event vs the completion events (data-derived check on the begin-event choice)
    bs = {}
    for r in bom["bomb_state"]:
        if r["event_type"] in ("bomb_planted", "bomb_defused"):
            bs.setdefault((r["match_key"], r["round"], r["event_type"]), []).append(r["tick"])
    pg = []
    for e in plants:
        t = bs.get((e["match_key"], e["round"], "bomb_planted"))
        if t:
            pg.append((min(t) - e["tick"]) / TICK)
    dg = []
    for e in defs:
        t = bs.get((e["match_key"], e["round"], "bomb_defused"))
        if t:
            dg.append((min(t) - e["tick"]) / TICK)
    if pg:
        print(f"  bomb_planted minus bomb_begin_plant (s), first per round: n={len(pg)} p05={pct(pg, .05):.2f} "
              f"p50={pct(pg, .5):.2f} p95={pct(pg, .95):.2f}")
    if dg:
        print(f"  bomb_defused minus bomb_begin_defuse (s): n={len(dg)} p05={pct(dg, .05):.2f} "
              f"p50={pct(dg, .5):.2f} p95={pct(dg, .95):.2f}")
    note("report states the begin-to-completion lag of the plant/defuse events it counts", False,
         "report uses bomb_begin_plant / bomb_begin_defuse only; lag shown above")
    global BOMB_RES
    BOMB_RES = per_ev


# ---------------------------------------------------------------- S7 seeded items
def read_csv(name):
    with open(FEAS / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def s7_seeded(gs, ms, hurt_raw, RK, side_of):
    print("\n== S7 seeded recompute: random.Random(0), one item per output list")
    pops = {"smoke": read_csv("feas_smokes_sample.csv"),
            "he": read_csv("feas_he_sample.csv"),
            "molotov": read_csv("feas_molotov_starts_sample.csv")}
    bomb_rows = read_csv("feas_bomb_smoke_check.csv")
    pops["plant"] = [r for r in bomb_rows if r["ev_type"] == "plant"]
    pops["defuse"] = [r for r in bomb_rows if r["ev_type"] == "defuse"]
    print("  population sizes (from report CSVs): " + ", ".join(f"{k}={len(v)}" for k, v in pops.items()))
    check("CSV rows smoke list", 8289, len(pops["smoke"]))
    check("CSV rows HE list", 7775, len(pops["he"]))
    check("CSV rows molotov list", 5868, len(pops["molotov"]))
    rng = random.Random(0)
    picks = {}
    for cat in ["smoke", "he", "molotov", "plant", "defuse"]:
        picks[cat] = rng.randrange(len(pops[cat]))
    print(f"  picks (0-based index): {picks}")

    smoke_row = pops["smoke"][picks["smoke"]]
    m, r = smoke_row["match_key"], int(smoke_row["round"])
    ent = int(smoke_row["entity_id_fixed"])
    det_t = int(smoke_row["det_tick"])
    dets = SMOKE_ROWS["det"]
    mine = [d for d in dets if d["match_key"] == m and d["round"] == r and d["entity_id_fixed"] == ent
            and d["tick"] == det_t]
    print(f"  ITEM smoke: csv={smoke_row}")
    if mine:
        d = mine[0]
        e = next((x for dd, x in SMOKE_ROWS["links"] if dd is d and x is not None), None)
        exp_t = e["tick"] if e else d["tick"] + SMOKE_FB
        src = "expired_row_entity" if e else "fallback_18s"
        ok = (abs(d["x_pos"] - float(smoke_row["x_pos"])) < 1e-6 and int(smoke_row["exp_final"]) == exp_t
              and smoke_row["exp_source"] == src)
        print(f"    raw: det_tick={d['tick']} x={d['x_pos']:.3f} y={d['y_pos']:.3f} exp_tick={exp_t} src={src}")
        note("ITEM smoke recompute matches CSV (position, exp_final, exp_source)", ok)
    else:
        note("ITEM smoke found in raw", False)

    he_row = pops["he"][picks["he"]]
    m, r = he_row["match_key"], int(he_row["round"])
    det_t = int(he_row["det_tick"])
    ent = int(he_row["entity_id_fixed"])
    hd = [d for d in HE_DETS if d["match_key"] == m and d["round"] == r and d["tick"] == det_t
          and d["entity"] == ent]
    print(f"  ITEM HE: csv={he_row}")
    if hd:
        d = hd[0]
        n1 = enemy_counts([d], HE_HI, side_of, HE_WIN)[0]
        na = enemy_counts([d], HE_HI, side_of, HE_WIN, att_only=True)[0]
        ts = side_of(m, r, d["thrower"], d["tteam"])
        print(f"    raw: thrower={d['thrower']} thrower_side={ts} recomputed n_enemy_1s={n1} att_only={na}")
        note("ITEM HE recompute matches CSV (n_enemy_1s, n_enemy_1s_att_thrower)",
             n1 == int(he_row["n_enemy_1s"]) and na == int(he_row["n_enemy_1s_att_thrower"]))
    else:
        note("ITEM HE found in raw", False)

    mo_row = pops["molotov"][picks["molotov"]]
    m, r = mo_row["match_key"], int(mo_row["round"])
    st_t = int(mo_row["start_tick"])
    ent = int(mo_row["entity_id_fixed"])
    print(f"  ITEM molotov: csv={mo_row}")
    sb = [x for x in ms if x["event_type"] == "inferno_startburn" and x["match_key"] == m
          and x["round"] == r and x["tick"] == st_t and x["entity_id_fixed"] == ent]
    if sb:
        x = sb[0]
        thr = to_int(x["player_id_fixed"])
        s_fake = {"match_key": m, "round": r, "tick": st_t, "thrower": thr, "tteam": x["player_team_code"]}
        ts = side_of(m, r, thr, x["player_team_code"])
        hi = defaultdict(list)
        for h in hurt_raw:
            if h["weapon"] in FIRE_W and h["victim"] is not None:
                hi[(h["match_key"], h["round"])].append(h)
        n3 = enemy_counts([s_fake], hi, side_of, FIRE_WIN)[0] if ts is not None else None
        na = enemy_counts([s_fake], hi, side_of, FIRE_WIN, att_only=True)[0] if ts is not None else None
        print(f"    raw: thrower={thr} thrower_side={ts} recomputed n_enemy_3s={n3} att_only={na}")
        note("ITEM molotov recompute matches CSV (n_enemy_3s, att_only)",
             (n3 == (int(mo_row["n_enemy_3s"]) if mo_row["n_enemy_3s"] != "" else None))
             and (na == (int(mo_row["n_enemy_3s_att_thrower"]) if mo_row["n_enemy_3s_att_thrower"] != "" else None)))
    else:
        note("ITEM molotov found in raw", False)

    for cat in ["plant", "defuse"]:
        row = pops[cat][picks[cat]]
        print(f"  ITEM {cat}: csv={row}")
        m, r = row["match_key"], int(row["round"])
        tk = int(row["ev_tick"])
        ev = [e for e in BOMB_RES if e[0]["match_key"] == m and e[0]["round"] == r and e[0]["tick"] == tk
              and e[0]["type"] == cat]
        if not ev:
            note(f"ITEM {cat} found in raw", False)
            continue
        e, act, n2, n3, nx = ev[0]
        print(f"    raw: active={act} 2D<=150={n2} 3D<=150={n3} exact2D={nx}")
        note(f"ITEM {cat} recompute matches CSV (any_active, any_near2d)",
             str(act).lower() == row["any_active"].lower() and str(n2).lower() == row["any_near2d"].lower())


# ---------------------------------------------------------------- S9 tail / scope diagnostics
def s9_tail(gs, ms, RK, keys):
    print("\n== S9 diagnostics (scope of burn_duration, round tail, flight times, next-round overlap)")
    bd_s = [r["burn_duration"] for r in ms if r["burn_duration"] is not None]
    check("burn_duration non-null rows, SAMPLE (report's 9,764 figure)", 9764, len(bd_s))
    print(f"  sample burn_duration negative rows={sum(1 for v in bd_s if v < 0)}")
    for et in sorted({r["event_type"] for r in gs if r["event_type"].endswith("_detonate")}):
        rows = [r for r in gs if r["event_type"] == et]
        print(f"  tick_throw null on {et}: {sum(r['tick_throw'] is None for r in rows)} of {len(rows)}")
    for et in ["smokegrenade_detonate", "hegrenade_detonate"]:
        fl = [(r["tick"] - r["tick_throw"]) / TICK for r in gs
              if r["event_type"] == et and r["tick_throw"] is not None]
        print(f"  {et} flight time (s): n={len(fl)} p50={pct(fl, .5):.2f} p95={pct(fl, .95):.2f} "
              f"max={max(fl):.1f}; >10 s={sum(1 for f in fl if f > 10)}; >60 s={sum(1 for f in fl if f > 60)}")
    links = SMOKE_ROWS["links"]
    late_linked = sum(1 for d, e in links if e is not None
                      and e["tick"] > RK[(d["match_key"], d["round"])][1])
    check("linked smoke expiry rows after round_end (report's 'rounds ended before expiry' explanation implies 0)",
          0, late_linked)

    def end_of(k):
        return RK[k][1]

    def in_next(k, t):
        nk = (k[0], k[1] + 1)
        if nk not in RK:
            return None
        return RK[nk][0] <= t <= RK[nk][1]

    sm_after = [d for d in SMOKE_ROWS["det"] if d["tick"] > end_of((d["match_key"], d["round"]))]
    secs = sorted((d["tick"] - end_of((d["match_key"], d["round"]))) / TICK for d in sm_after)
    print(f"  smoke detonations after round_end: n={len(secs)} seconds past end min={secs[0]:.2f} "
          f"p50={pct(secs, .5):.2f} max={secs[-1]:.2f}")
    print(f"  win_reason of those rounds: {dict(Counter(WR[(d['match_key'], d['round'])] for d in sm_after))}; "
          f"inside NEXT round's live window: "
          f"{sum(1 for d in sm_after if in_next((d['match_key'], d['round']), d['tick']))}")
    pl_after = [e for e, a, n2, n3, nx in BOMB_RES if e["type"] == "plant"
                and e["tick"] > end_of((e["match_key"], e["round"]))]
    psecs = sorted((e["tick"] - end_of((e["match_key"], e["round"]))) / TICK for e in pl_after)
    print(f"  plant begin events after round_end: n={len(psecs)} seconds past end min={psecs[0]:.2f} "
          f"p50={pct(psecs, .5):.2f} max={psecs[-1]:.2f}")
    print(f"  win_reason of those rounds: {dict(Counter(WR[(e['match_key'], e['round'])] for e in pl_after))}; "
          f"inside NEXT round's live window: "
          f"{sum(1 for e in pl_after if in_next((e['match_key'], e['round']), e['tick']))}")
    print(f"  win_reason=1 (Target_Bombed) rounds with a plant begin after round_end: "
          f"{sum(1 for e in pl_after if WR[(e['match_key'], e['round'])] == 1)}")
    gaps_next = sorted((RK[(k[0], k[1] + 1)][0] - RK[k][1]) / TICK for k in RK if (k[0], k[1] + 1) in RK)
    print(f"  gap from round_end to NEXT round live_start (s): n={len(gaps_next)} min={gaps_next[0]:.2f} "
          f"p05={pct(gaps_next, .05):.2f} p50={pct(gaps_next, .5):.2f}; negative={sum(1 for g in gaps_next if g < 0)}")

    # Why are 715 smoke expiry rows missing? Test whether a round reset truncates the expiry event.
    print("  -- missing-expiry mechanism test (candidate reset ticks from round_state)")
    rs_rows = load("round_state", ["match_key", "round", "tick", "event_type"]).join(
        keys, on=["match_key", "round"], how="semi").to_dicts()
    sm_after_reset = [d for d in SMOKE_ROWS["det"] if d["tick"] > RESET.get((d["match_key"], d["round"]), 10**12)]
    print(f"  smoke detonations after same-round round_officially_ended: {len(sm_after_reset)}")
    off = sorted((RESET[k] - RK[k][1]) / TICK for k in RESET if k in RK)
    print(f"  round_officially_ended minus round_end (s), per round: n={len(off)} min={off[0]:.2f} "
          f"p05={pct(off, .05):.2f} p50={pct(off, .5):.2f} p95={pct(off, .95):.2f} max={off[-1]:.2f}")
    print(f"  round_state event types in sample rounds: "
          f"{sorted(Counter(r['event_type'] for r in rs_rows).items())}")
    lk = [(d, e) for d, e in SMOKE_ROWS["links"] if e is not None]
    fb = [d for d, e in SMOKE_ROWS["links"] if e is None]
    print(f"  linked smokes whose expiry is after round_end: {sum(1 for d, e in lk if e['tick'] > RK[(d['match_key'], d['round'])][1])}"
          f" (min over those of expiry-end, s: "
          f"{min(((e['tick'] - RK[(d['match_key'], d['round'])][1]) / TICK for d, e in lk if e['tick'] > RK[(d['match_key'], d['round'])][1]), default=float('nan')):.2f}, "
          f"max: {max(((e['tick'] - RK[(d['match_key'], d['round'])][1]) / TICK for d, e in lk if e['tick'] > RK[(d['match_key'], d['round'])][1]), default=float('nan')):.2f})")
    print(f"  fallback smokes: det+22.06 s minus round_end, s: min={min((d['tick'] + OBS_LIFE - RK[(d['match_key'], d['round'])][1]) / TICK for d in fb):.2f} "
          f"max={max((d['tick'] + OBS_LIFE - RK[(d['match_key'], d['round'])][1]) / TICK for d in fb):.2f}")
    for ev_name, mode in [("round_officially_ended", "same"), ("round_prestart", "same"),
                          ("round_prestart", "next"), ("round_start", "next"), ("cs_pre_restart", "same")]:
        fd = {}
        for r in rs_rows:
            if r["event_type"] == ev_name:
                kk = (r["match_key"], r["round"])
                fd[kk] = min(fd.get(kk, r["tick"]), r["tick"])

        def reset_of(k, fd=fd, mode=mode):
            return fd.get(k if mode == "same" else (k[0], k[1] + 1))

        lk_eval = [(e["tick"], reset_of((d["match_key"], d["round"]))) for d, e in lk]
        lk_eval = [(x, y) for x, y in lk_eval if y is not None]
        fb_eval = [(d["tick"] + OBS_LIFE, reset_of((d["match_key"], d["round"]))) for d in fb]
        fb_eval = [(x, y) for x, y in fb_eval if y is not None]
        lk_after = sum(1 for x, y in lk_eval if x > y)
        fb_after = sum(1 for x, y in fb_eval if x > y)
        lk_max_margin = max(((x - y) / TICK for x, y in lk_eval), default=float("nan"))
        fb_min_margin = min(((x - y) / TICK for x, y in fb_eval), default=float("nan"))
        print(f"  candidate {ev_name}/{mode}: rounds with it={len(fd)}; linked expiry after it={lk_after} "
              f"of {len(lk_eval)} (largest margin exp-reset, s={lk_max_margin:.2f}); "
              f"fallback det+22.06 s after it={fb_after} of {len(fb_eval)} "
              f"(smallest margin, s={fb_min_margin:.2f})")


# ---------------------------------------------------------------- S8 impossible values
def s8_impossible(RK):
    print("\n== S8 impossible-value sweep (counts from S3-S7 recomputes, not hard-coded)")
    he_b = BUCKETS["he"]
    mo_b = BUCKETS["mo"]
    note("HE buckets sum to 7775 detonations", sum(he_b) == 7775, f"sum={sum(he_b)}")
    note("molotov buckets sum to 5280 known-side starts", sum(mo_b) == 5280, f"sum={sum(mo_b)}")
    note("no bucket share above 1 (HE and molotov)",
         all(x / sum(he_b) <= 1 and x / sum(mo_b) <= 1 for x in list(he_b) + list(mo_b)))
    note("matched hurt rows <= hurt rows (HE 4226 of 4419; fire 4686 of 5647)",
         BUCKETS["he_matched"] <= 4419 and BUCKETS["fire_matched"] <= 5647)
    note("expired (linked) rows <= smoke detonations (7574 <= 8289)", BUCKETS["smoke_linked"] <= 8289)
    note("smoke life min >= 0 ticks", BUCKETS["life_min"] >= 0, f"min={BUCKETS['life_min']}")
    note("molotov 5868 - 172 (no thrower) - 416 (no side) = 5280", 5868 - 172 - 416 == 5280)


if __name__ == "__main__":
    main()
