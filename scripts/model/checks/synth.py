"""A tiny synthetic tome store with known ground truth, for the selftests: build it, run scripts/data/prepare.py and make_grid.py on it, and
compare the outputs with what the generator knows (winners, used matches, grid rows). Seconds, no real data.

  truth = write_tomes(root, store="syn")      # root/tomes/<store>/tome/csds/...
  run_pipeline(root, store="syn")             # prepare.main() + make_grid.main() with their ROOT pointed at `root`
"""
import contextlib
import io
import os
import sys

import polars as pl

S = 64  # ticks per second
T, CT = 2, 3
MAX_ROUNDS = 24
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")


def swap_after(r, mr=MAX_ROUNDS):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def round_events(kind, F):
    """-> winner side, recorded end tick, deaths [(tick, side)], bomb events [(type, tick)], for a round whose freeze ends at tick F."""
    if kind == "explode":
        plant = F + 50 * S
        expl = plant + 40 * S
        return T, expl + 3 * S, [(F + 20 * S, CT), (F + 35 * S, CT), (F + 30 * S, T)], [("bomb_planted", plant), ("bomb_exploded", expl)]
    if kind == "defuse":
        plant = F + 45 * S
        dfz = plant + 20 * S
        return CT, dfz + 2 * S, [(F + 25 * S, T), (F + 38 * S, T), (F + 50 * S, T), (F + 30 * S, CT)], [("bomb_planted", plant), ("bomb_defused", dfz)]
    if kind in ("ct_wipe", "ct_wipe_recwrong"):
        ct = [(F + (20 + 10 * i) * S, CT) for i in range(5)]
        return T, ct[-1][0] + 4 * S, ct + [(F + 25 * S, T)], []
    if kind in ("t_wipe", "t_wipe_recwrong"):
        t = [(F + (20 + 10 * i) * S, T) for i in range(5)]
        return CT, t[-1][0] + 4 * S, t + [(F + 35 * S, CT), (F + 45 * S, CT)], []
    if kind == "time":
        return CT, F + 115 * S, [(F + 30 * S, CT), (F + 31 * S, CT), (F + 40 * S, T), (F + 41 * S, T)], []
    if kind == "defuse_tw":  # all T dead after the plant, CT still has to defuse
        plant = F + 30 * S
        dfz = F + 75 * S
        return CT, dfz + 2 * S, [(F + (35 + 5 * i) * S, T) for i in range(5)] + [(F + 32 * S, CT)], [("bomb_planted", plant), ("bomb_defused", dfz)]
    if kind == "explode_ctwipe":  # all CT dead before the blast
        plant = F + 40 * S
        expl = plant + 40 * S
        return T, expl + 3 * S, [(F + (45 + 5 * i) * S, CT) for i in range(5)] + [(F + 50 * S, T)], [("bomb_planted", plant), ("bomb_exploded", expl)]
    raise ValueError(kind)


KINDS = ["explode", "defuse", "ct_wipe", "t_wipe", "time", "defuse_tw", "explode_ctwipe", "ct_wipe_recwrong", "t_wipe_recwrong"]
# Known answers of the generator (seconds after the freeze end): the grid has one row per whole second strictly BEFORE the decisive event
# (blast, defuse, or the 5th death of a side), and an unplanted round that runs out stops after 115 rows.
EXPECTED_ROWS = {"explode": 90, "defuse": 65, "ct_wipe": 60, "t_wipe": 60, "time": 115, "defuse_tw": 75, "explode_ctwipe": 65,
                 "ct_wipe_recwrong": 60, "t_wipe_recwrong": 60}
PLANT_SEC = {"explode": 50, "defuse": 45, "defuse_tw": 30, "explode_ctwipe": 40}  # a plant exactly on a grid tick (whole second)

# (key suffix, era/day, map, build, wingman, n_rounds, first_round, regime_a, kinds)
def _cycle(n, shift=0):
    return [KINDS[(i + shift) % len(KINDS)] for i in range(n)]


MATCHES = [
    dict(key="csds/2026/09/01/aaaa0001/csds", map="de_mirage", build=10896, wingman=False, n=22, r0=1, regime_a=False, kinds=_cycle(22), swapped_header=False),
    dict(key="csds/2026/09/01/aaaa0002/csds", map="de_inferno", build=10896, wingman=False, n=20, r0=1, regime_a=False, kinds=_cycle(20, 3), swapped_header=True),
    dict(key="csds/2026/09/02/aaaa0003/csds", map="de_dust2", build=10924, wingman=False, n=24, r0=1, regime_a=False, kinds=_cycle(24, 5), swapped_header=False),
    dict(key="csds/2026/09/02/aaaa0004/csds", map="cs_office", build=10896, wingman=False, n=16, r0=1, regime_a=False, kinds=_cycle(16, 1), swapped_header=False),
    dict(key="csds/2026/09/02/aaaa0005/csds", map="de_vertigo", build=10896, wingman=False, n=16, r0=1, regime_a=False, kinds=_cycle(16, 2), swapped_header=False),
    dict(key="csds/2026/09/03/aaaa0006/csds", map="de_nuke", build=10896, wingman=True, n=16, r0=1, regime_a=False, kinds=["time"] * 16, swapped_header=False, players=4),
    dict(key="csds/2024/03/01/bbbb0001/csds", map="de_anubis", build=9961, wingman=False, n=20, r0=1, regime_a=True, kinds=_cycle(20, 6), swapped_header=False, invert_after_swap=True),
    dict(key="csds/2024/03/02/bbbb0002/csds", map="de_ancient", build=9842, wingman=False, n=19, r0=2, regime_a=False, kinds=_cycle(19, 7), swapped_header=False),
    dict(key="csds/2026/09/04/cccc0001/csds", map="de_dust2", build=10896, wingman=False, n=20, r0=1, regime_a=False, kinds=_cycle(20, 2), swapped_header=False),
]
# a re-processed copy of the last match: other key, ticks shifted, players renumbered -> a different exact signature, the same tolerant key
DUP = dict(MATCHES[-1], key="csds/2026/09/05/cccc0002/csds", tick_shift=100_000, pid_shift=20)
MATCHES = MATCHES + [DUP]


def _fix_kinds(m, clean):
    """Rounds whose label the vote cannot repair (no second signal) must not be the 'time' kind: swap boundary (the score step is undefined), the
    last round of a match (no next score) and the 2024 build's suspect round. clean=True also removes every deliberately wrong recorded winner."""
    out = []
    mr = 16 if m["wingman"] else MAX_ROUNDS
    for i, k in enumerate(m["kinds"]):
        r = m["r0"] + i
        edge = r in (mr // 2, mr // 2 + 1, mr) or i == m["n"] - 1
        if k == "time" and edge and not m.get("players"):
            k = "explode"
        if k.endswith("_recwrong") and (clean or edge):
            k = k[:-len("_recwrong")]
        out.append(k)
    return out


def _build_match(m, clean=False):
    """Rows of every channel for one match plus the ground truth."""
    key, n, r0 = m["key"], m["n"], m["r0"]
    kinds = _fix_kinds(m, clean)
    shift, pshift = m.get("tick_shift", 0), m.get("pid_shift", 0)
    n_pl = m.get("players", 10)
    sc_a = sc_b = 0
    rows = {k: [] for k in ("re", "rs", "rst", "pi", "de", "bs")}
    truth = []
    for i in range(n):
        r = r0 + i
        F = 5_000 + 20_000 * r + shift
        winner, end, deaths, bomb = round_events(kinds[i], F - shift)
        end += shift
        nsw = sum(1 for rr in range(1, r) if swap_after(rr, 16 if m["wingman"] else MAX_ROUNDS))
        side_a = T if nsw % 2 == 0 else CT       # team A started on T
        rec = winner
        if kinds[i].endswith("_recwrong"):
            rec = CT if winner == T else T
        if m.get("invert_after_swap") and r == MAX_ROUNDS // 2 + 1:
            rec = CT if winner == T else T      # the 2024 build-9961 defect: the recorded winner is inverted on the first round after the swap
        t_score = sc_a if side_a == T else sc_b
        ct_score = sc_b if side_a == T else sc_a
        rows["rst"].append((key, r, F, "round_freeze_end", t_score, ct_score))
        if m["regime_a"] and i == 0:
            rows["rst"].append((key, r, F - 5, "round_announce_match_start", t_score, ct_score))
        rows["rs"].append((key, r))
        rows["re"].append((key, r, end, rec, 1, n_pl))
        team_side = {1: side_a, 0: CT if side_a == T else T}  # team A (ids 1..5) / team B (ids 6..10)
        ids_a, ids_b = list(range(1, 6)), list(range(6, 11))
        if n_pl == 4:
            ids_a, ids_b = [1, 2], [6, 7]
        for pid in ids_a:
            rows["pi"].append((key, r, pid + pshift, team_side[1]))
        for pid in ids_b:
            rows["pi"].append((key, r, pid + pshift, team_side[0]))
        used = {T: [], CT: []}
        for tick, side in sorted(deaths):
            pool = ids_a if side == side_a else ids_b
            pid = pool[len(used[side])] + pshift
            used[side].append(pid)
            rows["de"].append((key, r, tick + shift, pid, side))
        for typ, tick in bomb:
            rows["bs"].append((key, r, tick + shift, typ))
        if winner == side_a:
            sc_a += 1
        else:
            sc_b += 1
        truth.append({"round": r, "winner": winner, "kind": kinds[i], "rec": rec})
    tf, cf = (sc_a, sc_b) if not m.get("swapped_header") else (sc_b, sc_a)
    return rows, truth, (tf, cf)


def write_tomes(root, store="syn", matches=None, clean=False):
    """clean=True: no deliberately wrong recorded winner (the store the e2e positive control runs on); False keeps them (the prepare known-answer test)."""
    matches = MATCHES if matches is None else matches
    base = os.path.join(root, "tomes", store, "tome", "csds")
    acc = {k: [] for k in ("re", "rs", "rst", "pi", "de", "bs")}
    headers, truth = [], {}
    for m in matches:
        rows, tr, (tf, cf) = _build_match(m, clean)
        for k in acc:
            acc[k] += rows[k]
        headers.append({"key": m["key"], "build_num": m["build"], "map_name": m["map"], "match_date": "2026-09-01", "match_type": "competitive",
                        "is_wingman": m["wingman"], "max_rounds": MAX_ROUNDS if not m["wingman"] else 16, "t_starters_avg_rank": 10.0,
                        "ct_starters_avg_rank": 10.0, "t_starters_score_final": tf, "ct_starters_score_final": cf})
        truth[m["key"]] = {"rounds": tr, "tf": tf, "cf": cf, "map": m["map"], "wingman": m["wingman"], "n": m["n"], "dup": "tick_shift" in m}

    def put(folder, df, keyset=True):
        d = os.path.join(base, folder)
        os.makedirs(d, exist_ok=True)
        df.write_parquet(os.path.join(d, "dataframe_00000"))
        if keyset:
            pl.DataFrame({"_": sorted(set(df["match_key"].to_list()))}, schema={"_": pl.String}).write_parquet(os.path.join(d, "keyset_00000"))

    sch = lambda **kw: kw  # noqa: E731
    put("header", pl.DataFrame(headers, schema={"key": pl.String, "build_num": pl.Int64, "map_name": pl.String, "match_date": pl.String, "match_type": pl.String,
                                                "is_wingman": pl.Boolean, "max_rounds": pl.Int64, "t_starters_avg_rank": pl.Float64, "ct_starters_avg_rank": pl.Float64,
                                                "t_starters_score_final": pl.Int64, "ct_starters_score_final": pl.Int64}), keyset=False)
    with_keys = pl.DataFrame({"_": sorted(m["key"] for m in matches)}, schema={"_": pl.String})
    with_keys.write_parquet(os.path.join(base, "header", "keyset_00000"))
    put("basic_round_end.COMBINED", pl.DataFrame(acc["re"], orient="row", schema={"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64, "winner_team_code": pl.Int64,
                                                                                  "win_reason_code": pl.Int64, "player_count_norm": pl.Int64}))
    put("basic_round_start.COMBINED", pl.DataFrame(acc["rs"], orient="row", schema={"match_key": pl.String, "round": pl.Int64}))
    put("basic_round_state.COMBINED", pl.DataFrame(acc["rst"], orient="row", schema={"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64, "event_type": pl.String,
                                                                                    "t_score": pl.Int64, "ct_score": pl.Int64}))
    put("basic_player_info.COMBINED", pl.DataFrame(acc["pi"], orient="row", schema={"match_key": pl.String, "round": pl.Int64, "player_id": pl.Int64, "team_code": pl.Int64}))
    put("basic_player_death.COMBINED", pl.DataFrame(acc["de"], orient="row", schema={"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64, "player_id": pl.Int64,
                                                                                    "player_team_code": pl.Int64}))
    put("basic_bomb_state.COMBINED", pl.DataFrame(acc["bs"], orient="row", schema={"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64, "event_type": pl.String}))
    empty = pl.DataFrame([], schema={"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64})
    put("basic_player_disconnect.COMBINED", empty, keyset=False)
    put("basic_bot_takeover.COMBINED", empty, keyset=False)
    return truth


@contextlib.contextmanager
def _quiet():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def run_pipeline(root, store="syn", do_grid=True):
    """prepare.py then make_grid.py on the synthetic tomes, with their ROOT pointed at `root` (their own code, not a copy)."""
    if DATA_SCRIPTS not in sys.path:
        sys.path.insert(0, DATA_SCRIPTS)
    import make_grid
    import prepare
    old_argv = sys.argv
    try:
        prepare.ROOT = root
        make_grid.ROOT = root
        with _quiet():
            sys.argv = ["prepare.py", "--store", store]
            prepare.main()
            if do_grid:
                sys.argv = ["make_grid.py", "--store", store]
                make_grid.main()
    finally:
        sys.argv = old_argv
