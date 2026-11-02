"""End-to-end selftest on a synthetic tome store with known ground truth (tens of seconds, no real data).

A. KNOWN ANSWERS of the real scripts/data/prepare.py and make_grid.py on 10 synthetic matches (the code under test is imported, not copied):
   * every labelled round has the winner the generator planted, also where the recorded winner is wrong and the vote must decide
     (kills P1 majority-T labelled CT and P4 inverted elimination vote), the elimination vote has the planted side,
   * hostage / out-of-pool / wingman matches are not used and exactly one of two duplicate copies is (kills P2 and P3),
   * the grid has the planted number of rows per round, the plant exactly on a grid tick gives bomb_planted 1 and seconds_since_plant 0.0,
     alive counts step at the death tick (<=), the clock equals the second, the label equals the winner (kills G1, G2, G3, G4).
B. POSITIVE CONTROL: e2e_checks.main on a clean synthetic store must not FAIL.
C. NEGATIVE CONTROLS: damaged copies of the tables must turn the intended checks red (a check that stays green on damage is blind), an empty
   population must FAIL (never a vacuous pass), and the report is written even when the data are broken."""
import contextlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not put the script folder on sys.path
from harness import T  # noqa: E402

import polars as pl  # noqa: E402

import wp_common as W  # noqa: E402
from checks import core, synth  # noqa: E402
from checks import ctx as C  # noqa: E402
from checks import grid_checks, label_checks, split_checks, store_checks  # noqa: E402

T_, CT_ = 2, 3
MAPS = "de_mirage,de_inferno,de_dust2,de_nuke,de_anubis,de_ancient"
t = T("synthetic store: prepare.py -> make_grid.py -> checks")

# =================================================================== A. known answers
tmp = tempfile.mkdtemp(prefix="wp_selftest_a_")
truth = synth.write_tomes(tmp, "syn", clean=False)
synth.run_pipeline(tmp, "syn")
d = os.path.join(tmp, "data", "syn")
R = pl.read_parquet(os.path.join(d, "rounds.parquet"))
M = pl.read_parquet(os.path.join(d, "matches.parquet"))
G = pl.read_parquet(os.path.join(d, "grid.parquet"))
rounds_truth = {(k, r["round"]): r for k, v in truth.items() for r in v["rounds"]}

bad_final = [(r["match_key"][-12:], r["round"], rounds_truth[(r["match_key"], r["round"])]["kind"]) for r in R.iter_rows(named=True)
             if r["player_count_norm"] == 10 and r["final"] != rounds_truth[(r["match_key"], r["round"])]["winner"]]
t.check("A1 every 10-player round is labelled with the planted winner (also where the recorded winner is wrong)", not bad_final, f"{len(R)} rounds; wrong or unresolved: {bad_final[:5]}")
n_split_ct = sum(1 for (k, r), x in rounds_truth.items() if x["kind"] == "ct_wipe_recwrong")
n_split_t = sum(1 for (k, r), x in rounds_truth.items() if x["kind"] == "t_wipe_recwrong")
t.check("A1b the fixture exercises rounds decided by the elimination vote against a wrong recorded winner", n_split_ct >= 1 and n_split_t >= 1, f"{n_split_ct} + {n_split_t}")
bad_elim = [(r["match_key"][-12:], r["round"]) for r in R.iter_rows(named=True)
            if rounds_truth[(r["match_key"], r["round"])]["kind"].startswith("ct_wipe") and r["elim"] != T_
            or rounds_truth[(r["match_key"], r["round"])]["kind"].startswith("t_wipe") and r["elim"] != CT_]
t.check("A2 the elimination vote names the side that did NOT get wiped", not bad_elim, f"{bad_elim[:5]}")
used = set(R.filter(pl.col("use"))["match_key"].to_list())
want_used = {k for k, v in truth.items() if v["map"].startswith("de_") and v["map"] not in ("de_vertigo",) and not v["wingman"] and k != "csds/2026/09/05/m_partial/csds"}
t.check("A3 used matches = in-scope, non-wingman, one copy of the duplicate pair (hostage and out-of-pool maps, wingman and the re-processed copy are not used)",
        used == want_used, f"used {sorted(x[-12:] for x in used)} want {sorted(x[-12:] for x in want_used)}")
dup_used = [k for k in ("csds/2026/09/04/m_partial/csds", "csds/2026/09/05/m_partial/csds") if k in used]
t.check("A3b exactly one copy of the duplicate pair is used", len(dup_used) == 1, str(dup_used))
sus = R.filter(pl.col("rec_suspect")).select("match_key", "round").rows()
t.check("A4 the recorded winner is only suspect on the first round after the half of the 2024 build", sus == [("csds/2024/03/01/m_partial/csds", 13)], str(sus))

# --- grid
gk = {}
for r in G.iter_rows(named=True):
    gk[(r["match_key"], r["round"], r["second"])] = r
rows_per = G.group_by(["match_key", "round"]).len()
bad_rows = [(k[-12:], rd, n, synth.EXPECTED_ROWS[rounds_truth[(k, rd)]["kind"]]) for k, rd, n in rows_per.iter_rows() if n != synth.EXPECTED_ROWS[rounds_truth[(k, rd)]["kind"]]]
t.check("A5 rows per round = whole seconds strictly before the decisive event (an extra point per round is wrong)", not bad_rows and rows_per.height == int(R["use"].sum()), f"{rows_per.height} rounds, wrong: {bad_rows[:4]}")
bad_plant = []
for (k, rd), x in rounds_truth.items():
    sec = synth.PLANT_SEC.get(x["kind"])
    if sec is None or (k, rd, sec) not in gk:
        continue
    on, before = gk[(k, rd, sec)], gk[(k, rd, sec - 1)]
    if not (on["bomb_planted"] == 1 and on["seconds_since_plant"] == 0.0 and before["bomb_planted"] == 0 and before["seconds_since_plant"] == -1.0):
        bad_plant.append((k[-12:], rd, on["bomb_planted"], on["seconds_since_plant"]))
n_aligned = sum(1 for (k, rd), x in rounds_truth.items() if x["kind"] in synth.PLANT_SEC and (k, rd, 0) in gk)
t.check("A6 a plant exactly on a grid tick: bomb_planted 1 and seconds_since_plant 0.0 on that row, 0 / -1.0 one second earlier", not bad_plant and n_aligned > 0, f"{n_aligned} aligned plants; wrong: {bad_plant[:4]}")
clock_bad = sum(1 for r in G.iter_rows(named=True) if r["seconds_since_round_live"] != float(r["second"]))
t.check("A7 seconds_since_round_live equals the second since the freeze end", clock_bad == 0, f"{clock_bad} of {G.height} rows")
lab_bad = sum(1 for r in G.iter_rows(named=True) if r["label_ct_win"] != int(rounds_truth[(r["match_key"], r["round"])]["winner"] == CT_))
t.check("A8 label_ct_win is 1 exactly when CT won the round", lab_bad == 0, f"{lab_bad} of {G.height} rows")
first = next(((k, rd) for (k, rd), x in rounds_truth.items() if x["kind"] == "ct_wipe" and (k, rd, 0) in gk), None)
steps = {}
if first:
    k, rd = first
    for s_ in (19, 20, 29, 30, 59):
        steps[s_] = (gk[(k, rd, s_)]["alive_ct"], gk[(k, rd, s_)]["alive_t"])
    steps_t = (gk[(k, rd, 24)]["alive_t"], gk[(k, rd, 25)]["alive_t"])
    ok_steps = steps == {19: (5, 5), 20: (4, 5), 29: (4, 4), 30: (3, 4), 59: (1, 4)} and steps_t == (5, 4)
else:
    ok_steps, steps_t = False, None
t.check("A9 alive counts step AT the death tick (a death on a grid tick is already counted on that row)", ok_steps, f"{steps} T at 24/25: {steps_t}")

# =================================================================== B. positive control on a clean store
tmp2 = tempfile.mkdtemp(prefix="wp_selftest_b_")
synth.write_tomes(tmp2, "syn", clean=True)
synth.run_pipeline(tmp2, "syn")
import e2e_checks  # noqa: E402

report = os.path.join(tmp2, "e2e.json")
import io  # noqa: E402

try:
    with contextlib.redirect_stdout(io.StringIO()):  # the 60 lines of the positive control are not interesting here
        e2e_checks.main(["--root", tmp2, "--store", "syn", "--model", "nomodel", "--fast", "--allow-missing", "--maps", MAPS, "--param", "reconcile_min_matches=1",
                         "--param", "signal_min_n=10", "--report", report])
    code = 0
except SystemExit as e:
    code = e.code
rep = json.load(open(report, encoding="utf-8"))
t.check("B1 e2e_checks on a clean synthetic store exits 0 with no FAIL", code == 0 and rep["all_passed"] and rep["status"] == "complete", f"exit {code}, failed {rep['failed']}, counts {rep['counts']}")
t.check("B2 the report carries an input manifest, parameters, row counts and the peak memory",
        rep["inputs"]["files"]["grid.parquet"].get("rows", 0) > 0 and rep["inputs"]["files"]["rounds.parquet"].get("sha256") and "tick_rate" in rep["params"] and rep["peak_rss_mb"], "")
ids = {r["id"]: r["status"] for r in rep["checks"]}
t.check("B3 the checks that must have run on this store did run and passed", all(ids.get(i) in ("PASS", "WARN") for i in ("S1", "S2", "S3", "A1", "A3", "B3", "C4", "C6", "C9", "D8", "D9", "D10", "E1", "E4")),
        str({i: ids.get(i) for i in ("S1", "S2", "S3", "A1", "A3", "B3", "C4", "C6", "C9", "D8", "D9", "D10", "E1", "E4")}))

# =================================================================== C. negative controls
GROUPS = {"S": store_checks.store_consistency, "A": store_checks.scope_checks, "K": store_checks.completeness_checks, "B": store_checks.duplicate_checks,
          "C": label_checks.label_checks, "D": grid_checks.grid_checks, "E": split_checks.split_checks}


def run_groups(groups, manifest=None, params=(), fast=True):
    P = core.Params(C.DEFAULTS, ["reconcile_min_matches=1", "signal_min_n=10", *params])
    Rp = core.Reporter(allow_missing=True, echo=False)
    ctx = C.Ctx(tmp2, "syn", "nomodel", P, Rp, maps=MAPS, fast=fast, split_manifest=manifest or "none")
    for g in groups:
        GROUPS[g](ctx)
    return {r["id"]: r["status"] for r in Rp.results}


@contextlib.contextmanager
def damaged(name, fn):
    """Rewrite data/syn/<name>.parquet with fn(DataFrame) for the duration of the block, then restore the original bytes."""
    path = os.path.join(tmp2, "data", "syn", name)
    shutil.copyfile(path, path + ".bak")
    try:
        fn(pl.read_parquet(path)).write_parquet(path)
        yield
    finally:
        os.replace(path + ".bak", path)


base = run_groups(["S", "A", "K", "B", "C", "D", "E"])
t.check("C0 undamaged store: no check is red", "FAIL" not in base.values(), str({k: v for k, v in base.items() if v == "FAIL"}))

# ---- grid damage
def extra_point(G):
    last = G.group_by(["match_key", "round"], maintain_order=True).last()
    last = last.with_columns((pl.col("tick") + 64), (pl.col("second") + 1), (pl.col("seconds_since_round_live") + 1.0))
    return pl.concat([G, last.select(G.columns)]).sort(["match_key", "round", "tick"])


with damaged("grid.parquet", lambda G: G.with_columns((1 - pl.col("label_ct_win")).cast(pl.Int8).alias("label_ct_win"))):
    t.expect_fail("C1 inverted grid label -> D7", run_groups(["D"]), ["D7"])
with damaged("grid.parquet", extra_point):
    t.expect_fail("C2 one extra grid point per round -> D9", run_groups(["D"]), ["D9", "D7"][:1])
with damaged("grid.parquet", lambda G: G.with_columns(pl.when(pl.col("seconds_since_plant") == 0.0).then(0).otherwise(pl.col("bomb_planted")).cast(pl.Int8).alias("bomb_planted"),
                                                       pl.when(pl.col("seconds_since_plant") == 0.0).then(-1.0).otherwise(pl.col("seconds_since_plant")).cast(pl.Float32).alias("seconds_since_plant"))):
    t.expect_fail("C3 the planted flag one tick late -> D8", run_groups(["D"]), ["D8"])
with damaged("grid.parquet", lambda G: G.with_columns((pl.col("seconds_since_round_live") - 1.0).cast(pl.Float32))):
    t.expect_fail("C4 the clock one second behind -> D8", run_groups(["D"]), ["D8"])
with damaged("grid.parquet", lambda G: G.with_columns(pl.when(pl.col("alive_t") < 5).then(pl.col("alive_t") + 1).otherwise(pl.col("alive_t")).cast(pl.Int8).alias("alive_t")).with_columns(
        (pl.col("alive_ct") - pl.col("alive_t")).cast(pl.Int8).alias("ct_alive_minus_t"))):
    t.expect_fail("C5 alive_t one too high on every row (consistent difference column) -> D10", run_groups(["D"]), ["D10"])
import bisect  # noqa: E402

DEATHS = {}
for mk_, rd_, tk_, sd_ in pl.read_parquet(os.path.join(tmp2, "data", "syn", "deaths.parquet")).filter(pl.col("side_ok")).select("match_key", "round", "tick", "side").iter_rows():
    DEATHS.setdefault((mk_, rd_, sd_), []).append(tk_)
for v_ in DEATHS.values():
    v_.sort()


def alive_by(G, cut):
    """alive counts from the deaths with tick <= cut(tick) (cut = tick: the correct rule; tick - 1: a strict '<'; tick + 64: one second of the future)."""
    ac, at = [], []
    for mk, rd, tk in G.select("match_key", "round", "tick").iter_rows():
        ac.append(5 - bisect.bisect_right(DEATHS.get((mk, rd, 3), []), cut(tk)))
        at.append(5 - bisect.bisect_right(DEATHS.get((mk, rd, 2), []), cut(tk)))
    return G.with_columns(pl.Series("alive_ct", ac, dtype=pl.Int8), pl.Series("alive_t", at, dtype=pl.Int8),
                          pl.Series("ct_alive_minus_t", [a - b for a, b in zip(ac, at)], dtype=pl.Int8))


t.check("C5a the causal canary (raw tome events, tick <= row tick) passes on the undamaged grid", run_groups(["D"], fast=False).get("D12") == "PASS", "")
t.check("C5b alive counts computed with the correct '<=' leave D10 green", run_groups(["D"]).get("D10") == "PASS", "")
with damaged("grid.parquet", lambda G: alive_by(G, lambda tk: tk - 1)):
    st = run_groups(["D"], fast=False)
    t.expect_fail("C5c a death ON a grid tick not yet counted ('<' instead of '<=') -> D10 and D12", st, ["D10", "D12"])
with damaged("grid.parquet", lambda G: alive_by(G, lambda tk: tk + 64)):
    st = run_groups(["D"], fast=False)
    t.expect_fail("C5d alive counts that look one second into the future -> D10 and D12", st, ["D10", "D12"])
with damaged("grid.parquet", lambda G: G.with_columns(pl.when(pl.col("second") == 5).then(0).otherwise(pl.col("alive_ct")).cast(pl.Int8).alias("alive_ct"))):
    t.expect_fail("C5e rows with the CT side wiped -> D6", run_groups(["D"]), ["D6"])
with damaged("grid.parquet", lambda G: G.with_columns(pl.when(pl.col("second") == 5).then(0).otherwise(pl.col("alive_t")).cast(pl.Int8).alias("alive_t"))):
    t.expect_fail("C5f rows with the T side wiped and no plant -> W1", run_groups(["D"]), ["W1"])
with damaged("grid.parquet", lambda G: G.head(0)):
    st = run_groups(["D", "E"])
    t.expect_fail("C6 an EMPTY grid fails (never a vacuous pass)", st, ["D1"])
with damaged("grid.parquet", lambda G: G.filter(pl.col("round") != 3)):
    t.expect_fail("C7 used rounds without grid rows -> D7 and S3", run_groups(["D", "S"]), ["D7", "S3"])

# ---- label damage
def flip_elim(Rn):
    return Rn.with_columns(pl.when(pl.col("elim").is_null()).then(None).otherwise(5 - pl.col("elim")).cast(pl.Int32).alias("elim"))


def p1_like(Rn):  # majority-T rounds labelled CT
    return Rn.with_columns(pl.when(pl.col("bomb").is_null() & (pl.col("final") == T_)).then(CT_).otherwise(pl.col("final")).cast(pl.Int32).alias("final"))


with damaged("rounds.parquet", flip_elim):
    t.expect_fail("C8 inverted elimination vote -> C6 and C4", run_groups(["C"]), ["C6", "C4"])
with damaged("rounds.parquet", p1_like):
    st = run_groups(["C", "D"])
    t.expect_fail("C9 non-bomb T wins labelled CT -> C4, C9 (header reconciliation), D7", st, ["C4", "C9", "D7"])
with damaged("rounds.parquet", lambda Rn: Rn.with_columns(pl.when(pl.col("bomb").is_not_null()).then(5 - pl.col("bomb")).otherwise(None).cast(pl.Int32).alias("bomb"))):
    t.expect_fail("C10 inverted bomb signal -> C5", run_groups(["C"]), ["C5"])
with damaged("matches.parquet", lambda Mt: Mt.with_columns(
        pl.when(pl.col("match_key").str.contains("aaaa0001")).then(pl.col("t_starters_score_final") + 1).otherwise(pl.col("t_starters_score_final")).alias("t_starters_score_final"),
        pl.when(pl.col("match_key").str.contains("aaaa0001")).then(pl.col("ct_starters_score_final") - 1).otherwise(pl.col("ct_starters_score_final")).alias("ct_starters_score_final"))):
    t.expect_fail("C11 a header score pair that does not match the round winners (same total) -> C9", run_groups(["C"]), ["C9"])
with damaged("rounds.parquet", lambda Rn: Rn.with_columns(pl.lit("recorded_only").alias("label_source"))):
    t.expect_fail("C12 recorded_only everywhere -> C2", run_groups(["C"]), ["C2"])

# ---- scope, duplicates, store consistency
with damaged("matches.parquet", lambda Mt: Mt.with_columns(pl.when(pl.col("match_key").str.contains("aaaa0001")).then(pl.lit("de_vertigo")).otherwise(pl.col("map_name")).alias("map_name"))):
    t.expect_fail("C13 a used match on a map outside the model's list -> A1", run_groups(["A"]), ["A1"])
with damaged("rounds.parquet", lambda Rn: Rn.with_columns(pl.when(pl.col("match_key").str.contains("aaaa0004")).then(True).otherwise(pl.col("use")).alias("use"))):
    t.expect_fail("C14 a hostage-map match used -> A1b and A2", run_groups(["A"]), ["A1b", "A2"])
with damaged("rounds.parquet", lambda Rn: Rn.with_columns(pl.when(pl.col("match_key").str.contains("cccc0002")).then(True).otherwise(pl.col("use")).alias("use"))):
    t.expect_fail("C15 both copies of a duplicate pair used -> B3", run_groups(["B"]), ["B3"])
with damaged("rounds.parquet", lambda Rn: Rn.with_columns(pl.lit(False).alias("use"))):
    t.expect_fail("C16 no used round at all fails (vacuous population)", run_groups(["A", "C", "E"]), ["A1", "A2", "C1", "E2"])
ks = os.path.join(tmp2, "tomes", "syn", "tome", "csds", "basic_round_end.COMBINED", "keyset_00000")
shutil.copyfile(ks, ks + ".bak")
try:
    pl.concat([pl.read_parquet(ks), pl.DataFrame({"_": ["csds/2026/09/09/orphan/csds"]})]).write_parquet(ks)
    t.expect_fail("C17 a channel key that the header does not have (half-finished append) -> S1", run_groups(["S"]), ["S1"])
finally:
    os.replace(ks + ".bak", ks)
page = os.path.join(tmp2, "tomes", "syn", "tome", "csds", "basic_bomb_state.COMBINED", "dataframe_00000")
mt = os.path.getmtime(page)
try:
    os.utime(page, (mt + 3600, mt + 3600))
    t.expect_fail("C18 a tome file newer than the prepared tables (stale snapshot) -> S4", run_groups(["S"]), ["S4"])
finally:
    os.utime(page, (mt, mt))

# ---- completeness
with damaged("matches.parquet", lambda Mt: Mt.with_columns((pl.col("t_starters_score_final") + 5).alias("t_starters_score_final"))):
    t.expect_fail("C19 every match shorter than its header total -> K1 (hard bound)", run_groups(["K"]), ["K1"])

# ---- split: the legacy rule, and the frozen manifest
sp_old = W._holdout


def swapped(keys, seed, test_frac=0.2, val_frac=0.1):
    a, b, c = sp_old(keys, seed, test_frac, val_frac)
    return a, c, b  # validation and test traded places (mutant C2)


W._holdout = swapped
try:
    t.expect_fail("C20 validation and test swapped in the legacy split -> E4", run_groups(["E"]), ["E4"])
finally:
    W._holdout = sp_old
Gm = pl.read_parquet(os.path.join(tmp2, "data", "syn", "grid.parquet"), columns=["match_key", "era"]).unique("match_key").join(
    pl.read_parquet(os.path.join(tmp2, "data", "syn", "matches.parquet")).select("match_key", "match_date"), on="match_key")
sp = W.make_split("random_legacy", Gm, seed=0)
assign = {k: s for s in sp for k in sp[s]}
man = {"seed": 0, "assign": assign, "sha256": split_checks.manifest_digest(assign), "history": []}
mp = os.path.join(tmp2, "split_manifest.json")
json.dump(man, open(mp, "w"))
st = run_groups(["E"], manifest=mp)
t.check("C21 a valid frozen manifest: E1, E5, E6 pass", all(st.get(i) == "PASS" for i in ("E1", "E5", "E6")), str({i: st.get(i) for i in ("E1", "E5", "E6")}))
json.dump(dict(man, sha256="0" * 64), open(mp, "w"))
t.expect_fail("C22 a manifest whose sha256 does not match its content -> E5", run_groups(["E"], manifest=mp), ["E5"])
json.dump(dict(man, assign={k: v for k, v in list(assign.items())[1:]}, sha256=split_checks.manifest_digest(dict(list(assign.items())[1:]))), open(mp, "w"))
t.expect_fail("C23 a manifest that leaves a grid match unassigned -> E6", run_groups(["E"], manifest=mp), ["E6"])

# ---- the report is written (and not passing) when the data are broken, and a crash cannot leave an old all_passed=true file
shutil.copyfile(os.path.join(tmp2, "data", "syn", "grid.parquet"), os.path.join(tmp2, "grid.bak"))
open(os.path.join(tmp2, "data", "syn", "grid.parquet"), "wb").write(b"not a parquet file")
rep2 = os.path.join(tmp2, "e2e_broken.json")
json.dump({"all_passed": True, "status": "complete"}, open(rep2, "w"))
try:
    with contextlib.redirect_stdout(io.StringIO()):
        e2e_checks.main(["--root", tmp2, "--store", "syn", "--model", "nomodel", "--fast", "--allow-missing", "--maps", MAPS, "--only", "D,E", "--report", rep2])
    code2 = 0
except SystemExit as e:
    code2 = e.code
os.replace(os.path.join(tmp2, "grid.bak"), os.path.join(tmp2, "data", "syn", "grid.parquet"))
r2 = json.load(open(rep2, encoding="utf-8"))
t.check("C24 a corrupt grid file: exit 1, the old all_passed=true report is replaced by one that records the FAIL", code2 == 1 and r2["all_passed"] is False and r2["failed"], f"exit {code2} failed {r2['failed']}")

shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(tmp2, ignore_errors=True)
t.finish()
