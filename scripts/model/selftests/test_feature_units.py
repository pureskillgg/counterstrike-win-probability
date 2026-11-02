"""Known-answer unit tests of the loadout features and the snapshot-table exclusion (seconds, synthetic tables, no data):
  * item-code pins against a pinned weapon-name table (a dropped mp9 in wp_items.SMG)
  * hand-computed loadouts with zero-armor players, integer and text side codes, 300+300 seeded random player tables against an independent reference
    (armor_n counting everyone, swapped integer side codes)
  * the key set of the serving dict against the pinned 46 names (a dropped DIFF field), polars training path vs python serving path (st_ak counting the awp code)
  * features_status.inconsistent_rounds on rounds where only one side's table count differs (an AND instead of an OR)
  * the snapshot physical-bounds check: a clean sample passes, each kind of damage fails."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not put the script folder on sys.path
from harness import T  # noqa: E402

import polars as pl  # noqa: E402

from checks import core, unit  # noqa: E402
from checks import ctx as C  # noqa: E402
from checks import model_checks  # noqa: E402

t = T("feature units")
for name, res in unit.run_suite(unit.SUITE):
    t.res(name, res)

# ---- the physical-bounds check on a synthetic sample
import random

P = core.Params(C.DEFAULTS)
rng = random.Random(3)
rows = []
for _ in range(400):
    ps = unit.random_players(rng, n_ct=rng.randint(1, 5), n_t=rng.randint(1, 5))
    ref = unit.reference_loadout(ps)
    rows.append({"alive_ct": sum(1 for p in ps if p["side"] == 3), "alive_t": sum(1 for p in ps if p["side"] == 2), **ref})
s = pl.DataFrame(rows).with_columns([pl.col(c).cast(pl.Float32) for c in rows[0] if c.startswith("st_")] + [pl.col("alive_ct").cast(pl.Int8), pl.col("alive_t").cast(pl.Int8)])
t.res("bounds: a clean sample (400 random tables) passes", model_checks.snapshot_bounds_result(s, P))


def damaged(col, fn):
    vals = s[col].to_list()
    vals[0] = fn(vals[0], s["alive_ct"][0], s["alive_t"][0])
    return s.with_columns(pl.Series(col, vals, dtype=s[col].dtype))


bad = {
    "armor_n above the living players": damaged("st_armor_n_ct", lambda v, a, b: a + 1),
    "rifles above the living players": damaged("st_rifle_t", lambda v, a, b: b + 1),
    "hp above 100 per living player": damaged("st_hp_ct", lambda v, a, b: 100.0 * a + 1),
    "a NaN feature": damaged("st_equip_t", lambda v, a, b: float("nan")),
    "a diff column that is not ct - t": damaged("st_hp_diff", lambda v, a, b: v + 5.0),
}
for name, frame in bad.items():
    r = model_checks.snapshot_bounds_result(frame, P)
    t.check(f"bounds: {name} fails", r.status == "FAIL", r.detail)
t.finish()
