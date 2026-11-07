"""A8 check 12: round-end-tick state share by folder day in the 2026 kept set.

The batch-holdout check (scripts/model/sufficiency/batch_holdout.py) compares day 08 with days 01-07.
If the share of round-end-tick (decided) states differs by day, the batch gap includes a composition
effect. Descriptive only: counts per folder day, plus the decided share. Uses xgb_primary_seed0.build_data.
Run from the project root:
  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A8_batch_composition.py
"""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    xgbs = load("xgb_primary_seed0", ROOT / "scripts" / "model" / "xgb_primary_seed0.py")
    bo = xgbs.bo
    fv = bo.fv
    D = xgbs.build_data(dedupe=True)
    S = D["S"]
    RE = fv.read("round_end", ["match_key", "round", "tick"])
    end = {(m, int(r)): int(t) for m, r, t in RE.iter_rows()}
    rows = S.select(["match_key", "round", "tick", "alive_ct", "alive_t", "bomb_planted"]).to_dicts()
    day_states = Counter()
    day_end = Counter()
    day_rounds = {}
    for x in rows:
        day = x["match_key"].split("/")[3]
        day_states[day] += 1
        if x["tick"] == end.get((x["match_key"], int(x["round"])), -1):
            day_end[day] += 1
        day_rounds.setdefault(day, set()).add((x["match_key"], int(x["round"])))
    print("== 2026 kept states by folder day ==")
    print(f"{'day':<6}{'rounds':>8}{'states':>9}{'round_end':>11}{'share':>8}")
    tot_s = sum(day_states.values())
    tot_e = sum(day_end.values())
    for d in sorted(day_states):
        print(f"{d:<6}{len(day_rounds[d]):>8}{day_states[d]:>9}{day_end[d]:>11}{100 * day_end[d] / day_states[d]:>7.2f}%")
    print(f"{'all':<6}{sum(len(v) for v in day_rounds.values()):>8}{tot_s:>9}{tot_e:>11}{100 * tot_e / tot_s:>7.2f}%")
    d8 = day_end["08"] / day_states["08"]
    rest_s = tot_s - day_states["08"]
    rest_e = tot_e - day_end["08"]
    print(f"\nday 08 share {100 * d8:.2f}% vs days 01-07 share {100 * rest_e / rest_s:.2f}%")


if __name__ == "__main__":
    main()
