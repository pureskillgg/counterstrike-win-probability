"""verify D1-7 (swarm2 verifier): does the claim-side error need a tied pre-round score? (own check, not a copy of any other log)

Uses the per-round tables saved by verify_D1_7_mech.py in the scratchpad. t0/c0 are the first round_state row values of the round.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(16)
SP = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
J24 = pl.read_parquet(SP / "verify_D1_7_mech_rounds2024.parquet")
J26 = pl.read_parquet(SP / "verify_D1_7_mech_rounds2026.parquet")


def tab(J, lab, rounds, pc=None):
    x = J if pc is None else J.filter(pl.col("player_count") == pc)
    x = x.filter(pl.col("round").is_in(rounds) & pl.col("claim_side").is_not_null())
    x = x.with_columns((pl.col("t0") == pl.col("c0")).alias("tie"))
    print(f"\n{lab} rounds {rounds} (pc={pc}): claim-side rows {x.height}")
    print(x.group_by(["n_swap", "tie", "claim_mm"]).len().sort(["n_swap", "tie", "claim_mm"]))


tab(J26, "2026", [27, 33, 39, 45], pc=10)
tab(J24, "2024", [27], pc=20)
tab(J24, "2024", [27], pc=None)

# exchange rows in 2026 with no claim value: tie status
ex = J26.filter((pl.col("n_swap") > 0) & pl.col("claim_side").is_null()).with_columns((pl.col("t0") == pl.col("c0")).alias("tie"))
print("\n2026 exchange rows with null claim side, by round and tie:")
print(ex.group_by(["round", "tie"]).len().sort(["round", "tie"]))
