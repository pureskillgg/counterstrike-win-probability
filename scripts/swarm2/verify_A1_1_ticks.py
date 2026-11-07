# verify_A1_1_ticks.py
# Independent discriminator for A1-1: does per-tick raw data continue past each match's last round_end tick?
# If the tick-level channels (tick, player_status, player_vector) run on past the last recorded round, rounds were lost
# (truncation). If they stop at the last round_end tick, the match ended there and the header excess is not play.
# Read-only, raw E: files, one match folder at a time.
import os, csv, collections
import polars as pl
E = r"E:\csds_final\csds\2026\10"
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
rows = list(csv.DictReader(open(os.path.join(SCRATCH, "verify_A1_1_permatch.csv"), encoding="utf-8")))
want = {"m_4ceed6a37dd561e2028e114d": "C_reg_clinch",
        "m_ff06fa88583ee8b6377e2c3a": "C_reg_noclinch",
        "m_cc66b3bc82e7bd795fafd34e": "C_OT",
        "368f3ae-7353-4f68-8d04-7c97f22ea2b8": "B_sample"}
picked = []
for r in rows:
    for uid, lab in want.items():
        if r["key"].endswith(uid + "/csds") or (lab == "B_sample" and r["key"].endswith("368f3ae-7353-4f68-8d04-7c97f22ea2b8/csds")):
            picked.append((lab, r))
# add two agreeing regulation matches (one clinch at last round, one early end) for comparison
agree_reg = [r for r in rows if r["cat"] == "A_agree" and int(r["n"]) <= int(r["mr"]) and int(r["n"]) == int(r["first_reach"])]
agree_noclinch = [r for r in rows if r["cat"] == "A_agree" and int(r["first_reach"]) == -1 and int(r["n"]) < int(r["mr"])]
picked.append(("A_clinch_at_end", agree_reg[0]))
picked.append(("A_early_end_noclinch", agree_noclinch[0]))
print("== verify_A1_1_ticks ==")
CH = ["tick", "player_status", "player_vector", "round_state", "round_start", "round_end", "score_update", "bomb_state"]
for lab, r in picked:
    key = r["key"]
    dd, mid = key.split("/")[3], key.split("/")[4]
    base = os.path.join(E, dd, mid)
    print("\n", lab, key[-40:], "| csv n", r["n"], "S", r["S"], "mr", r["mr"], "cat", r["cat"], "first_reach", r["first_reach"])
    rend = pl.read_parquet(os.path.join(base, "round_end"))
    last_tick = int(rend.get_column("tick").max())
    print("   round_end rows", rend.height, "| max round", int(rend.get_column("round").max()), "| last round_end tick", last_tick,
          "| columns", rend.columns)
    for ch in CH:
        p = os.path.join(base, ch)
        if not os.path.exists(p):
            print("   ", ch, "missing")
            continue
        df = pl.read_parquet(p)
        cols = df.columns
        info = f"   {ch}: rows {df.height} cols {len(cols)}"
        if "tick" in cols:
            mt = int(df.get_column("tick").max())
            after = int(df.filter(pl.col("tick") > last_tick).height)
            info += f" | max tick {mt} | rows after last round_end tick {after}"
        for rc in ["round", "round_number", "game_round", "round_num"]:
            if rc in cols:
                info += f" | max {rc} {int(df.get_column(rc).max())}"
                break
        print(info)
        if ch in ("tick", "player_status") and "tick" in cols:
            print("      tick cols sample:", cols[:14])
