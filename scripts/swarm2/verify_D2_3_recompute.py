"""verify_D2_3 recompute: independent check of the 2024 round-13/25/28 "recorded winner flips" claim.

Built from the raw tome tables (no D2 intermediate, except a read-only reproduction check against
logs/swarm2/D2_truth_2024.parquet). Intermediates go to the scratchpad, not the project.

Channels
  R            round_end.winner_team_code, first row per (match, round) with player_count == PC (D2_test rule)
  sx           side of team X (X = players on T in the first valid 5v5 round), from player_info (D2_test rule)
  s_score_sx   D2 team-tracked score truth (uses sx and sx_next; the claim's first source)
  s_score_free side-attached score delta: t_score or ct_score rises by 1 between freeze_end(r) and freeze_end(r+1).
               No team identity or sx is used. ASSUMPTION: t_score / ct_score are side-attached.
  s_bomb_w     bomb outcome in [freeze_end, round_end + 40] (D2 window). exploded -> 2, defused -> 3. Sx-free.
  s_bomb_u     bomb outcome, no window, first event in round tag. Sx-free.
  s_elim_sx    D2 elimination truth (uses sx and player_info membership).
  s_elim_free  elimination truth from player_death.player_team_code (victim side, 5 deaths rule). Sx-free.
  T_sx         coalesce(s_score_sx, s_bomb_w, s_elim_sx)   (claim's truth)
  T_free       coalesce(s_score_free, s_bomb_w, s_elim_free)   (sx-free truth)

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_3_recompute.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
PC = {"2024": 20, "2026": 10}
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
               r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D2_3")
SCRATCH.mkdir(parents=True, exist_ok=True)
KEY = ["match_key", "round"]


def tbl(label, ch, cols):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    return pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*"))).select(cols).collect()


def grp_expr():
    r = pl.col("round")
    return (pl.when(r == 13).then(pl.lit("r13")).when(r == 25).then(pl.lit("r25"))
            .when(r == 26).then(pl.lit("r26")).when(r == 27).then(pl.lit("r27"))
            .when(r == 28).then(pl.lit("r28")).when(r <= 12).then(pl.lit("r01_12"))
            .when(r <= 24).then(pl.lit("r14_24")).otherwise(pl.lit("r29plus")).alias("grp"))


def build(label):
    pc = PC[label]
    re_all = tbl(label, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    # window end: first round_end row of any player_count (D2_truth rule)
    re_t = (re_all.sort("tick").unique(KEY, keep="first", maintain_order=True)
            .select(KEY + [pl.col("tick").alias("t_end_any")]))
    re_ = (re_all.filter(pl.col("player_count") == pc).sort(["match_key", "round", "tick"])
           .unique(KEY, keep="first", maintain_order=True)
           .select(KEY + [pl.col("winner_team_code").cast(pl.Int64).alias("R"), "win_reason_code"]))

    pi = (tbl(label, "player_info", ["match_key", "round", "player_id", "team_code"])
          .unique(["match_key", "round", "player_id"], keep="first", maintain_order=True))
    cnt = pi.group_by(KEY).agg(pl.len().alias("np"), (pl.col("team_code") == 2).sum().alias("nt"),
                               (pl.col("team_code") == 3).sum().alias("nct"))
    valid = cnt.filter((pl.col("np") == 10) & (pl.col("nt") == 5) & (pl.col("nct") == 5)).select(KEY)
    pv = pi.join(valid, on=KEY, how="inner")
    first = valid.group_by("match_key").agg(pl.col("round").min().alias("F"))
    xmap = (pv.join(first, on="match_key", how="inner")
            .filter((pl.col("round") == pl.col("F")) & (pl.col("team_code") == 2))
            .select(["match_key", "player_id"]).unique().with_columns(pl.lit(True).alias("isX")))
    pv = pv.join(xmap, on=["match_key", "player_id"], how="left").with_columns(pl.col("isX").fill_null(False))
    agg = pv.group_by(KEY).agg(((pl.col("team_code") == 2) & pl.col("isX")).sum().alias("x_t"),
                               pl.col("isX").sum().alias("n_x"),
                               ((pl.col("team_code") == 2) & (~pl.col("isX"))).sum().alias("y_t"),
                               (~pl.col("isX")).sum().alias("n_y"))
    side = agg.with_columns(
        pl.when((pl.col("x_t") == pl.col("n_x")) & (pl.col("y_t") == 0)).then(pl.lit(2))
        .when((pl.col("x_t") == 0) & (pl.col("y_t") == pl.col("n_y"))).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("sx")).select(KEY + ["sx"])

    base = re_.join(side, on=KEY, how="left")
    base = base.join(side.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("sx").alias("sx_prev")]),
                     on=KEY, how="left")
    base = base.join(side.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("sx").alias("sx_nx")]),
                     on=KEY, how="left")
    base = base.join(re_t, on=KEY, how="left")

    rs = tbl(label, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    fe = (rs.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
          .unique(KEY, keep="first", maintain_order=True)
          .select(KEY + [pl.col("tick").alias("t_fe"), pl.col("t_score").alias("t0"), pl.col("ct_score").alias("ct0")]))
    base = base.join(fe, on=KEY, how="left")
    base = base.join(fe.select(["match_key", (pl.col("round") - 1).alias("round"),
                                pl.col("t0").alias("t1"), pl.col("ct0").alias("ct1")]), on=KEY, how="left")

    # ---- team-tracked score truth (claim's s_score)
    base = base.with_columns(
        pl.when(pl.col("sx") == 2).then(pl.col("t0")).otherwise(pl.col("ct0")).alias("x0"),
        pl.when(pl.col("sx_nx") == 2).then(pl.col("t1")).otherwise(pl.col("ct1")).alias("x1"),
        pl.when(pl.col("sx") == 3).then(pl.col("t0")).otherwise(pl.col("ct0")).alias("y0"),
        pl.when(pl.col("sx_nx") == 3).then(pl.col("t1")).otherwise(pl.col("ct1")).alias("y1"))
    base = base.with_columns((pl.col("x1") - pl.col("x0")).alias("dX"), (pl.col("y1") - pl.col("y0")).alias("dY"))
    base = base.with_columns(
        pl.when((pl.col("dX") == 1) & (pl.col("dY") == 0)).then(pl.col("sx"))
        .when((pl.col("dY") == 1) & (pl.col("dX") == 0)).then(5 - pl.col("sx"))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_score_sx"))
    # ---- sx-free side score delta
    dT = pl.col("t1") - pl.col("t0")
    dC = pl.col("ct1") - pl.col("ct0")
    base = base.with_columns(
        pl.when((dT == 1) & (dC == 0)).then(pl.lit(2)).when((dC == 1) & (dT == 0)).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_score_free"))

    # ---- bomb outcome: windowed (D2) and unwindowed
    bs = tbl(label, "bomb_state", ["match_key", "round", "tick", "event_type"]).filter(
        pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
    bo_u = bs.sort("tick").group_by(KEY).agg(pl.col("event_type").first().alias("bo_u"))
    bo_w = (bs.join(base.select(KEY + ["t_fe", "t_end_any"]), on=KEY, how="inner")
            .filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end_any") + 40))
            .sort("tick").group_by(KEY).agg(pl.col("event_type").first().alias("bo_w")))
    base = base.join(bo_w, on=KEY, how="left").join(bo_u, on=KEY, how="left")
    base = base.with_columns(
        pl.when(pl.col("bo_w") == "bomb_exploded").then(pl.lit(2)).when(pl.col("bo_w") == "bomb_defused").then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_bomb_w"),
        pl.when(pl.col("bo_u") == "bomb_exploded").then(pl.lit(2)).when(pl.col("bo_u") == "bomb_defused").then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_bomb_u"))

    # ---- elimination: sx-free (player_team_code of victims, one row per victim per round)
    dth = tbl(label, "player_death", ["match_key", "round", "player_id", "player_team_code"])
    dth_u = dth.unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
    el_f = (dth_u.filter(pl.col("player_team_code").is_not_null()).group_by(KEY)
            .agg((pl.col("player_team_code") == 2).sum().alias("t_dead_f"),
                 (pl.col("player_team_code") == 3).sum().alias("ct_dead_f")))
    el_f = el_f.with_columns(
        pl.when((pl.col("ct_dead_f") == 5) & (pl.col("t_dead_f") < 5)).then(pl.lit(2))
        .when((pl.col("t_dead_f") == 5) & (pl.col("ct_dead_f") < 5)).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_elim_free")).select(KEY + ["s_elim_free"])
    base = base.join(el_f, on=KEY, how="left")

    # ---- elimination: D2 sx-based rule (replicated as in D2_truth, no victim dedupe)
    pi0 = tbl(label, "player_info", ["match_key", "round", "player_id", "team_code"]).unique(
        ["match_key", "round", "player_id"], keep="first", maintain_order=True)
    firstv = base.filter(pl.col("sx").is_not_null()).group_by("match_key").agg(pl.col("round").min().alias("F"))
    pim = pi0.join(firstv, on="match_key").filter(pl.col("round") == pl.col("F"))
    mem = pim.select(["match_key", "player_id", (pl.col("team_code") == 2).alias("isX")]).unique(["match_key", "player_id"])
    dx = dth.select(["match_key", "round", "player_id", "player_team_code"]).join(mem, on=["match_key", "player_id"], how="left")
    dx = dx.join(base.select(KEY + ["sx"]), on=KEY, how="left")
    dx = dx.with_columns(pl.when(pl.col("isX")).then(pl.col("sx")).otherwise(5 - pl.col("sx")).alias("vside"))
    el_sx = dx.group_by(KEY).agg((pl.col("vside") == 2).sum().alias("t_dead"), (pl.col("vside") == 3).sum().alias("ct_dead"))
    el_sx = el_sx.with_columns(
        pl.when((pl.col("ct_dead") == 5) & (pl.col("t_dead") < 5)).then(pl.lit(2))
        .when((pl.col("t_dead") == 5) & (pl.col("ct_dead") < 5)).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_elim_sx")).select(KEY + ["s_elim_sx"])
    base = base.join(el_sx, on=KEY, how="left")

    base = base.with_columns(
        pl.coalesce(["s_score_sx", "s_bomb_w", "s_elim_sx"]).alias("T_sx"),
        pl.coalesce(["s_score_free", "s_bomb_w", "s_elim_free"]).alias("T_free"),
        pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"),
        pl.lit(label).alias("year"),
        grp_expr())
    base = base.with_columns(
        (pl.col("R") != pl.col("T_sx")).alias("flip_sx"),
        (pl.col("R") != pl.col("T_free")).alias("flip_free"))
    return base


def pct(a, b):
    return "NA" if b == 0 else f"{100.0 * a / b:.2f}%"


def main():
    out = {}
    for label in ["2024", "2026"]:
        b = build(label)
        out[label] = b
        b.write_parquet(SCRATCH / f"base_{label}.parquet")
        print(f"\n################ {label} ################")
        print("base rows (round_end, player_count == %d, first row per match/round): %d" % (PC[label], b.height))
        print("with sx: %d   with T_sx: %d   with T_free: %d   with s_score_sx: %d   with s_score_free: %d"
              % (b.filter(pl.col("sx").is_not_null()).height, b.filter(pl.col("T_sx").is_not_null()).height,
                 b.filter(pl.col("T_free").is_not_null()).height, b.filter(pl.col("s_score_sx").is_not_null()).height,
                 b.filter(pl.col("s_score_free").is_not_null()).height))
        print("with s_bomb_w: %d   with s_bomb_u: %d   with s_elim_sx: %d   with s_elim_free: %d"
              % (b.filter(pl.col("s_bomb_w").is_not_null()).height, b.filter(pl.col("s_bomb_u").is_not_null()).height,
                 b.filter(pl.col("s_elim_sx").is_not_null()).height, b.filter(pl.col("s_elim_free").is_not_null()).height))

    # ---- reproduction against the claim's own T column (read-only)
    claim = pl.read_parquet(ROOT / "logs" / "swarm2" / "D2_truth_2024.parquet").select(
        KEY + [pl.col("T").alias("T_claim"), pl.col("flip").alias("flip_claim"), pl.col("R").alias("R_claim")])
    b24 = out["2024"].select(KEY + ["T_sx", "T_free", "R", "flip_sx", "flip_free"]).join(claim, on=KEY, how="full", coalesce=True)
    both = b24.filter(pl.col("T_claim").is_not_null() & pl.col("T_sx").is_not_null())
    print("\n== Reproduction of the claim's truth (2024) ==")
    print("claim rows with T:", claim.filter(pl.col("T_claim").is_not_null()).height,
          " my rows with T_sx:", out["2024"].filter(pl.col("T_sx").is_not_null()).height)
    print("rows with both T present:", both.height,
          " T_sx == T_claim:", both.filter(pl.col("T_sx") == pl.col("T_claim")).height,
          " T_sx != T_claim:", both.filter(pl.col("T_sx") != pl.col("T_claim")).height)
    print("rows with claim T but no T_sx:", b24.filter(pl.col("T_claim").is_not_null() & pl.col("T_sx").is_null()).height,
          " rows with T_sx but no claim T:", b24.filter(pl.col("T_sx").is_not_null() & pl.col("T_claim").is_null()).height)

    for label in ["2024", "2026"]:
        b = out[label]
        k_sx = b.filter(pl.col("T_sx").is_not_null())
        k_fr = b.filter(pl.col("T_free").is_not_null())
        print(f"\n== {label}: flip rate by round group, claim truth (T_sx) vs sx-free truth (T_free) ==")
        t1 = k_sx.group_by("grp").agg(pl.len().alias("n_sx"), pl.col("flip_sx").sum().alias("flip_sx"),
                                       pl.col("flip_sx").mean().round(4).alias("rate_sx")).sort("grp")
        t2 = k_fr.group_by("grp").agg(pl.len().alias("n_free"), pl.col("flip_free").sum().alias("flip_free"),
                                       pl.col("flip_free").mean().round(4).alias("rate_free")).sort("grp")
        print(t1.join(t2, on="grp", how="full", coalesce=True).sort("grp"))

    # ---- channel-by-channel evidence at rounds 13 / 25 / 26 / 27 / 28 / others (2024)
    b = out["2024"]
    print("\n== 2024: each sx-free channel vs recorded winner R, by round group (rows where the channel exists) ==")
    for ch in ["s_score_free", "s_bomb_w", "s_bomb_u", "s_elim_free", "s_score_sx", "s_elim_sx"]:
        k = b.filter(pl.col(ch).is_not_null())
        r = k.group_by("grp").agg(pl.len().alias("n"), (pl.col(ch) != pl.col("R")).sum().alias("R_ne_ch"),
                                   (pl.col(ch) != pl.col("R")).mean().round(4).alias("rate")).sort("grp")
        print(f"-- {ch}")
        print(r)

    print("\n== 2024: channel agreement on rows where both channels exist (by group) ==")
    pairs = [("s_score_free", "s_bomb_w"), ("s_score_free", "s_elim_free"), ("s_bomb_w", "s_elim_free"),
             ("s_score_free", "s_score_sx"), ("s_bomb_u", "s_bomb_w"), ("s_score_sx", "s_bomb_w")]
    for a, c in pairs:
        k = b.filter(pl.col(a).is_not_null() & pl.col(c).is_not_null())
        r = k.group_by("grp").agg(pl.len().alias("n"), (pl.col(a) == pl.col(c)).sum().alias("agree")).sort("grp")
        tot_n, tot_a = k.height, k.filter(pl.col(a) == pl.col(c)).height
        print(f"-- {a} vs {c}: n={tot_n} agree={tot_a} disagree={tot_n - tot_a}")
        print(r)

    # ---- bomb-channel mismatch by round number (sx-free, no team identity used)
    print("\n== 2024: s_bomb_w vs R by individual round number (13, 25, 26, 27, 28, 29, 30) ==")
    kb = b.filter(pl.col("s_bomb_w").is_not_null())
    print(kb.filter(pl.col("round").is_in([9, 13, 25, 26, 27, 28, 29, 30])).group_by("round").agg(
        pl.len().alias("n"), (pl.col("s_bomb_w") != pl.col("R")).sum().alias("mismatch")).sort("round"))

    # ---- the rounds 13/25/28 truth from sx-free channels only
    print("\n== 2024: round-13 rows where sx-free truth is available: flip counts ==")
    r13 = b.filter(pl.col("round") == 13)
    print("round 13 rows:", r13.height, " T_free known:", r13.filter(pl.col("T_free").is_not_null()).height,
          " flip_free:", r13.filter(pl.col("flip_free") == True).height,  # noqa: E712
          " flip_sx:", r13.filter(pl.col("flip_sx") == True).height)  # noqa: E712
    print("round 13 rows by which sx-free channel supplies T_free:")
    r13k = r13.filter(pl.col("T_free").is_not_null()).with_columns(
        pl.when(pl.col("s_score_free").is_not_null()).then(pl.lit("score_free"))
        .when(pl.col("s_bomb_w").is_not_null()).then(pl.lit("bomb_w")).otherwise(pl.lit("elim_free")).alias("src"))
    print(r13k.group_by("src").agg(pl.len().alias("n"), pl.col("flip_free").sum().alias("flipped")).sort("src"))

    # ---- match-level pattern at 13 / 25 / 28 (2024), sx-free truth and claim truth
    for col, lab in [("flip_free", "sx-free truth T_free"), ("flip_sx", "claim truth T_sx")]:
        pat = (b.filter(pl.col("round").is_in([13, 25, 28]) & pl.col(col).is_not_null())
               .select(["match_key", "round", col]).pivot(on="round", index="match_key", values=col))
        print(f"\n== 2024 match-level flip pattern at rounds 13/25/28 ({lab}) ==")
        print("columns:", pat.columns)
        if {"13", "25", "28"}.issubset(set(pat.columns)):
            full = pat.drop_nulls(["13", "25", "28"])
            print("matches with truth at 13, 25, 28:", full.height)
            print(full.group_by(["13", "25", "28"]).len().sort(["13", "25", "28"]))
        if "13" in pat.columns:
            with_13 = pat.drop_nulls(["13"])
            print("matches with truth at 13:", with_13.height, " flipped at 13:", with_13.filter(pl.col("13") == True).height)  # noqa: E712

    # ---- day split of round-13 flips (sx-free truth)
    print("\n== 2024 round 13 flips by match day (T_free) ==")
    print(r13.filter(pl.col("T_free").is_not_null()).group_by("day").agg(
        pl.len().alias("n"), pl.col("flip_free").sum().alias("flipped"), pl.col("flip_free").mean().round(3).alias("rate"))
        .sort("day"))


if __name__ == "__main__":
    main()
