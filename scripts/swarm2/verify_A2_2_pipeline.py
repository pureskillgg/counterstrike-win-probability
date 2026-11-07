# verify_A2_2_pipeline.py
# Pipeline-faithful effect of the rst_keys filter in the 2024 path (cross_period._run_2024 order):
#   channel filter -> roster (5 T and 5 CT via player_info) -> disconnect/bot takeover -> rst key -> live (freeze_end).
# Replica is validated against the logged counts (cross_period.md:58-62): 14960 / 4784 / 141 / 0 / 0, survivors 60351.
# Compares the current key (label k) with the claimed fix key (label k+1), and with no rst filter (live check only).
# ASSUMPTION P1: T = 2 and CT = 3 (winner_team_code convention used by bakeoff and the score test).
# ASSUMPTION P2: 2024 channel filter reproduced as: match has bomb_state rows and round_start rows (the only dense misses).
# ASSUMPTION P3: a round is identified by (match_key, round_end label); duplicate round_end keys collapse to one.
import os, json
import polars as pl

TOME24 = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
T, CT, START = 2, 3, 5


def read(base, cols, filt=None):
    names = [x for x in os.listdir(TOME24) if x.split(".", 1)[0] == base]
    assert len(names) == 1, (base, names)
    dp = os.path.join(TOME24, names[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = []
    for p in meta["pages"]:
        lf = pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols)
        if filt is not None:
            lf = lf.filter(filt)
        parts.append(lf.collect())
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]


def main():
    hdr = read("header", ["key"])
    all_m = set(hdr["key"].to_list())
    RE = read("basic_round_end", ["match_key", "round"]).with_columns(pl.col("round").cast(pl.Int64))
    RST = read("basic_round_start", ["match_key", "round"]).with_columns(pl.col("round").cast(pl.Int64))
    BS = read("basic_bomb_state", ["match_key"])
    bs_m = set(BS["match_key"].unique().to_list())
    rst_m = set(RST["match_key"].unique().to_list())
    kept = all_m & bs_m & rst_m
    print(f"header matches {len(all_m)}; kept after channel filter (P2): {len(kept)} (logged 4221)")

    re_k = RE.filter(pl.col("match_key").is_in(list(kept))).unique(["match_key", "round"], keep="first")
    rd = set(zip(re_k["match_key"].to_list(), re_k["round"].to_list()))
    print(f"round_end pairs in kept matches: {len(rd)} (logged 80236)")

    PI = read("basic_player_info", ["match_key", "round", "player_id", "team_code"]).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("team_code").cast(pl.Int64, strict=False))
    PI = PI.filter(pl.col("match_key").is_in(list(kept))).unique(["match_key", "round", "player_id"], keep="last")
    cnt = PI.group_by(["match_key", "round"]).agg(
        (pl.col("team_code") == T).sum().alias("n_t"), (pl.col("team_code") == CT).sum().alias("n_ct"))
    roster_ok = set(zip(cnt.filter((pl.col("n_t") == START) & (pl.col("n_ct") == START))["match_key"].to_list(),
                        cnt.filter((pl.col("n_t") == START) & (pl.col("n_ct") == START))["round"].to_list()))

    disc = set()
    for base in ["basic_player_disconnect", "basic_bot_takeover"]:
        try:
            X = read(base, ["match_key", "round"]).with_columns(pl.col("round").cast(pl.Int64))
            X = X.filter(pl.col("match_key").is_in(list(kept)))
            disc |= set(zip(X["match_key"].to_list(), X["round"].to_list()))
        except Exception as e:
            print("  note: could not read", base, str(e)[:80])

    ST = read("basic_round_state", ["match_key", "round", "event_type"],
              filt=pl.col("event_type") == "round_freeze_end").with_columns(pl.col("round").cast(pl.Int64))
    live = set(zip(ST["match_key"].to_list(), ST["round"].to_list()))

    rst_cur = set(zip(RST["match_key"].to_list(), RST["round"].to_list()))
    rst_fix = set((m, k + 1) for (m, k) in rst_cur)
    # per-match last label and round-1 flags
    last = {}
    for (m, k) in rd:
        last[m] = max(last.get(m, 0), k)

    stage = {"roster": 0, "disc": 0, "rst": 0, "live": 0}
    surv_cur, surv_fix, surv_norst = 0, 0, 0
    fix_drop_r1 = 0
    fix_drop_r1_pass = 0
    cur_drop_last_pass = 0
    fix_keep_last_pass = 0
    cur_drop_r1_pass = 0
    for (m, k) in rd:
        if (m, k) not in roster_ok:
            stage["roster"] += 1
            continue
        if (m, k) in disc:
            stage["disc"] += 1
            continue
        is_last = (k == last[m])
        is_r1 = (k == 1)
        # current
        if (m, k) not in rst_cur:
            stage["rst"] += 1
            if is_last:
                cur_drop_last_pass += 1
            if is_r1:
                cur_drop_r1_pass += 1
        elif (m, k) in live:
            surv_cur += 1
        if (m, k) in live:
            surv_norst += 1
        # claimed fix
        if (m, k) not in rst_fix:
            if is_r1:
                fix_drop_r1_pass += 1
        elif (m, k) in live:
            surv_fix += 1
            if is_last:
                fix_keep_last_pass += 1
        if (m, k) in live and (m, k) in rst_cur:
            pass
    print(f"replica stages (logged 14960 / 4784 / 141 / 0 / 0):")
    print(f"  dropped by roster filter: {stage['roster']}")
    print(f"  dropped by disconnect/bot takeover: {stage['disc']}")
    print(f"  dropped by rst check (current key, label k): {stage['rst']} (of which final round: {cur_drop_last_pass}; round 1: {cur_drop_r1_pass})")
    print(f"  survivors after rst and live, current key: {surv_cur} (logged 60351 before duplicate and death stages)")
    print(f"claimed fix (label k+1 key), same stages:")
    print(f"  rounds reaching rst check that the fix drops: round 1: {fix_drop_r1_pass}")
    print(f"  survivors after rst and live, fix key: {surv_fix}; difference vs current: {surv_fix - surv_cur:+d}")
    print(f"  final rounds kept by fix key (reached and passed live): {fix_keep_last_pass}")
    print(f"no rst filter at all (live check only, same earlier stages): survivors {surv_norst}; difference vs current: {surv_norst - surv_cur:+d}")


if __name__ == "__main__":
    main()
    print("\nDONE")
