# VERIFIER for A1-1, part 3: independent end-of-match evidence.
#  (a) tome basic_round_state: presence and round of event 'cs_win_panel_match' (engine match-end signal),
#      and the engine's final t_score + ct_score on the last round_state row, vs n and header S.
#  (b) raw E: tick file (read only, 'round','tick','previous_phase'): the final phase, whether a 'postgame'
#      tail exists, the first postgame round, and whether any tick has round > n. Gap between the last
#      round_end tick and the last tick, in ticks (tick rate 64 per second, ASSUMPTION from header tick_rate).
#  Controls: 40 seeded-random matches with S == rows and threshold-ended, to compare tails.
# Reads E: only (no writes). Memory: tick files are ~0.5 MB each.
import os, glob, random
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
E = r"E:\csds_final\csds\2026\10"
def pages(ch):
    return sorted(glob.glob(os.path.join(T, ch, "dataframe_*")))
def chdir(prefix):
    d = [x for x in os.listdir(T) if x.startswith(prefix + ".")]
    assert len(d) == 1, (prefix, d)
    return d[0]

h = pl.concat([pl.scan_parquet(p).select(["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final",
                                           "tick_rate"]) for p in pages("header")], how="vertical").collect()
h = h.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("S"),
                   (pl.col("max_rounds") // 2 + 1).alias("thr"))
re_ = pl.concat([pl.scan_parquet(p).select(["match_key", "round", "tick"]) for p in pages(chdir("basic_round_end"))],
                how="diagonal_relaxed").collect()
rn = re_.group_by("match_key").agg(pl.len().alias("rows"), pl.col("round").max().alias("n"),
                                    pl.col("tick").max().alias("last_re_tick"))
m = h.join(rn, left_on="key", right_on="match_key", how="left")

# (a) tome round_state
rs = pl.concat([pl.scan_parquet(p).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
                for p in pages(chdir("basic_round_state"))], how="diagonal_relaxed").collect()
wp = (rs.filter(pl.col("event_type") == "cs_win_panel_match")
        .group_by("match_key").agg(pl.len().alias("n_wp"), pl.col("round").max().alias("wp_round"),
                                   pl.col("t_score").last().alias("wp_t"), pl.col("ct_score").last().alias("wp_ct")))
lastrs = rs.sort(["match_key", "tick"]).group_by("match_key").last().select(
    ["match_key", pl.col("t_score").alias("fin_t"), pl.col("ct_score").alias("fin_ct"), pl.col("round").alias("fin_round")])
m = m.join(wp, left_on="key", right_on="match_key", how="left").join(lastrs, left_on="key", right_on="match_key", how="left")
m = m.with_columns((pl.col("fin_t") + pl.col("fin_ct")).alias("rs_sum"))
print("(a) matches:", m.height, "| with cs_win_panel_match event:", m.filter(pl.col("n_wp").is_not_null()).height)
print("(a) matches with win panel exactly once:", m.filter(pl.col("n_wp") == 1).height)
print("(a) win panel round == n (rows):", m.filter(pl.col("wp_round") == pl.col("n")).height, "of", m.filter(pl.col("n_wp").is_not_null()).height)
print("(a) final round_state t+ct == rows:", m.filter(pl.col("rs_sum") == pl.col("rows")).height, "| == S:", m.filter(pl.col("rs_sum") == pl.col("S")).height, "of", m.height)
sh = m.filter(pl.col("S") > pl.col("rows"))
print("(a) SHORTFALL n =", sh.height, "| has win panel:", sh.filter(pl.col("n_wp").is_not_null()).height,
      "| win panel round == n:", sh.filter(pl.col("wp_round") == pl.col("n")).height,
      "| final round_state t+ct == rows:", sh.filter(pl.col("rs_sum") == pl.col("rows")).height,
      "| final round_state t+ct == S:", sh.filter(pl.col("rs_sum") == pl.col("S")).height)
print("(a) SHORTFALL wp (t,ct) == round_state final:", sh.filter((pl.col("wp_t") == pl.col("fin_t")) & (pl.col("wp_ct") == pl.col("fin_ct"))).height)
print("(a) SHORTFALL win-panel-round list (key, n, wp_round, wp_t, wp_ct, S, t_hdr, ct_hdr):")
print(sh.filter(pl.col("n_wp").is_null()).select(["key", "rows", "S"]).rows())
print("(a) normal (S==rows) win panel present:", m.filter((pl.col("S") == pl.col("rows")) & pl.col("n_wp").is_not_null()).height,
      "of", m.filter(pl.col("S") == pl.col("rows")).height)
print("(a) header t/ct vs round_state final for shortfall, first 12:")
print(sh.select(["key", "rows", "S", "t_starters_score_final", "ct_starters_score_final", "fin_t", "fin_ct"]).head(12).rows())

# (b) raw tick tails
rng = random.Random(11)
short_keys = sh.get_column("key").to_list()
normal_ok = m.filter((pl.col("S") == pl.col("rows")) & (pl.col("rows") >= pl.col("max_rounds") // 2 + 1))
ctrl_keys = rng.sample(normal_ok.get_column("key").to_list(), 40)
def raw_base(key):
    p = key.split("/")
    return os.path.join(E, p[3], p[4])
def tick_tail(key, n, last_re_tick):
    base = raw_base(key)
    tk = pl.read_parquet(os.path.join(base, "tick"), columns=["round", "tick", "previous_phase"])
    tk = tk.sort("tick")
    last = tk.tail(1).row(0)
    post = tk.filter(pl.col("previous_phase") == "postgame")
    first_post_tick = int(post.get_column("tick").min()) if post.height else None
    first_post_round = int(post.get_column("round").min()) if post.height else None
    max_round = int(tk.get_column("round").max())
    return dict(last_round=int(last[0]), last_tick=int(last[1]), last_phase=last[2],
                n_post=post.height, first_post_tick=first_post_tick, first_post_round=first_post_round,
                max_round=max_round, gap_last_re_to_end=int(last[1]) - int(last_re_tick),
                gap_last_re_to_firstpost=(first_post_tick - int(last_re_tick)) if first_post_tick is not None else None)
def summarize(label, keys, mm):
    rows = []
    for k in keys:
        rec = mm.filter(pl.col("key") == k).row(0, named=True)
        try:
            tt = tick_tail(k, rec["n"], rec["last_re_tick"])
        except Exception as ex:
            print("   ERR", k, ex); continue
        rows.append((k, rec["n"], rec["S"], tt))
    print(label, "count", len(rows))
    print("   last tick phase:", {p: sum(1 for _, _, _, t in rows if t["last_phase"] == p) for p in set(t["last_phase"] for _, _, _, t in rows)})
    print("   has postgame tail:", sum(1 for _, _, _, t in rows if t["n_post"] > 0), "of", len(rows))
    print("   first postgame round == n:", sum(1 for _, n, _, t in rows if t["first_post_round"] == n), "of", len(rows))
    print("   max tick round > n (tick file runs past last round_end):", sum(1 for _, n, _, t in rows if t["max_round"] > n), "of", len(rows))
    g = sorted(t["gap_last_re_to_end"] for _, _, _, t in rows)
    gp = sorted(t["gap_last_re_to_firstpost"] for _, _, _, t in rows if t["gap_last_re_to_firstpost"] is not None)
    if g:
        print("   gap last round_end tick -> last tick (ticks): min", g[0], "median", g[len(g)//2], "max", g[-1])
    if gp:
        print("   gap last round_end tick -> first postgame tick (ticks): min", gp[0], "median", gp[len(gp)//2], "max", gp[-1])
    return rows
sh_rows = summarize("(b) SHORTFALL (80)", short_keys, m)
ctrl_rows = summarize("(b) CONTROL normal matches (40, seed 11)", ctrl_keys, m)
print("(b) shortfall detail, non-threshold or OT (rows > max_rounds or below threshold):")
for k, n, S, t in sh_rows:
    mm = m.filter(pl.col("key") == k).row(0, named=True)
    if n > mm["max_rounds"] or n < mm["thr"]:
        print("   ", k, "n=", n, "S=", S, "max_rounds=", mm["max_rounds"], "last_phase=", t["last_phase"],
              "postgame_ticks=", t["n_post"], "first_post_round=", t["first_post_round"], "gap_to_end=", t["gap_last_re_to_end"],
              "fin_rs=", (mm["fin_t"], mm["fin_ct"]), "wp_round=", mm["wp_round"])
print("done")
