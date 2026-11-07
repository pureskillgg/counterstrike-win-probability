# VERIFIER for A1-1, part 2: engine-side checks that avoid the side-swap assumption.
# score_update team_code is a SIDE code (2/3). At halftime the engine writes swap rows that move a team's
# score to the other code. Final per-slot score = new_score of the chronologically last row for that code.
# Checks:
#  (a) engine final sum (last row per code, summed) vs round_end rows and header S, all 1752 matches;
#  (b) last-round game-over test: the winner of the last recorded round (round_end winner_team_code) has
#      new_score == win threshold on its last score_update row at that round (no swap rule used);
#  (c) round-level channels: max round per match in round_start / round_state / round_end / score_update /
#      bomb_plant-type channels vs n, to see whether any channel has rows past the last round_end;
#  (d) the 2 non-game-over shortfall matches: the engine score path.
# ASSUMPTION: win threshold = max_rounds // 2 + 1.
import os, glob
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
def pages(ch):
    return sorted(glob.glob(os.path.join(T, ch, "dataframe_*")))
def chdir(prefix):
    d = [x for x in os.listdir(T) if x.startswith(prefix + ".") or x == prefix]
    assert len(d) == 1, (prefix, d)
    return d[0]

h = pl.concat([pl.scan_parquet(p).select(["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final"])
               for p in pages("header")], how="vertical").collect()
h = h.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("S"),
                   (pl.col("max_rounds") // 2 + 1).alias("thr"))

re_ = pl.concat([pl.scan_parquet(p).select(["match_key", "round", "winner_team_code"]) for p in pages(chdir("basic_round_end"))],
                how="diagonal_relaxed").collect()
rows_n = re_.group_by("match_key").agg(pl.len().alias("rows"), pl.col("round").max().alias("last_round"))
last_winner = (re_.sort(["match_key", "round"]).group_by("match_key").last()
               .select(["match_key", pl.col("round").alias("lr"), pl.col("winner_team_code").alias("lw")]))

su = pl.concat([pl.scan_parquet(p).select(["match_key", "round", "tick", "team_code", "old_score", "new_score"])
                for p in pages(chdir("basic_score_update"))], how="diagonal_relaxed").collect()
su = su.sort(["match_key", "round", "tick"])
print("score_update rows", su.height, "matches", su.get_column("match_key").n_unique())
# (a) last row per code
lastcode = su.group_by(["match_key", "team_code"], maintain_order=True).last().select(
    ["match_key", "team_code", "round", "tick", "new_score"])
eng = lastcode.group_by("match_key").agg([pl.col("new_score").sum().alias("eng_sum"), pl.len().alias("ncodes"),
                                          pl.col("new_score").max().alias("eng_max")])
m = (h.join(rows_n, left_on="key", right_on="match_key", how="left")
      .join(eng, left_on="key", right_on="match_key", how="left")
      .join(last_winner, left_on="key", right_on="match_key", how="left"))
print("matches", m.height, "| with exactly 2 codes:", m.filter(pl.col("ncodes") == 2).height)
print("(a) eng_sum == rows:", m.filter(pl.col("eng_sum") == pl.col("rows")).height,
      "| eng_sum == S:", m.filter(pl.col("eng_sum") == pl.col("S")).height,
      "| eng_sum == S & rows agree (1672 set):", m.filter((pl.col("S") == pl.col("rows")) & (pl.col("eng_sum") == pl.col("rows"))).height,
      "of", m.filter(pl.col("S") == pl.col("rows")).height)
sh = m.filter(pl.col("S") > pl.col("rows"))
print("(a) shortfall n =", sh.height, "| eng_sum == rows:", sh.filter(pl.col("eng_sum") == pl.col("rows")).height,
      "| eng_sum == S:", sh.filter(pl.col("eng_sum") == pl.col("S")).height)
print("(a) shortfall: last_round == rows (contiguous, no gap):", sh.filter(pl.col("last_round") == pl.col("rows")).height, "of", sh.height)

# (b) last-round game-over test (no swap rule)
lw = (su.join(last_winner, on="match_key", how="inner")
        .filter((pl.col("round") == pl.col("lr")) & (pl.col("team_code") == pl.col("lw"))))
lw_last = lw.group_by("match_key").agg(pl.col("new_score").last().alias("winner_score_last_round"))
m = m.join(lw_last, left_on="key", right_on="match_key", how="left")
print("(b) matches where last-round winner has no score_update row at that round:",
      m.filter(pl.col("winner_score_last_round").is_null()).height)
m = m.with_columns((pl.col("winner_score_last_round") == pl.col("thr")).alias("gameover_at_last_round"))
print("(b) ALL matches: last-round winner new_score == thr:", m.filter(pl.col("gameover_at_last_round")).height, "of", m.height)
print("(b) SHORTFALL: last-round winner new_score == thr:", m.filter((pl.col("S") > pl.col("rows")) & pl.col("gameover_at_last_round")).height,
      "of", m.filter(pl.col("S") > pl.col("rows")).height)
print("(b) S==rows: last-round winner new_score == thr:", m.filter((pl.col("S") == pl.col("rows")) & pl.col("gameover_at_last_round")).height,
      "of", m.filter(pl.col("S") == pl.col("rows")).height)
print("(b) shortfall non-gameover:")
print(m.filter((pl.col("S") > pl.col("rows")) & ~pl.col("gameover_at_last_round")).select(
    ["key", "max_rounds", "rows", "S", "t_starters_score_final", "ct_starters_score_final", "winner_score_last_round", "eng_sum"]).rows())

# (c) round-level channels: any rows past the last round_end round?
chans = ["basic_round_start", "basic_round_state", "basic_round_end", "basic_score_update", "basic_round_mvp",
         "basic_bomb_action", "basic_player_death"]
short_keys = sh.get_column("key").to_list()
print("(c) per-channel count of shortfall matches with max round > last round_end round (or rows past n):")
for ch in chans:
    try:
        d = chdir(ch)
    except AssertionError as e:
        print("   skip", ch, "not found"); continue
    cols = pl.scan_parquet(pages(d)[0]).collect_schema().names()
    if "round" not in cols or "match_key" not in cols:
        print("   skip", ch, "no round/match_key"); continue
    x = pl.concat([pl.scan_parquet(p).select(["match_key", "round"]) for p in pages(d)], how="diagonal_relaxed").collect()
    mx = x.group_by("match_key").agg(pl.col("round").max().alias("mx"), pl.len().alias("nrows"))
    j = mx.join(m.select(["key", "rows", "S"]), left_on="match_key", right_on="key", how="inner")
    past = j.filter(pl.col("mx") > pl.col("rows"))
    shj = j.filter(pl.col("match_key").is_in(short_keys))
    print("   ", ch, "matches with channel rows:", j.height, "| max round > rows (all matches):", past.height,
          "| shortfall: max round == rows:", shj.filter(pl.col("mx") == pl.col("rows")).height, "of", shj.height,
          "| shortfall max round > rows:", shj.filter(pl.col("mx") > pl.col("rows")).height)
print("done")
