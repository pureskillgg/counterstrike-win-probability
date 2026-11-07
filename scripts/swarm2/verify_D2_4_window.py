"""Verifier D2-4 diagnostic: where do the 2024 bomb events fall relative to the round window, and does the
claimed 12,920 score-vs-bomb count depend on the round_end row used for the window end?

Also checks the death-membership gap (deaths whose player_id is not in the round-F roster).
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_4_window.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import polars as pl  # noqa: E402

from verify_D2_4_build import PC, bin_expr, build, channel  # noqa: E402

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)


def hdr(s):
    print("\n" + "=" * 10 + " " + s + " " + "=" * 10)


label = "2024"
pc = PC[label]
b = build(label).with_columns(bin_expr())   # claim-style (pc20 round_end row, window slack 40)

# all round_end rows, including the other player_count values
re_all = channel(label, "round_end", ["match_key", "round", "tick", "player_count"])
re_first_any = (re_all.sort(["match_key", "round", "tick"])
                .unique(["match_key", "round"], keep="first", maintain_order=True)
                .select(["match_key", "round", pl.col("tick").alias("t_end_any")]))
re_pc = re_all.filter(pl.col("player_count") == pc).sort(["match_key", "round", "tick"]).unique(
    ["match_key", "round"], keep="first", maintain_order=True).select(
    ["match_key", "round", pl.col("tick").alias("t_end_pc")])
print("(match, round) with more than one round_end row:",
      re_all.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height)
dup = re_all.group_by(["match_key", "round"]).agg(pl.col("player_count").n_unique().alias("npc"), pl.len().alias("n"))
print("  of which with different player_count values:", dup.filter(pl.col("npc") > 1).height)

bs = channel(label, "bomb_state", ["match_key", "round", "tick", "event_type"])
bo = bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
first_bo_any = (bo.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("tick").first().alias("t_bo"), pl.col("event_type").first().alias("bo_first_any"),
    pl.col("event_type").n_unique().alias("bo_types_any")))
rr = (b.select(["match_key", "round", "t_fe", "t_end", "R", "T", "s_side", "s_score", "s_bomb", "flip", "bin", "day"])
      .join(first_bo_any, on=["match_key", "round"], how="inner")
      .join(re_first_any, on=["match_key", "round"], how="left")
      .join(re_pc, on=["match_key", "round"], how="left"))
rr = rr.with_columns(
    (pl.col("t_bo") - pl.col("t_end")).alias("off_end"),
    (pl.col("t_bo") - pl.col("t_fe")).alias("off_fe"),
    pl.when(pl.col("bo_first_any") == "bomb_exploded").then(2).otherwise(3).alias("s_bomb_any"),
)
print("\n2024 pc20 rounds with any bomb_exploded/defused event:", rr.height)
print("  in claim window [t_fe, t_end+40]:", rr.filter(pl.col("s_bomb").is_not_null()).height)
print("  outside claim window:", rr.filter(pl.col("s_bomb").is_null()).height)
out = rr.filter(pl.col("s_bomb").is_null())
print("\noutside-window rows: offset (tick - t_end) quantiles:")
print(out.select(pl.col("off_end").quantile(q).alias(f"q{int(q*100)}") for q in [0.0, 0.1, 0.5, 0.9, 1.0]))
print("outside-window rows: offset (tick - t_fe) quantiles:")
print(out.select(pl.col("off_fe").quantile(q).alias(f"q{int(q*100)}") for q in [0.0, 0.1, 0.5, 0.9, 1.0]))
print("outside-window rows by bin:")
print(out.group_by("bin").len().sort("bin"))
print("outside-window rows with t_fe null:", out.filter(pl.col("t_fe").is_null()).height)
print("outside-window: off_end > 40:", out.filter(pl.col("off_end") > 40).height,
      "  off_end < 0:", out.filter(pl.col("off_end") < 0).height,
      "  off_fe < 0:", out.filter(pl.col("off_fe") < 0).height)
print("outside-window: first-any event agrees with s_side (where s_side known):",
      out.filter(pl.col("s_side").is_not_null()).height, "rows,",
      out.filter(pl.col("s_side").is_not_null() & (pl.col("s_side") == pl.col("s_bomb_any"))).height, "equal")
print("outside-window: R == first-any bomb side:",
      out.filter(pl.col("R") == pl.col("s_bomb_any")).height, "of", out.height)
os_ = out.filter(pl.col("s_side").is_not_null())
print("outside-window rows with s_side:", os_.height, "  R != s_side:", os_.filter(pl.col("R") != pl.col("s_side")).height,
      "  by bin (n / R!=side):")
print(os_.group_by("bin").agg(pl.len().alias("n"), (pl.col("R") != pl.col("s_side")).sum().alias("R!=side")).sort("bin"))
print("in-window rows with s_side (same definition):",
      rr.filter(pl.col("s_bomb").is_not_null() & pl.col("s_side").is_not_null()).height, "  R != s_side:",
      rr.filter(pl.col("s_bomb").is_not_null() & pl.col("s_side").is_not_null() & (pl.col("R") != pl.col("s_side"))).height)

hdr("claimed 12,920 check: score vs bomb with alternative window ends")
for name, col in [("claim (pc20 round_end, +40)", "t_end"), ("any-pc first round_end, +40", "t_end_any"),
                  ("pc20 round_end, no slack", "t_end_pc")]:
    w = rr.filter(pl.col(col).is_not_null())
    for slack in ([40] if name.startswith("claim") or name.startswith("any") else [0]):
        win = w.filter((pl.col("t_bo") >= pl.col("t_fe")) & (pl.col("t_bo") <= pl.col(col) + slack))
        ok = win.filter(pl.col("s_score").is_not_null())
        print(f"{name:32s} in-window bomb rounds: {win.height:6d}  with s_score: {ok.height:6d}  "
              f"s_score==bomb: {ok.filter(pl.col('s_score') == pl.col('s_bomb_any')).height}")

hdr("any-tick bomb (no window) vs score: reproduces the claimed 12,920 denominator?")
anyb = rr.filter(pl.col("s_score").is_not_null())
print("rounds with s_score and any-tick bomb:", anyb.height, "  s_score==any-tick bomb:",
      anyb.filter(pl.col("s_score") == pl.col("s_bomb_any")).height)
print("  of which out-of-window (late bomb):", anyb.filter(pl.col("s_bomb").is_null()).height,
      "  s_score==late bomb:", anyb.filter((pl.col("s_bomb").is_null()) & (pl.col("s_score") == pl.col("s_bomb_any"))).height)

hdr("death membership gap")
pi = channel(label, "player_info", ["match_key", "round", "player_id", "team_code"]).unique(
    ["match_key", "round", "player_id"], keep="first", maintain_order=True)
any_ids = pi.select(["match_key", "player_id"]).unique().with_columns(pl.lit(True).alias("in_pi_any"))
dth = channel(label, "player_death", ["match_key", "round", "player_id"]).join(
    any_ids, on=["match_key", "player_id"], how="left")
print("player_death rows (2024):", dth.height)
print("  player_id present in player_info of the same match (any round):", dth.filter(pl.col("in_pi_any").is_not_null()).height)
print("  player_id absent from that match's player_info entirely:", dth.filter(pl.col("in_pi_any").is_null()).height)
print("  top absent player_id values:")
print(dth.filter(pl.col("in_pi_any").is_null()).group_by("player_id").len().sort("len", descending=True).head(8))
# same round: is the player present in player_info for that round?
same_rd = pi.select(["match_key", "round", "player_id"]).with_columns(pl.lit(True).alias("in_round"))
dr = dth.join(same_rd, on=["match_key", "round", "player_id"], how="left")
print("  death player present in player_info of the same round:", dr.filter(pl.col("in_round").is_not_null()).height,
      "  not present in that round:", dr.filter(pl.col("in_round").is_null()).height)
