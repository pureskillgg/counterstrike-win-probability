"""D1-9 verification, part 2: is 2024 player_count a count, doubled, or a constant? (read-only)

Tests the strongest alternatives to the claim's 'apparently doubled' reading:
  (a) player_info is incomplete in the rounds where pc=20 but few players are listed
      (test: match-level distinct players, and team split inside those rounds).
  (b) the 2024 value is a match-level constant (test: pc varies within match or not).
  (c) the value marks rebuilt rows (test: crosstab with win_reason_code in {8, 9}).
Also looks at the single 2024 round_end duplicate pair (match 2f5bcb89, round 1).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(90)
pl.Config.set_tbl_width_chars(200)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
S24 = ".2023-12-28,2024-02-25"
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
S26 = ".2026-09-18,2026-10-08"
T_CODE, CT_CODE = 2, 3


def lf(base, suf, ch):
    fs = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no files for {ch} under {base}")
    return pl.scan_parquet(fs)


def hdr(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


RE24 = lf(T24, S24, "round_end").select(
    "match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count").collect()
PI24 = lf(T24, S24, "player_info").select(
    "match_key", "round", "player_id", "player_id_fixed", "team_code").collect()
RE26 = lf(T26, S26, "round_end").select(
    "match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count").collect()
PI26 = lf(T26, S26, "player_info").select(
    "match_key", "round", "player_id", "player_id_fixed", "team_code").collect()


def build(RE, PI):
    PR = PI.group_by(["match_key", "round"]).agg(
        pl.col("player_id_fixed").n_unique().alias("n"),
        pl.col("player_id_fixed").filter(pl.col("team_code") == T_CODE).n_unique().alias("n_t"),
        pl.col("player_id_fixed").filter(pl.col("team_code") == CT_CODE).n_unique().alias("n_ct"),
    )
    MP = PI.group_by("match_key").agg(
        pl.col("player_id_fixed").n_unique().alias("m_pid"),
        pl.col("round").n_unique().alias("m_rounds_pi"),
    )
    RU = RE.unique(["match_key", "round"], keep="first", maintain_order=True)
    J = RU.join(PR, on=["match_key", "round"], how="left").join(MP, on="match_key", how="left")
    return J


J24 = build(RE24, PI24)
J26 = build(RE26, PI26)

# ---------------------------------------------------------------- (c) win code marker
hdr("(c) 2024 player_count == 20 vs win_reason_code in {8, 9} (rebuilt-row marker test)")
x = RE24.with_columns(
    code89=pl.col("win_reason_code").is_in([8, 9]),
    pc20=pl.col("player_count") == 20,
)
print(x.group_by(["code89", "pc20"]).len().sort(["code89", "pc20"]))
print("2024 rows with win_reason_code in {8,9}:", x.filter(pl.col("code89")).height,
      "of", x.height, f"({100 * x.filter(pl.col('code89')).height / x.height:.2f}%)")
print("docs claim (CL F6): rebuilt 2024-02-12..2026-08-03 rows carry only codes 8 and 9.")
print("2024 win codes outside {8,9}:",
      x.filter(~pl.col("code89")).group_by("win_reason_code").len().sort("win_reason_code").to_dicts())
print("2024 winner_team_code by win_reason_code:")
print(RE24.group_by(["win_reason_code", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"]))

# ---------------------------------------------------------------- (a) is player_info incomplete, or are these real small rosters?
hdr("(a) 2024 rounds with pc == 20 but few listed players: match-level distinct players")
K = J24.filter(pl.col("n").is_not_null())
print("2024 match-level distinct player_id_fixed (m_pid) for rounds with pc=20 by listed n:")
print(K.filter(pl.col("player_count") == 20).group_by(["n", "m_pid"]).len()
      .sort(["n", "len"], descending=[False, True]).filter(pl.col("n") < 10).head(60))
print("\n2024 rounds with n == 10 (pc == 20 in most), match-level m_pid distribution:")
print(K.filter(pl.col("n") == 10).group_by("m_pid").len().sort("len", descending=True).head(10))

print("\n2024 rounds with n == 4: team split (n_t, n_ct) and pc:")
print(K.filter(pl.col("n") == 4).group_by(["n_t", "n_ct", "player_count"]).len()
      .sort("len", descending=True).head(20))
print("\n2024 rounds with n == 9: team split (n_t, n_ct) and pc:")
print(K.filter(pl.col("n") == 9).group_by(["n_t", "n_ct", "player_count"]).len()
      .sort("len", descending=True).head(20))

# matches where some rounds have 4 listed players and others 10 listed players
per_match = K.group_by("match_key").agg(
    pl.len().alias("rounds"),
    (pl.col("n") == 10).sum().alias("r10"),
    (pl.col("n") == 4).sum().alias("r4"),
    (pl.col("n") == 9).sum().alias("r9"),
    pl.col("player_count").unique().sort().alias("pcs"),
    pl.col("m_pid").first().alias("m_pid"),
)
mixed = per_match.filter((pl.col("r4") > 0) & (pl.col("r10") > 0))
pure4 = per_match.filter((pl.col("r4") > 0) & (pl.col("r10") == 0))
print("2024 matches with both 4-listed and 10-listed rounds:", mixed.height)
print("2024 matches with 4-listed rounds and no 10-listed round:", pure4.height)
print("  their match-level m_pid:", pure4.group_by("m_pid").len().sort("m_pid").to_dicts())
print("  their pc values:", pure4["pcs"].explode().value_counts().sort("count", descending=True).to_dicts())
print("2024 matches with 4-listed rounds and any 10-listed round: pc values:",
      mixed["pcs"].explode().value_counts().sort("count", descending=True).to_dicts())
print("2024 matches with 4-listed rounds: count of rounds listing 4 players in mixed matches:",
      int(mixed["r4"].sum()))

# ---------------------------------------------------------------- (b) pc within match
hdr("(b) 2024 pc constant within match? (rounds per match, pc values, listed-n values)")
pcm = RE24.group_by("match_key").agg(pl.col("player_count").n_unique().alias("npc"),
                                     pl.col("player_count").mode().first().alias("pc_mode"))
print("2024 matches with constant pc:", pcm.filter(pl.col("npc") == 1).height,
      "of", pcm.height)
mix_pc = pcm.filter(pl.col("npc") > 1)
print("2024 matches with mixed pc:", mix_pc.height)
ex = mix_pc.head(3)["match_key"].to_list()
for mk in ex:
    rr = (RE24.filter(pl.col("match_key") == mk).sort("round")
          .join(PI24.group_by(["match_key", "round"]).agg(pl.col("player_id_fixed").n_unique().alias("n")),
                on=["match_key", "round"], how="left")
          .select(["round", "tick", "winner_team_code", "player_count", "n"]))
    print("example", mk)
    print(rr)

# ---------------------------------------------------------------- 2026 comparison: wingman rounds
hdr("2026 comparison: pc vs listed players (is pc a real count in 2026?)")
K26 = J26.filter(pl.col("n").is_not_null())
print("2026 rounds:", K26.height, " pc == n:", int((K26["player_count"] == K26["n"]).sum()),
      " pc == 2n:", int((K26["player_count"] == 2 * K26["n"]).sum()))
print(K26.group_by(["player_count", "n"]).len().sort("len", descending=True).head(15))

# ---------------------------------------------------------------- duplicate round_end pair
hdr("Duplicate round_end pair: match csds/2024/02/12/2f5bcb89-... round 1")
mk = "csds/2024/02/12/2f5bcb89-9ef8-4788-9f3a-2fa3d633ea4d/csds"
m = RE24.filter(pl.col("match_key") == mk).sort(["round", "tick"])
print("round_end rows in this match:", m.height, " distinct rounds:", m["round"].n_unique(),
      " round range:", m["round"].min(), m["round"].max())
print(m.head(6))
print("rows with round == 1:")
print(m.filter(pl.col("round") == 1))
rnd_counts = m.group_by("round").len().filter(pl.col("len") > 1)
print("rounds with >1 round_end row in this match:", rnd_counts.to_dicts())
pi_m = PI24.filter(pl.col("match_key") == mk)
print("player_info rows in this match:", pi_m.height,
      " distinct player_id_fixed:", pi_m["player_id_fixed"].n_unique())

print("\nDONE")
