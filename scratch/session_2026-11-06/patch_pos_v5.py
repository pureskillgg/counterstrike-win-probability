"""Make pos_v5_keep.py from pos_bucket_every_second_keep.py: explicit side override, and only the affected matches.

Changes (each asserted to match exactly once):
  1. load the override table (ovr_all.parquet) and the 45 affected match keys;
  2. join the override onto each player row, and take side = override, then roster, then player_vector code;
  3. skip matches that are not affected.
Everything else (features, bomb lookup, grid, output format) is unchanged.
"""
import sys

SRC = "/home/sagemaker-user/adam/builder/fixes/pos_every_second/pos_bucket_every_second_keep.py"
DST = "/home/sagemaker-user/adam/builder/fixes/fix332/v5_build/pos_v5_keep.py"
src = open(SRC, encoding="utf-8").read()


def rep(s, old, new):
    n = s.count(old)
    assert n == 1, f"expected 1 match, found {n}: {old[:70]!r}"
    return s.replace(old, new)


src = rep(src,
          'import make_pos_features as MPF  # noqa: E402',
          'import make_pos_features as MPF  # noqa: E402\n'
          'OVR_PATH = os.environ.get("POSB_OVR", "/home/sagemaker-user/adam/builder/fixes/fix332/v5_build/ovr_all.parquet")\n'
          'AFF_PATH = os.environ.get("POSB_AFF", "/home/sagemaker-user/adam/builder/fixes/fix332/v5_build/affected_matches.txt")\n'
          'OVR = pl.read_parquet(OVR_PATH).select("match_key", pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64),\n'
          '                                        pl.col("side").cast(pl.Int64).alias("ov_side")).unique(["match_key", "round", "player_id"])\n'
          'AFFECTED = set(l.strip() for l in open(AFF_PATH) if l.strip())\n')

src = rep(src,
          '        .join(ro.rename({"team_code": "roster_team"}), on=["round", "player_id"], how="left")\n',
          '        .join(ro.rename({"team_code": "roster_team"}), on=["round", "player_id"], how="left")\n'
          '               .join(OVR, on=["match_key", "round", "player_id"], how="left")\n')

src = rep(src,
          'pl.coalesce([pl.col("roster_team"), pl.col("team_code")]).cast(pl.Int64).alias("side")',
          'pl.coalesce([pl.col("ov_side"), pl.col("roster_team"), pl.col("team_code")]).cast(pl.Int64).alias("side")')

src = rep(src,
          '    todo = [m for m in mk_list if m in bkeys and m in kid]',
          '    todo = [m for m in mk_list if m in bkeys and m in kid and m in AFFECTED]')

open(DST, "w", encoding="utf-8").write(src)
print("wrote", DST, len(src), "chars")
