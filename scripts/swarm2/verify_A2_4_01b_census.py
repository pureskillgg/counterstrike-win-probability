# verify_A2_4 step 01b: independent census of 2024 round labels, split by header build_num.
# Read-only. Inputs: 2024 tome only (G:\pureskill\csds-cs2\tomes\2024-02-g). Nothing is fixed or written except stdout.
# ASSUMPTION A1: header build_num is the build id (the claim uses the same field).
# ASSUMPTION A2: header t_starters_score_final + ct_starters_score_final = total rounds won (no ties) in a complete match.
import os, json
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"


def chan_frame(name, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + name]
    assert len(d) == 1, (name, d)
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect()
             for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]


hdir = os.path.join(TOME, "header")
hm = json.load(open(os.path.join(hdir, "tome"), encoding="utf8"))
H = pl.concat([pl.read_parquet(os.path.join(hdir, p["dataframe"]["dataframe"]))
               .select(["key", "build_num", "match_date", "map_name",
                        "t_starters_score_final", "ct_starters_score_final"])
               for p in hm["pages"]], how="vertical_relaxed")
H = H.with_columns(pl.col("key").str.split("/").list.get(3).alias("day"))
print("== 0. header rows", H.height, "; unique keys", H["key"].n_unique())
print("build_num counts:", H.group_by("build_num").len().sort("build_num").rows())
print("build by day (day, build, n):", H.group_by(["day", "build_num"]).len().sort(["day", "build_num"]).rows())
for b in sorted(H["build_num"].unique().to_list()):
    sub = H.filter(pl.col("build_num") == b)
    print(f"  build {b}: match_date min {sub['match_date'].min()} max {sub['match_date'].max()}")

HB = H.select([pl.col("key").alias("match_key"), "build_num", "day",
               "t_starters_score_final", "ct_starters_score_final"])
st = chan_frame("round_state", ["round", "tick", "event_type", "t_score", "ct_score",
                                "t_score_raw", "ct_score_raw", "phase", "is_warmup", "match_key"])
re_ = chan_frame("round_end", ["round", "tick", "winner_team_code", "match_key"])
rs = chan_frame("round_start", ["round", "tick", "match_key"])
hdr_keys = set(HB["match_key"].to_list())
for nm, df in [("round_state", st), ("round_end", re_), ("round_start", rs)]:
    mk = set(df["match_key"].unique().to_list())
    print(f"  {nm}: rows {df.height}, matches {len(mk)}, matches not in header {len(mk - hdr_keys)}")
st = st.join(HB.select(["match_key", "build_num"]), on="match_key", how="left")
re_ = re_.join(HB.select(["match_key", "build_num"]), on="match_key", how="left")
rs = rs.join(HB.select(["match_key", "build_num"]), on="match_key", how="left")

# 1. per-match label presence by build
fe_rows = st.filter(pl.col("event_type") == "round_freeze_end")
print("\n== 1. freeze_end rows", fe_rows.height, "; duplicate (match,label) groups:",
      fe_rows.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height)


def lab(df, name):
    return df.group_by("match_key").agg(pl.col("round").min().alias(f"{name}_min"),
                                        (pl.col("round") == 1).any().alias(f"{name}_has1"),
                                        pl.col("round").n_unique().alias(f"{name}_nlab"))


M = (HB.join(lab(fe_rows, "fe"), on="match_key", how="left")
       .join(lab(re_, "re"), on="match_key", how="left")
       .join(lab(rs, "rs"), on="match_key", how="left"))
M = M.with_columns([pl.col("fe_has1").fill_null(False), pl.col("re_has1").fill_null(False),
                    pl.col("rs_has1").fill_null(False)])
for b in sorted(M["build_num"].unique().to_list()):
    s = M.filter(pl.col("build_num") == b)
    print(f"  build {b}: n={s.height}; fe_min={s.group_by('fe_min').len().sort('fe_min').rows()}; "
          f"re_min={s.group_by('re_min').len().sort('re_min').rows()}; "
          f"rs_min={s.group_by('rs_min').len().sort('rs_min').rows()}")
    print(f"     label-1 presence (fe_has1, re_has1, rs_has1): "
          f"{s.group_by(['fe_has1', 're_has1', 'rs_has1']).len().sort(['fe_has1', 're_has1', 'rs_has1']).rows()}")
M = M.with_columns(pl.col("day").map_elements(
    lambda d: "01" if d == "01" else ("02_07" if d <= "07" else "12_25"), return_dtype=pl.Utf8).alias("grp"))
print("crosstab build x day-group x fe_has1:",
      M.group_by(["build_num", "grp", "fe_has1"]).len().sort(["build_num", "grp", "fe_has1"]).rows())

# 2. rows labelled round<=1 or null, and round 2 event mix, by build; warmup and phase
print("\n== 2. round_state rows with round<=1 or null, by build/round/event_type/is_warmup")
low = st.filter((pl.col("round") <= 1) | pl.col("round").is_null())
print(low.group_by(["build_num", "round", "event_type", "is_warmup"]).len()
         .sort(["build_num", "round", "event_type", "is_warmup"]).rows())
print("round_state round==2 event mix by build:",
      st.filter(pl.col("round") == 2).group_by(["build_num", "event_type"]).len()
        .sort(["build_num", "event_type"]).rows())
print("is_warmup counts by build:", st.group_by(["build_num", "is_warmup"]).len()
      .sort(["build_num", "is_warmup"]).rows())
print("phase by build and round (rounds 1-2):",
      st.filter(pl.col("round") <= 2).group_by(["build_num", "round", "phase"]).len()
        .sort(["build_num", "round", "phase"]).rows())


# 3. tick ordering per match
def tk(df, name, label):
    return (df.filter(pl.col("round") == label).group_by("match_key")
              .agg(pl.col("tick").min().alias(name)))


T = (HB.select(["match_key", "build_num"])
       .join(tk(rs, "rs1", 1), on="match_key", how="left")
       .join(tk(fe_rows, "fe1", 1), on="match_key", how="left")
       .join(tk(fe_rows, "fe2", 2), on="match_key", how="left")
       .join(tk(re_, "re1", 1), on="match_key", how="left")
       .join(tk(re_, "re2", 2), on="match_key", how="left"))
print("\n== 3. tick order per build (counts of matches where the stated order holds)")
for b in sorted(T["build_num"].unique().to_list()):
    s = T.filter(pl.col("build_num") == b)
    n = s.height
    c = lambda e: s.filter(e).height
    print(f"  build {b} n={n}: rs1<fe2 {c(pl.col('rs1') < pl.col('fe2'))}; fe2<re2 {c(pl.col('fe2') < pl.col('re2'))}; "
          f"rs1<fe1 {c(pl.col('rs1') < pl.col('fe1'))}; fe1<re1 {c(pl.col('fe1') < pl.col('re1'))}; "
          f"re1<fe2 {c(pl.col('re1') < pl.col('fe2'))}; null rs1 {c(pl.col('rs1').is_null())}; "
          f"null fe2 {c(pl.col('fe2').is_null())}; null re1 {c(pl.col('re1').is_null())}")

# 4. score sum test at freeze_end label k: t+ct == k-1 ? (t_score and t_score_raw)
fe1u = fe_rows.unique(["match_key", "round"], keep="first").with_columns([
    ((pl.col("t_score") + pl.col("ct_score")) == (pl.col("round") - 1)).alias("sum_ok"),
    ((pl.col("t_score_raw") + pl.col("ct_score_raw")) == (pl.col("round") - 1)).alias("sum_ok_raw"),
    pl.col("t_score").is_null().alias("tnull"),
])
print("\n== 4. freeze_end sum test (t+ct == k-1)")
for b in sorted(fe1u["build_num"].unique().to_list()):
    s = fe1u.filter(pl.col("build_num") == b)
    k2 = s.filter(pl.col("round") == 2)
    print(f"  build {b}: rows {s.height}; sum_ok(t_score) {int(s['sum_ok'].sum())}; "
          f"sum_ok(raw) {int(s['sum_ok_raw'].sum())}; null t_score {int(s['tnull'].sum())}")
    print(f"     label k=2 rows {k2.height}; sum_ok(t) at k=2 {int(k2['sum_ok'].sum())}; "
          f"(t,ct) pattern at k=2: {k2.group_by(['t_score', 'ct_score']).len().sort(['t_score', 'ct_score']).rows()}")

# 5. header final score total minus number of round_end labels (independent of round_state)
nre = (re_.unique(["match_key", "round"]).group_by("match_key")
          .agg(pl.len().alias("n_re"), pl.col("round").max().alias("re_max")))
X = (HB.join(nre, on="match_key", how="left")
       .with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("hdr_total")))
X = X.with_columns((pl.col("hdr_total") - pl.col("n_re")).alias("hdr_minus_nre"),
                   (pl.col("hdr_total") - pl.col("re_max")).alias("hdr_minus_remax"))
print("\n== 5. header final total minus round_end label count, by build")
for b in sorted(X["build_num"].unique().to_list()):
    s = X.filter(pl.col("build_num") == b)
    print(f"  build {b}: hdr_minus_nre {s.group_by('hdr_minus_nre').len().sort('hdr_minus_nre').rows()}; "
          f"hdr_minus_re_max {s.group_by('hdr_minus_remax').len().sort('hdr_minus_remax').rows()}")
print("DONE census")
