"""A11: spot-check six VERIFIED claims in notes\\eda\\EDA_REPORT.md by recomputing from the 2026 tome (read-only).

Claims:
  1. rank_update rows fall after their round's round_end tick and in each match's last round (sections 3.9, 4).
  2. 1,479 bomb_planted rows fall after round_end (section 3.3; flow.md item 1 says 1,479 of 18,306).
  3. round_start tick == 66 sentinel: 678 rows, all round 1, 38.7% of matches (section 3.4).
  4. 10-08 batch share of clean 5v5 rounds: 23,086 of 32,041 (section 1; the file labels this line UNVERIFIED).
  5. weapon_name differs from the modal name for its player_weapon_code in weapon_fire: 2,939 rows (section 3.12).
  6. molotov_state "no player" encoded as -1 in 15,414 rows (section 3.10).

Run:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 G:\\pureskill\\csds-cs2\\scripts\\swarm2\\A11_claims.py

ASSUMPTIONS (marked where used): tick rate 64 ticks/s for second conversions (header tick_rate is printed to check);
live start = first round_freeze_end tick per (match, round); "clean 5v5 round" = round_end.player_count == 10.
Reads are lazy scans with column projection; no channel is held in memory for more than one check at a time.
"""
import glob
import traceback
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
NEW_DAY = "2026-10-08"
TICK = 64  # ASSUMPTION: 64 ticks per second; checked against header tick_rate in main()


def folder(ch):
    return TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")


def read(ch, cols=None):
    files = sorted(glob.glob(str(folder(ch) / "dataframe_*")))
    lf = pl.scan_parquet(files)
    if cols:
        lf = lf.select(cols)
    return lf.collect()


def share(a, b):
    return f"{a:,} of {b:,} ({100.0 * a / b:.3f}%)" if b else f"{a:,} of {b:,}"


def p(*args):
    print(*args, flush=True)


def load_header():
    h = read("header", ["key", "match_date", "is_wingman", "max_rounds", "tick_rate", "match_type", "platform"])
    return h.with_columns(day=pl.col("match_date").str.slice(0, 10))


def load_round_end():
    return read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])


def live_start(rs):
    # ASSUMPTION: live start = earliest round_freeze_end tick per (match, round)
    return (rs.filter(pl.col("event_type") == "round_freeze_end")
              .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("live_tick")))


# ---------------------------------------------------------------- claim 1
def claim1_rank_update(re, hdr):
    p("=" * 72)
    p("CLAIM 1: rank_update rows after round_end and in the match's last round")
    ru = read("rank_update", ["match_key", "round", "tick", "player_id"])
    p(f"rank_update rows: {ru.height:,}; matches with rows: {ru['match_key'].n_unique():,}")
    p(f"round_end duplicate (match, round) pairs: {re.group_by(['match_key', 'round']).len().filter(pl.col('len') > 1).height}")
    end = re.select(["match_key", "round", pl.col("tick").alias("end_tick")])
    j = ru.join(end, on=["match_key", "round"], how="left")
    unmatched = j.filter(pl.col("end_tick").is_null()).height
    m = j.filter(pl.col("end_tick").is_not_null())
    after = m.filter(pl.col("tick") > pl.col("end_tick")).height
    eq = m.filter(pl.col("tick") == pl.col("end_tick")).height
    before = m.filter(pl.col("tick") < pl.col("end_tick")).height
    p(f"rank rows with no round_end for their (match, round): {unmatched}")
    p(f"rank tick vs round_end tick: strictly after {share(after, m.height)}; equal {eq}; before {before}")
    g = m.with_columns(g=(pl.col("tick") - pl.col("end_tick")) / TICK)["g"]
    p(f"seconds after round_end (tick 64/s): min {g.min():.2f}, median {g.median():.2f}, max {g.max():.2f}")

    last = re.group_by("match_key").agg(pl.col("round").max().alias("last_round"),
                                        pl.col("round").n_unique().alias("n_rounds"))
    p(f"matches with round_end rounds not contiguous 1..N: {last.filter(pl.col('last_round') != pl.col('n_rounds')).height} of {last.height}")
    jl = ru.join(last, on="match_key", how="left")
    in_last = jl.filter(pl.col("round") == pl.col("last_round"))
    p(f"rank rows in the match's last round (max round_end round): {share(in_last.height, ru.height)}")
    not_last = jl.filter(pl.col("round") != pl.col("last_round"))
    p(f"rank rows NOT in the last round: {not_last.height}")
    if not_last.height:
        p(not_last.select(["match_key", "round", "last_round"]).head(10))
    per_m = ru.group_by("match_key").agg(pl.col("round").n_unique().alias("nr"))
    p(f"matches with rank rows in more than one round: {per_m.filter(pl.col('nr') > 1).height}")
    p("rank rows by round (round, rows):", ru.group_by("round").len().sort("round").rows())
    lr = in_last.group_by("last_round").len().sort("last_round").rows()
    p("last_round of matches with rank rows (last_round, rows):", lr)
    # is the last round the header's max_rounds (regulation length) or overtime?
    mr = jl.join(hdr.select(["key", "max_rounds", "is_wingman"]), left_on="match_key", right_on="key", how="left")
    p(f"rank rows in last round where last_round == header max_rounds: "
      f"{mr.filter((pl.col('round') == pl.col('last_round')) & (pl.col('last_round') == pl.col('max_rounds'))).height}")
    p(f"rank rows in last round where last_round > header max_rounds (overtime): "
      f"{mr.filter((pl.col('round') == pl.col('last_round')) & (pl.col('last_round') > pl.col('max_rounds'))).height}")
    p(f"rank rows in last round where last_round < header max_rounds (ended early): "
      f"{mr.filter((pl.col('round') == pl.col('last_round')) & (pl.col('last_round') < pl.col('max_rounds'))).height}")
    rank_matches = ru["match_key"].unique()
    p(f"matches with rank rows that have a header row: {hdr.filter(pl.col('key').is_in(rank_matches.implode())).height} of {rank_matches.len()}")


# ---------------------------------------------------------------- claim 2
def claim2_plants(re, rs):
    p("=" * 72)
    p("CLAIM 2: bomb_planted rows after round_end")
    bs = read("bomb_state", ["match_key", "round", "tick", "event_type"])
    p("bomb_state event_type counts:", bs.group_by("event_type").len().sort("len", descending=True).rows())
    pr = bs.filter(pl.col("event_type") == "bomb_planted")
    p(f"bomb_planted rows: {pr.height:,}")
    end = re.select(["match_key", "round", pl.col("tick").alias("end_tick"), "win_reason_code"])
    j = pr.join(end, on=["match_key", "round"], how="left")
    p(f"plant rows with no round_end for their (match, round): {j.filter(pl.col('end_tick').is_null()).height}")
    m = j.filter(pl.col("end_tick").is_not_null())
    post = m.filter(pl.col("tick") > pl.col("end_tick"))
    p(f"plant rows strictly after round_end: {share(post.height, pr.height)}  (report: 1,479 of 18,306, 8.1%)")
    p(f"plant rows equal to round_end tick: {m.filter(pl.col('tick') == pl.col('end_tick')).height}")
    p(f"plant rows before round_end: {m.filter(pl.col('tick') < pl.col('end_tick')).height}")
    p("post-end plant rows by win_reason_code (code, rows):", post.group_by("win_reason_code").len().sort("win_reason_code").rows())
    p("all rounds by win_reason_code (code, rounds):", re.group_by("win_reason_code").len().sort("win_reason_code").rows())
    p(f"distinct rounds with a post-end plant: {post.select(['match_key', 'round']).unique().height}")
    g = post.with_columns(g=(pl.col("tick") - pl.col("end_tick")) / TICK)["g"]
    if g.len():
        p(f"post-end gap seconds: min {g.min():.2f}, median {g.median():.2f}, max {g.max():.2f}")
    per_round = pr.group_by(["match_key", "round"]).len()
    p(f"rounds with >1 bomb_planted row: {per_round.filter(pl.col('len') > 1).height}; max plants in one round: {per_round['len'].max()}")
    # independent cross-check: round_state carries its own round_end and bomb_planted events with ticks
    rs_end = rs.filter(pl.col("event_type") == "round_end").select(
        ["match_key", "round", pl.col("tick").alias("rs_end_tick")]).unique(["match_key", "round"])
    rs_plant = rs.filter(pl.col("event_type") == "bomb_planted").select(
        ["match_key", "round", pl.col("tick").alias("rs_plant_tick")]).unique(["match_key", "round"])
    cmp_end = end.join(rs_end, on=["match_key", "round"], how="full", coalesce=True)
    p(f"round_end tick: round_end channel vs round_state round_end event: both present {cmp_end.filter(pl.col('end_tick').is_not_null() & pl.col('rs_end_tick').is_not_null()).height}; "
      f"equal {cmp_end.filter(pl.col('end_tick') == pl.col('rs_end_tick')).height}; "
      f"only round_end channel {cmp_end.filter(pl.col('rs_end_tick').is_null()).height}; only round_state {cmp_end.filter(pl.col('end_tick').is_null()).height}")
    rs_post = (pr.join(rs_end, on=["match_key", "round"], how="left")
                 .join(rs_plant, on=["match_key", "round"], how="left"))
    p(f"bomb_planted rows (bomb_state) after round_state round_end tick: {rs_post.filter(pl.col('tick') > pl.col('rs_end_tick')).height}; "
      f"bomb_state plant tick equal to round_state plant tick: {rs_post.filter(pl.col('tick') == pl.col('rs_plant_tick')).height} of {rs_post.filter(pl.col('rs_plant_tick').is_not_null()).height} with a round_state plant")
    p(f"round_state bomb_planted rows: {rs.filter(pl.col('event_type') == 'bomb_planted').height}; "
      f"round_state round_end events: {rs.filter(pl.col('event_type') == 'round_end').height}")
    ls = live_start(rs)
    jl = m.join(ls, on=["match_key", "round"], how="left")
    pre_live = jl.filter(pl.col("tick") < pl.col("live_tick")).height
    p(f"plant rows before the live start (round_freeze_end): {pre_live}; rounds without a freeze_end row: "
      f"{jl.filter(pl.col('live_tick').is_null()).height}")
    win = jl.filter((pl.col("tick") >= pl.col("live_tick")) & (pl.col("tick") <= pl.col("end_tick")))
    n_rounds = re.height
    naive_rounds = per_round.height
    p(f"naive plant rate (distinct rounds with any plant / all round_end rounds {n_rounds:,}): "
      f"{naive_rounds:,} = {100.0 * naive_rounds / n_rounds:.2f}% (report: 50.9%)")
    p(f"windowed plant rate (plant in live start..round_end): rows {win.height:,}, "
      f"distinct rounds {win.select(['match_key', 'round']).unique().height:,} = "
      f"{100.0 * win.select(['match_key', 'round']).unique().height / n_rounds:.2f}% (report: 46.8%, 16,827 rounds)")


# ---------------------------------------------------------------- claim 3
def claim3_tick66(re, rs, hdr):
    p("=" * 72)
    p("CLAIM 3: round_start tick == 66 sentinel")
    rst = read("round_start", ["match_key", "round", "tick", "second"])
    p(f"round_start rows: {rst.height:,}; tick value counts with the top 5: "
      f"{rst.group_by('tick').len().sort('len', descending=True).head(5).rows()}")
    s66 = rst.filter(pl.col("tick") == 66)
    p(f"round_start rows with tick 66: {s66.height}; distinct matches: {s66['match_key'].n_unique()}; "
      f"share of header keys ({hdr.height}): {100.0 * s66['match_key'].n_unique() / hdr.height:.2f}% (report: 678, 38.7%)")
    p("round numbers of tick-66 rows (round, rows):", s66.group_by("round").len().sort("round").rows())
    p("seconds value on tick-66 rows (distinct):", s66["second"].unique().sort().to_list()[:5])
    per_m = s66.group_by("match_key").len()
    p(f"tick-66 rows per match (rows per match, matches): "
      f"{per_m.group_by('len').agg(pl.len().alias('n_matches')).sort('len').rows()}")
    dup = rst.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
    p(f"(match, round) with >1 round_start row, all ticks: {dup}")
    # round 1 round_start for non-66 matches: distribution
    r1 = rst.filter(pl.col("round") == 1)
    r1_non = r1.filter(pl.col("tick") != 66)
    p(f"round 1 round_start rows (all): {r1.height}; non-66: {r1_non.height}; "
      f"non-66 tick quantiles p01/p50/p99: {r1_non['tick'].quantile(0.01)}, {r1_non['tick'].quantile(0.5)}, {r1_non['tick'].quantile(0.99)}")
    p(f"round_start rows in non-66 with tick < 200: {rst.filter((pl.col('tick') < 200) & (pl.col('tick') != 66)).height}")
    # is 66 a real time? compare round_start -> live start gap
    ls = live_start(rs)
    j = rst.join(ls, on=["match_key", "round"], how="left").with_columns(
        is66=(pl.col("tick") == 66), gap_s=(pl.col("live_tick") - pl.col("tick")) / TICK)
    for flag in [True, False]:
        sub = j.filter(pl.col("is66") == flag).filter(pl.col("live_tick").is_not_null())
        gg = sub["gap_s"]
        if gg.len():
            p(f"round_start -> live start gap seconds, tick66={flag} (rows {gg.len():,}): "
              f"p01 {gg.quantile(0.01):.2f}, median {gg.median():.2f}, p99 {gg.quantile(0.99):.2f}")
    # date pattern of the tick-66 matches (checks the report's open item)
    d = s66.join(hdr.select(["key", "day"]), left_on="match_key", right_on="key", how="left")
    dist = d.group_by("day").len().sort("day").rows()
    p("tick-66 matches by match_date day (day, matches):", dist)
    hd = hdr.group_by("day").len().sort("day").rows()
    p("all header matches by day (day, matches):", hd)
    n_pre = hdr.filter((pl.col("day") >= "2026-10-01") & (pl.col("day") <= "2026-10-07")).height
    n_66_pre = d.filter((pl.col("day") >= "2026-10-01") & (pl.col("day") <= "2026-10-07")).height
    n_08 = hdr.filter(pl.col("day") == NEW_DAY).height
    n_66_08 = d.filter(pl.col("day") == NEW_DAY).height
    p(f"tick-66 share by batch: 10-01..10-07 {share(n_66_pre, n_pre)}; 10-08 {share(n_66_08, n_08)}")
    # round-1 start tick and start->freeze-end gap by batch, for tick-66 and other round-1 starts
    r1h = (rst.filter(pl.col("round") == 1)
              .join(hdr.select(["key", "day"]), left_on="match_key", right_on="key", how="left")
              .join(ls, on=["match_key", "round"], how="left")
              .with_columns(batch=pl.when(pl.col("day") == NEW_DAY).then(pl.lit("10-08"))
                                    .when((pl.col("day") >= "2026-10-01") & (pl.col("day") <= "2026-10-07")).then(pl.lit("10-01..07"))
                                    .otherwise(pl.lit("outside")),
                            gap_s=(pl.col("live_tick") - pl.col("tick")) / TICK))
    for b in ["10-01..07", "10-08"]:
        for flag in [True, False]:
            sub = r1h.filter((pl.col("batch") == b) & ((pl.col("tick") == 66) == flag))
            if sub.height:
                p(f"round 1 start, batch {b}, tick66={flag}: rows {sub.height}; tick median {sub['tick'].median()}; "
                  f"gap-to-freeze-end median {sub['gap_s'].median():.2f} s (rows with freeze_end {sub['gap_s'].drop_nulls().len()})")
    # overall sanity: rounds over 200 s and how many carry tick 66
    re_s = re.select(["match_key", "round", "tick"]).join(
        rst.select(["match_key", "round", pl.col("tick").alias("start_tick")]), on=["match_key", "round"], how="left")
    dur = re_s.with_columns(dur=(pl.col("tick") - pl.col("start_tick")) / TICK).filter(pl.col("start_tick").is_not_null())
    long = dur.filter(pl.col("dur") > 200)
    p(f"rounds with round_end - round_start > 200 s: {long.height}; of them start tick 66: "
      f"{long.filter(pl.col('start_tick') == 66).height} (report: 249 unexplained, UNVERIFIED)")


# ---------------------------------------------------------------- claim 4
def claim4_batch(re, hdr):
    p("=" * 72)
    p("CLAIM 4: 10-08 batch share of clean 5v5 rounds")
    h = hdr.select(["key", "day", "is_wingman"])
    p(f"header is_wingman nulls: {h.filter(pl.col('is_wingman').is_null()).height}; "
      f"match_date days outside 10-01..10-08: {h.filter((pl.col('day') < '2026-10-01') | (pl.col('day') > NEW_DAY)).height}")
    r = re.select(["match_key", "round", "player_count"]).join(h, left_on="match_key", right_on="key", how="left")
    p(f"round_end rows: {r.height:,}; rows with no header match: {r.filter(pl.col('day').is_null()).height}")
    clean = r.filter(pl.col("player_count") == 10)
    p(f"player_count == 10 rows (clean, definition A): {clean.height:,} (report: 32,041)")
    p(f"  of them in wingman matches: {clean.filter(pl.col('is_wingman')).height}")
    clean_nw = clean.filter(~pl.col("is_wingman"))
    p(f"  non-wingman player_count==10 rows (definition B): {clean_nw.height:,}")

    def bucket(df, label):
        b = df.with_columns(batch=pl.when(pl.col("day") == NEW_DAY).then(pl.lit("2026-10-08"))
                              .when((pl.col("day") >= "2026-10-01") & (pl.col("day") <= "2026-10-07")).then(pl.lit("2026-10-01..07"))
                              .otherwise(pl.lit("outside")))
        tot = b.height
        g = b.group_by("batch").agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches")).sort("batch")
        p(f"{label}: total {tot:,}")
        for row in g.iter_rows(named=True):
            p(f"   {row['batch']}: rounds {row['rounds']:,} ({100.0 * row['rounds'] / tot:.2f}%), matches {row['matches']:,}")
        n08 = b.filter(pl.col("batch") == "2026-10-08").height
        p(f"   10-08 share of rounds: {n08:,} of {tot:,} = {100.0 * n08 / tot:.2f}% (report: 23,086 of 32,041 = 72%)")

    bucket(clean, "definition A: player_count == 10 (all matches)")
    bucket(clean_nw, "definition B: player_count == 10 and not wingman")
    n_hdr_08 = h.filter(pl.col("day") == NEW_DAY).height
    p(f"header matches on 10-08: {n_hdr_08:,} of {h.height:,} = {100.0 * n_hdr_08 / h.height:.2f}% "
      f"(report says 66%; 1,133 on 10-08 is {100.0 * 1133 / 1752:.2f}% of 1,752)")
    # batch label depends on match_date; compare with the day in the key path (csds/YYYY/MM/DD/<id>/csds)
    kd = h.with_columns(kday=pl.col("key").str.extract(r"^csds/(\d{4}/\d{2}/\d{2})/", 1).str.replace_all("/", "-"))
    p(f"sample key: {kd['key'][0]}; parsed key day: {kd['kday'][0]}; keys not parsed: {kd.filter(pl.col('kday').is_null()).height}")
    agree = kd.filter(pl.col("kday") == pl.col("day")).height
    p(f"key-path day equals match_date day: {agree} of {kd.height} (model note says 1,730 of 1,752)")
    p(f"key-path day 10-08 count: {kd.filter(pl.col('kday') == NEW_DAY).height} (match_date 10-08 count {h.filter(pl.col('day') == NEW_DAY).height})")
    cm = clean.group_by("match_key").len().join(h, left_on="match_key", right_on="key", how="left")
    n_cm = cm.height
    n_cm08 = cm.filter(pl.col("day") == NEW_DAY).height
    p(f"matches with >=1 clean round: {n_cm:,}; of them on 10-08: {n_cm08:,} = {100.0 * n_cm08 / n_cm:.2f}% (batch_effect note: 1,099)")


# ---------------------------------------------------------------- claim 5
def claim5_weapon_code():
    p("=" * 72)
    p("CLAIM 5: weapon_fire player_weapon_code vs weapon_name")
    wf = read("weapon_fire", ["match_key", "player_weapon_code", "weapon_name"])
    p(f"weapon_fire rows: {wf.height:,}; null code: {wf.filter(pl.col('player_weapon_code').is_null()).height:,}; "
      f"null name: {wf.filter(pl.col('weapon_name').is_null()).height:,}")
    nn = wf.filter(pl.col("player_weapon_code").is_not_null())
    p(f"non-null code rows: {nn.height:,} (W1 note: 5,468,928)")
    pairs = nn.group_by(["player_weapon_code", "weapon_name"]).len()
    pairs = pairs.sort(["player_weapon_code", "len"], descending=[False, True])
    modal = (pairs.unique(subset=["player_weapon_code"], keep="first", maintain_order=True)
                  .select(["player_weapon_code", pl.col("weapon_name").alias("modal_name"), pl.col("len").alias("modal_rows")]))
    # tie check: codes where a second name has the same count as the modal name
    tot = pairs.group_by("player_weapon_code").agg(pl.col("len").sum().alias("code_rows"))
    ties = (pairs.join(modal, on="player_weapon_code")
                 .filter((pl.col("weapon_name") != pl.col("modal_name")) & (pl.col("len") == pl.col("modal_rows")))
                 .height)
    p(f"codes where a non-modal name ties the modal count: {ties}")
    jj = pairs.join(modal, on="player_weapon_code").join(tot, on="player_weapon_code")
    dis = jj.filter(pl.col("weapon_name") != pl.col("modal_name"))
    p(f"disagreeing rows (name != modal name for its code): {dis['len'].sum():,} of {nn.height:,} = "
      f"{100.0 * dis['len'].sum() / nn.height:.4f}% (report: 2,939, 0.054%)")
    per_code = (dis.group_by(["player_weapon_code", "modal_name", "code_rows", "modal_rows"])
                   .agg(pl.col("len").sum().alias("dis_rows")).sort("dis_rows", descending=True))
    p("top 15 codes by disagreeing rows (code, modal name, code rows, modal rows, disagreeing rows):")
    for row in per_code.head(15).iter_rows(named=True):
        p(f"   {row['player_weapon_code']:>4} {row['modal_name']:<16} {row['code_rows']:>10,} {row['modal_rows']:>10,} {row['dis_rows']:>8,}")
    p("codes the report names (59 knife_t, 45 smokegrenade, 500 bayonet, plus 49):")
    named = per_code.filter(pl.col("player_weapon_code").is_in([59, 45, 500, 49]))
    for row in named.iter_rows(named=True):
        p(f"   {row['player_weapon_code']:>4} {row['modal_name']:<16} code rows {row['code_rows']:,}; modal rows {row['modal_rows']:,}; disagreeing rows {row['dis_rows']:,}")
    p("sum of disagreeing rows for codes 59, 45, 500:",
      per_code.filter(pl.col("player_weapon_code").is_in([59, 45, 500]))["dis_rows"].sum())
    # what names do those disagreeing rows carry, for codes 59, 45, 500
    other = dis.filter(pl.col("player_weapon_code").is_in([59, 45, 500])).select(
        ["player_weapon_code", "modal_name", "weapon_name", "len"]).sort(["player_weapon_code", "len"], descending=[False, True])
    p("disagreeing names for codes 59, 45, 500 (code, modal, name, rows):")
    for row in other.iter_rows(named=True):
        p(f"   {row['player_weapon_code']} {row['modal_name']} -> {row['weapon_name']}: {row['len']}")


# ---------------------------------------------------------------- claim 6
def claim6_molotov():
    p("=" * 72)
    p("CLAIM 6: molotov_state -1 no-player encoding")
    ms = read("molotov_state", ["match_key", "event_type", "player_id", "player_id_fixed", "tick_throw",
                                "extinguisher_id", "burn_duration", "player_x_pos", "player_y_pos",
                                "player_z_pos", "player_x_vel", "player_team_code", "player_weapon_code"])
    p(f"molotov_state rows: {ms.height:,}")
    p("event_type counts:", ms.group_by("event_type").len().sort("len", descending=True).rows())
    cols = ["player_id", "player_id_fixed", "tick_throw", "extinguisher_id"]
    for c in cols:
        p(f"  {c}: == -1 {ms.filter(pl.col(c) == -1).height:,}; null {ms.filter(pl.col(c).is_null()).height:,}; "
          f"min non-negative value {ms.filter(pl.col(c) >= 0)[c].min()}")
    any4 = pl.any_horizontal([pl.col(c) == -1 for c in cols]).fill_null(False)
    all4 = pl.all_horizontal([pl.col(c) == -1 for c in cols]).fill_null(False)
    ms = ms.with_columns(any4=any4, all4=all4)
    p(f"rows with -1 in ANY of {cols}: {ms.filter(pl.col('any4')).height:,} (report: 15,414)")
    p(f"rows with -1 in ALL of {cols}: {ms.filter(pl.col('all4')).height:,}")
    p("-1 rows by event_type (event, rows):", ms.filter(pl.col("any4")).group_by("event_type").len().sort("event_type").rows())
    p("any -1 vs event_type (event, any4, rows):",
      ms.group_by(["event_type", "any4"]).len().sort(["event_type", "any4"]).rows())
    p(f"burn_duration == -1: {ms.filter(pl.col('burn_duration') == -1).height:,}; "
      f"by event: {ms.filter(pl.col('burn_duration') == -1).group_by('event_type').len().rows()}")
    pos_cols = ["player_x_pos", "player_y_pos", "player_z_pos", "player_x_vel", "player_team_code", "player_weapon_code"]
    null_any = pl.any_horizontal([pl.col(c).is_null() for c in pos_cols])
    null_all = pl.all_horizontal([pl.col(c).is_null() for c in pos_cols])
    ms = ms.with_columns(null_all=null_all.fill_null(False), null_any=null_any.fill_null(False))
    n = ms.height
    p(f"player_x_pos null: {ms.filter(pl.col('player_x_pos').is_null()).height:,} = "
      f"{100.0 * ms.filter(pl.col('player_x_pos').is_null()).height / n:.2f}% (report: 118,682, 56.8%)")
    p(f"rows null in ALL of {pos_cols}: {ms.filter(pl.col('null_all')).height:,}; null in ANY: {ms.filter(pl.col('null_any')).height:,}")
    p("overlap: -1 rows that are also all-null on position/team/weapon:",
      ms.filter(pl.col("any4") & pl.col("null_all")).height)
    p("overlap: -1 rows with a non-null player_x_pos:", ms.filter(pl.col("any4") & pl.col("player_x_pos").is_not_null()).height)
    p("event_type x null_all (event, null_all, rows):",
      ms.group_by(["event_type", "null_all"]).len().sort(["event_type", "null_all"]).rows())
    # how player_id itself is encoded per event type: -1, null, or a value >= 0
    ms = ms.with_columns(pid_enc=pl.when(pl.col("player_id").is_null()).then(pl.lit("null"))
                                 .when(pl.col("player_id") == -1).then(pl.lit("minus1"))
                                 .otherwise(pl.lit("value>=0")))
    p("player_id encoding by event_type (event, encoding, rows):",
      ms.group_by(["event_type", "pid_enc"]).len().sort(["event_type", "pid_enc"]).rows())
    p("extinguisher_id encoding on startburn -1 rows (value, rows):",
      ms.filter((pl.col("event_type") == "inferno_startburn") & pl.col("any4"))
        .group_by("extinguisher_id").len().sort("len", descending=True).head(5).rows())
    p(f"-1 rows that are NOT null in player_x_pos: {ms.filter(pl.col('any4') & pl.col('player_x_pos').is_not_null()).height}; "
      f"null-position rows with no -1 in any ID column: "
      f"{ms.filter(pl.col('null_all') & ~pl.col('any4')).height} (these are the 'null-only' rows)")


def main():
    p(f"polars {pl.__version__}")
    hdr = load_header()
    p("header tick_rate values:", hdr.group_by("tick_rate").len().rows(), "(ASSUMPTION used: 64)")
    p(f"header keys: {hdr.height}; unique keys: {hdr['key'].n_unique()}")
    re = load_round_end()
    rs = read("round_state", ["match_key", "round", "tick", "event_type"])
    p("round_state event_type counts:", rs.group_by("event_type").len().sort("len", descending=True).head(12).rows())
    steps = [
        ("claim1", lambda: claim1_rank_update(re, hdr)),
        ("claim2", lambda: claim2_plants(re, rs)),
        ("claim3", lambda: claim3_tick66(re, rs, hdr)),
        ("claim4", lambda: claim4_batch(re, hdr)),
        ("claim5", claim5_weapon_code),
        ("claim6", claim6_molotov),
    ]
    for name, fn in steps:
        try:
            fn()
        except Exception:
            p(f"!! {name} failed")
            p(traceback.format_exc())


if __name__ == "__main__":
    main()
