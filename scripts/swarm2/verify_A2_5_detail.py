# verify_A2_5_detail.py
# Follow-up for verify_A2_5: (A) round_end-only fingerprint groups that are not header groups,
# (B) per-channel row differences for the near-duplicate pairs, (C) the one duplicated
# (match, round) row in round_end, (D) round-13 bomb-outcome rows (e2_side_check_r13 population)
# that sit in duplicate copies. Read-only on the tome. Writes only logs\swarm2\verify_A2_5.log
# (appends after the main run).
import os, sys, json, glob, collections, hashlib, itertools
import polars as pl

BASE = r"G:\pureskill\csds-cs2"
TOME = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds")
SUF = ".2023-12-28,2024-02-25"
LOG_PATH = os.path.join(BASE, "logs", "swarm2", "verify_A2_5.log")
LOG = open(LOG_PATH, "a", encoding="utf-8")


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def fdir(ch):
    return os.path.join(TOME, "basic_" + ch + SUF)


def read(ch, cols):
    files = sorted(glob.glob(os.path.join(fdir(ch), "dataframe_*")))
    frames = [pl.read_parquet(f, columns=cols) for f in files]
    return pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]


def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def groups(sig):
    g = collections.defaultdict(list)
    for k, s in sig.items():
        g[s].append(k)
    return sorted((sorted(v) for v in g.values() if len(v) > 1), key=lambda x: x[0])


def rows_by_match(df, cols):
    acc = collections.defaultdict(list)
    for row in df.select(["match_key"] + cols).iter_rows():
        acc[row[0]].append("\x1f".join("~" if v is None else str(v) for v in row[1:]))
    for m in acc:
        acc[m].sort()
    return acc


def symdiff(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    return sum((ca - cb).values()) + sum((cb - ca).values())


def short(k):
    return k.split("/")[4][:8]


HCOLS = ["key", "match_id", "match_date", "map_name", "t_starters_score_final",
         "ct_starters_score_final", "number_of_points", "build_num"]
hdr = pl.read_parquet(os.path.join(TOME, "header", "dataframe_00000"), columns=HCOLS)
ALL = hdr["key"].to_list()
MD = {r["key"]: r for r in hdr.iter_rows(named=True)}
DAY = {k: k.split("/")[3] for k in ALL}
sig_full = {k: (MD[k]["match_date"], MD[k]["map_name"], MD[k]["t_starters_score_final"],
                MD[k]["ct_starters_score_final"], MD[k]["number_of_points"]) for k in ALL}
G_hdr = groups(sig_full)
hdr_set = {frozenset(g) for g in G_hdr}
hdr_of = {k: gi for gi, g in enumerate(G_hdr) for k in g}
excess_hdr = {m for g in G_hdr for m in g[1:]}

RE = read("round_end", ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message",
                        "legacy_code", "player_count", "second", "match_key"])
RS_COLS = ["round", "tick", "event_type", "t_score", "ct_score"]
CH = {
    "round_end": (RE, ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message",
                       "legacy_code", "player_count", "second"]),
    "round_start": (read("round_start", ["round", "tick", "time_limit", "match_key"]), ["round", "tick", "time_limit"]),
    "round_state": (read("round_state", RS_COLS + ["match_key"]), RS_COLS),
    "player_death": (read("player_death", ["round", "tick", "player_id", "attacker_id", "weapon_name",
                                           "player_x_pos", "player_y_pos", "player_z_pos", "player_team_code",
                                           "match_key"]),
                     ["round", "tick", "player_id", "attacker_id", "weapon_name", "player_x_pos",
                      "player_y_pos", "player_z_pos", "player_team_code"]),
    "player_info": (read("player_info", ["round", "player_id", "team_code", "rank", "wins", "match_key"]),
                    ["round", "player_id", "team_code", "rank", "wins"]),
    "bomb_state": (read("bomb_state", ["round", "tick", "event_type", "player_id", "site_code", "match_key"]),
                   ["round", "tick", "event_type", "player_id", "site_code"]),
    "player_disconnect": (read("player_disconnect", ["round", "tick", "match_key"]), ["round", "tick"]),
}
ROWS = {ch: rows_by_match(df, cols) for ch, (df, cols) in CH.items()}
dig_re = {m: sha("\x1e".join(v)) for m, v in ROWS["round_end"].items()}
dig_all = {}
for m in ALL:
    parts = []
    for ch in CH:
        parts.append(ch + "\x1d" + "\x1e".join(ROWS[ch].get(m, [])))
    dig_all[m] = sha("\x1f".join(parts))
G_re = groups({m: dig_re[m] for m in ALL})
G_st = groups(dig_all)

# (A) round_end-only groups that are not header groups
out("== A. round_end-only fingerprint groups that are not header-signature groups ==")
extra = [g for g in G_re if frozenset(g) not in hdr_set]
out(f"  round_end-only groups not equal to a header group: {len(extra)}; "
    f"excess copies in them: {sum(len(g) - 1 for g in extra)}")
for g in extra:
    out("  group:")
    for m in g:
        h = MD[m]
        out(f"    {short(m)} folder_day={DAY[m]} match_id={h['match_id']} sig(match_date,map,T,CT,number_of_points)="
            f"{sig_full[m]} header_group={hdr_of.get(m)} round_end_rows={len(ROWS['round_end'][m])} "
            f"build_num={h['build_num']}")
    a = g[0]
    for b in g[1:]:
        diffs = {ch: symdiff(ROWS[ch].get(a, []), ROWS[ch].get(b, [])) for ch in CH}
        out(f"    pair {short(a)} vs {short(b)} row-level symmetric differences by channel: {diffs}")

# (B) near-duplicate pairs (round_end (round,tick) overlap >= 0.5, not in same header group)
out("== B. near-duplicate pairs: per-channel row differences ==")
inv = collections.defaultdict(set)
nre = collections.Counter()
for m, r, t in RE.select(["match_key", "round", "tick"]).iter_rows():
    inv[(int(r), int(t))].add(m)
    nre[m] += 1
pc = collections.Counter()
for v in inv.values():
    if 1 < len(v) <= 300:
        for a, b in itertools.combinations(sorted(v), 2):
            pc[(a, b)] += 1
near = []
for (a, b), c in pc.items():
    if hdr_of.get(a) is not None and hdr_of.get(a) == hdr_of.get(b):
        continue
    mn = min(nre[a], nre[b])
    if mn >= 5 and c / mn >= 0.5:
        near.append((a, b, c, mn))
out(f"  near-duplicate pairs (ratio>=0.5, min rounds>=5, different header group): {len(near)}")
for a, b, c, mn in sorted(near, key=lambda x: -x[2]):
    diffs = {ch: symdiff(ROWS[ch].get(a, []), ROWS[ch].get(b, [])) for ch in CH}
    out(f"    {short(a)} vs {short(b)} shared_round_tick={c} min_rounds={mn} day={DAY[a]}/{DAY[b]} "
        f"sig_a={sig_full[a]} sig_b={sig_full[b]} diffs={diffs}")
    for ch in ["round_end", "round_state", "player_death", "player_info"]:
        la, lb = ROWS[ch].get(a, []), ROWS[ch].get(b, [])
        ex = sorted(set(la) ^ set(lb))[:2]
        if ex:
            out(f"      {ch} sample differing rows: {[e.replace(chr(31), '|')[:140] for e in ex]}")

# (C) the duplicated (match, round) row in round_end
out("== C. duplicated (match, round) rows in round_end ==")
dupkeys = (RE.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1))
out(f"  (match,round) keys with >1 round_end row: {dupkeys.height}")
for m, r, n in dupkeys.iter_rows():
    rows_ = RE.filter((pl.col("match_key") == m) & (pl.col("round") == r)).to_dicts()
    identical = len({tuple(sorted((k, str(v)) for k, v in x.items())) for x in rows_}) == 1
    out(f"  {short(m)} folder_day={DAY[m]} round={r} rows={n} identical_rows={identical} "
        f"in_header_excess={m in excess_hdr} header_group_size={len(G_hdr[hdr_of[m]]) if m in hdr_of and len(G_hdr[hdr_of[m]])>1 else 1}")
    for x in rows_:
        out(f"    tick={x['tick']} winner={x['winner_team_code']} reason={x['win_reason_code']} "
            f"player_count={x['player_count']} second={x['second']}")

# (D) round-13 bomb-outcome population (e2_side_check_r13 definition) and copy overlap
out("== D. round-13 bomb-outcome rows (e2 definition) and duplicate-copy overlap ==")
BS = CH["bomb_state"][0]
BS2 = read("bomb_state", ["match_key", "round", "event_type", "player_team_code"])
outcome = (BS2.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
planter = (BS2.filter(pl.col("event_type") == "bomb_planted")
           .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter_side")))
RE2 = read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
j = (RE2.filter(pl.col("player_count") == 20)
     .join(outcome, on=["match_key", "round"], how="inner")
     .join(planter, on=["match_key", "round"], how="left")
     .with_columns(
         pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch")))
j = j.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("fday"))
j = j.with_columns(pl.col("match_key").map_elements(lambda k: MD[k]["match_date"][:10], return_dtype=pl.String).alias("mdate"))
j = j.with_columns(pl.col("match_key").map_elements(lambda k: k in excess_hdr, return_dtype=pl.Boolean).alias("in_excess"))
r13 = j.filter(pl.col("round") == 13)
out(f"  all rounds with outcome (player_count 20): {j.height}; mismatches: {int(j['mismatch'].sum())}")
for label, sel in [("folder day 12..25", r13.filter(pl.col("fday") >= "12")),
                   ("match_date day 12..25", r13.filter(pl.col("mdate").str.slice(8, 2) >= "12")),
                   ("folder day 01..07", r13.filter(pl.col("fday") <= "07"))]:
    n = sel.height
    mm = int(sel["mismatch"].sum())
    ne = int(sel["in_excess"].sum())
    mme = int(sel.filter(pl.col("in_excess"))["mismatch"].sum())
    ded = sel.filter(~pl.col("in_excess"))
    out(f"  round-13 {label}: rows={n} mismatches={mm}; rows in header excess copies={ne} "
        f"(mismatches among them {mme}); after dropping copies: rows={ded.height} mismatches={int(ded['mismatch'].sum())}")
out(f"  round-13 all days: rows={r13.height} mismatches={int(r13['mismatch'].sum())}; "
    f"in excess copies={int(r13['in_excess'].sum())}")
out("DONE_DETAIL")
LOG.close()
