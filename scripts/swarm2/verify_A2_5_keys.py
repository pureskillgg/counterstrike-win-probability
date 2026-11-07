# verify_A2_5_keys.py
# Last check for verify_A2_5: how the duplicate count depends on the key, what cross_period's
# content key still keeps, and the effect of dropping copies on the round-13 defect counts.
# Read-only on the tome. Appends to logs\swarm2\verify_A2_5.log.
import os, sys, json, glob, collections, hashlib
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


def as_set(G):
    return {frozenset(g) for g in G}


def acc_rows(acc, tag, df, cols):
    for row in df.select(["match_key"] + cols).iter_rows():
        acc[row[0]].append(tag + "\x1f" + "\x1f".join("~" if v is None else str(v) for v in row[1:]))


def digest(acc):
    return {m: sha("\x1e".join(sorted(v))) for m, v in acc.items()}


HCOLS = ["key", "match_id", "match_date", "map_name", "t_starters_score_final",
         "ct_starters_score_final", "number_of_points", "build_num"]
hdr = pl.read_parquet(os.path.join(TOME, "header", "dataframe_00000"), columns=HCOLS)
ALL = hdr["key"].to_list()
ALLS = set(ALL)
MD = {r["key"]: r for r in hdr.iter_rows(named=True)}
DAY = {k: k.split("/")[3] for k in ALL}
sig_full = {k: (MD[k]["match_date"], MD[k]["map_name"], MD[k]["t_starters_score_final"],
                MD[k]["ct_starters_score_final"], MD[k]["number_of_points"]) for k in ALL}
G_hdr = groups(sig_full)
X_hdr = sum(len(g) - 1 for g in G_hdr)

RE = read("round_end", ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message",
                        "legacy_code", "player_count", "second", "match_key"])
RST = read("round_start", ["round", "tick", "time_limit", "match_key"])
RS = read("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "match_key"])
BS = read("bomb_state", ["round", "tick", "event_type", "player_id", "site_code", "match_key"])
DC = read("player_disconnect", ["round", "tick", "match_key"])
DE = read("player_death", ["round", "tick", "player_id", "match_key"])
PI = read("player_info", ["round", "player_id", "team_code", "match_key"])

# keys
acc_re = collections.defaultdict(list)
acc_round_noRE = collections.defaultdict(list)   # round_start, round_state, bomb_state, disconnect (no round_end)
acc_cp = collections.defaultdict(list)
acc_re_all = collections.defaultdict(list)
add = acc_rows
add(acc_re, "R", RE, ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message",
                      "legacy_code", "player_count", "second"])
add(acc_round_noRE, "T", RST, ["round", "tick", "time_limit"])
add(acc_round_noRE, "S", RS, ["round", "tick", "event_type", "t_score", "ct_score"])
add(acc_round_noRE, "B", BS, ["round", "tick", "event_type", "player_id", "site_code"])
add(acc_round_noRE, "C", DC, ["round", "tick"])
add(acc_cp, "R", RE, ["round", "tick", "winner_team_code"])
add(acc_cp, "D", DE, ["round", "tick", "player_id"])
add(acc_cp, "P", PI, ["round", "player_id", "team_code"])
dig_re = digest(acc_re)
dig_rnoRE = digest(acc_round_noRE)
dig_cp = digest(acc_cp)

out("== K1. header key vs round_end key vs round-level-without-round_end key vs cross_period content key ==")
G_re = groups({m: dig_re[m] for m in ALL})
G_rn = groups({m: dig_rnoRE[m] for m in ALL})
G_cp = groups({m: dig_cp[m] for m in ALL})
for name, G in [("header sig", G_hdr), ("round_end fp", G_re), ("round-level w/o round_end", G_rn),
                ("cp content sig", G_cp)]:
    out(f"  {name:<28} groups={len(G)} excess={sum(len(g) - 1 for g in G)} distinct_matches={len(ALL) - sum(len(g) - 1 for g in G)}")
out(f"  round_end fp groups == round-level-w/o-round_end groups: {as_set(G_re) == as_set(G_rn)}")
out(f"  round_end fp groups == header groups: {as_set(G_re) == as_set(G_hdr)}")
extra_re = as_set(G_re) - as_set(G_hdr)
out(f"  round_end groups not in header groups: {len(extra_re)}; round-level-w/o-round_end groups not in header groups: "
    f"{len(as_set(G_rn) - as_set(G_hdr))}")
out(f"  header groups not in round-level-w/o-round_end groups: {len(as_set(G_hdr) - as_set(G_rn))}")

# K2. channel-kept set (replica of cross_period channel filter)
out("== K2. channel-kept set and what cross_period's dedupe keeps ==")
dropped = (ALLS - set(BS["match_key"].unique().to_list())) | (ALLS - set(RST["match_key"].unique().to_list()))
kept = ALLS - dropped
out(f"  dropped (bomb_state or round_start zero rows)={len(dropped)}; kept={len(kept)}")
G_re_k = groups({m: dig_re[m] for m in kept})
G_cp_k = groups({m: dig_cp[m] for m in kept})
out(f"  kept: round_end key groups={len(G_re_k)} excess={sum(len(g) - 1 for g in G_re_k)}; "
    f"cp content key groups={len(G_cp_k)} excess={sum(len(g) - 1 for g in G_cp_k)}")
retained = as_set(G_re_k) - as_set(G_cp_k)
cp_excess_k = {m for g in G_cp_k for m in g[1:]}
re_excess_k = {m for g in G_re_k for m in g[1:]}
net_extra = sorted(re_excess_k - cp_excess_k)
nre_all = collections.Counter(RE["match_key"].to_list())
out(f"  round_end-key groups not equal to any cp group on kept set: {len(retained)}")
out(f"  net extra copies the round_end key removes and cp keeps (kept set): {len(net_extra)}; "
    f"their raw round_end rows: {sum(nre_all[m] for m in net_extra)} (round count after survivor filters not computed)")
for m in net_extra:
    twin = [x for g in G_re_k if m in g for x in g if x != m]
    out(f"    extra copy {m.split('/')[4][:8]} day={DAY[m]} pts={sig_full[m][4]} round_end_rows={nre_all[m]} "
        f"identical_round_end_twin={[t.split('/')[4][:8] + '(day ' + DAY[t] + ')' for t in twin]}")
for g in sorted(retained, key=lambda s: min(s))[:10]:
    gl = sorted(g)
    out(f"    group: " + ", ".join(f"{m.split('/')[4][:8]}(day {DAY[m]}, pts {sig_full[m][4]})" for m in gl))
out(f"  cp removed copies vs round_end key on kept: cp={sum(len(g) - 1 for g in G_cp_k)}, "
    f"round_end={sum(len(g) - 1 for g in G_re_k)}")
out(f"  distinct kept matches after cp dedupe = {len(kept) - sum(len(g) - 1 for g in G_cp_k)} (cp key); "
    f"after round_end key = {len(kept) - sum(len(g) - 1 for g in G_re_k)}")
out(f"  distinct kept matches after header key on kept = {len(kept) - sum(len(g) - 1 for g in groups({m: sig_full[m] for m in kept}))}")

# K3. round-13 defect population with copy removal under each key
out("== K3. round-13 bomb-outcome population (e2 definition), copy removal under each key ==")
BS2 = read("bomb_state", ["match_key", "round", "event_type", "player_team_code"])
outcome = (BS2.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
RE2 = read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
j = (RE2.filter(pl.col("player_count") == 20)
     .join(outcome, on=["match_key", "round"], how="inner")
     .with_columns(
         pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"))
     .filter(pl.col("round") == 13))
j = j.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("fday"))
mdate = {k: MD[k]["match_date"][:10] for k in ALL}
rows = j.to_dicts()
ex_hdr = {m for g in G_hdr for m in g[1:]}
ex_re = {m for g in G_re for m in g[1:]}
ex_cp = {m for g in G_cp for m in g[1:]}
for label, sel in [("folder day 12..25", lambda r: r["fday"] >= "12"),
                   ("folder day 01..07", lambda r: r["fday"] <= "07")]:
    R = [r for r in rows if sel(r)]
    out(f"  {label}: raw rows={len(R)} mismatches={sum(r['mismatch'] for r in R)}")
    for kname, EX in [("header key", ex_hdr), ("round_end key", ex_re), ("cp content key", ex_cp)]:
        K = [r for r in R if r["match_key"] not in EX]
        out(f"    after removing {kname} copies: rows={len(K)} mismatches={sum(r['mismatch'] for r in K)} "
            f"rate={(sum(r['mismatch'] for r in K) / max(len(K), 1)):.4f} (raw rate {sum(r['mismatch'] for r in R) / max(len(R), 1):.4f})")
out("DONE_KEYS")
LOG.close()
