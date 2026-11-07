"""verify_D7_5_20_analysis: recompute the D7-5 numbers from the round table written by verify_D7_5_10_main.py.

Reads only the scratchpad parquet + pickle written by stage 10. Prints tables for the verification log.
Labels: W_code (winner_team_code, under test), W_within (no-swap within-round score), W_dp (own min-swap ledger),
W_bomb (bomb outcome, side-absolute). Side codes 2=T, 3=CT.
"""
import sys
import pickle
from pathlib import Path
from collections import Counter
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R_all = pl.read_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet"))
with open(str(SCRATCH / "verify_D7_5_state.pkl"), "rb") as fh:
    ST = pickle.load(fh)
hdr = ST["hdr"]

def agree_tab(df, a, b, label):
    d = df.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())
    g = d.with_columns((pl.col(a) == pl.col(b)).alias("agree")).group_by(["period", "agree"]).len().sort(["period", "agree"])
    tot = {}
    for p, ag, n in g.rows():
        tot.setdefault(p, {True: 0, False: 0})[ag] = n
    out = []
    for p in ["pre", "post"]:
        t = tot.get(p, {True: 0, False: 0})
        n = t[True] + t[False]
        out.append(f"{p}: agree {t[True]} / {n} disagree {t[False]} ({(t[False]/n*100 if n else 0):.2f}%)")
    print(f"{label}: " + " | ".join(out))

for pc_label, df in [("player_count==20", R_all.filter(pl.col("player_count") == 20)), ("all player counts", R_all)]:
    print("\n==================", pc_label, "rows:", df.height, "matches:", df["match_key"].n_unique())
    print("coverage by period (rows with label):")
    print(df.group_by("period").agg(
        pl.len().alias("rounds"),
        pl.col("W_code").is_not_null().sum().alias("W_code"),
        pl.col("W_within").is_not_null().sum().alias("W_within"),
        pl.col("W_dp").is_not_null().sum().alias("W_dp"),
        pl.col("W_bomb").is_not_null().sum().alias("W_bomb"),
        pl.col("orient").is_not_null().sum().alias("orient")).sort("period"))
    agree_tab(df, "W_code", "W_within", "code vs within")
    agree_tab(df, "W_dp", "W_within", "dp   vs within")
    agree_tab(df, "W_code", "W_dp", "code vs dp    ")
    print("-- by rbin (code vs within / dp vs within):")
    for rb in ["r13", "r25", "other"]:
        sub = df.filter(pl.col("rbin") == rb)
        agree_tab(sub, "W_code", "W_within", f"  {rb} code vs within")
        agree_tab(sub, "W_dp", "W_within", f"  {rb} dp vs within  ")

# first-after-swap rounds: orientation changes between r-1 and r (same match, consecutive rounds)
print("\n== first-after-swap rounds (orientation changes between r-1 and r) ==")
D = R_all.filter(pl.col("player_count") == 20).sort(["match_key", "round"]).with_columns(
    pl.col("orient").shift(1).over("match_key").alias("orient_prev"),
    pl.col("round").shift(1).over("match_key").alias("round_prev"),
    pl.col("W_code").shift(1).over("match_key").alias("code_prev"),
    pl.col("W_within").shift(1).over("match_key").alias("within_prev"),
    pl.col("W_dp").shift(1).over("match_key").alias("dp_prev"),
)
D = D.with_columns((pl.col("orient").is_not_null() & pl.col("orient_prev").is_not_null() &
                    (pl.col("orient") != pl.col("orient_prev")) & (pl.col("round_prev") == pl.col("round") - 1)).alias("is_fas"))
fas = D.filter(pl.col("is_fas"))
print("fas rows (player_count==20):", fas.height, "| by round:", fas.group_by("round").len().sort("round").rows())
for p in ["pre", "post"]:
    s = fas.filter(pl.col("period") == p)
    c_ok = s.filter(pl.col("W_code").is_not_null() & pl.col("W_within").is_not_null())
    n_c = c_ok.height
    code_eq_within = c_ok.filter(pl.col("W_code") == pl.col("W_within")).height
    d_ok = s.filter(pl.col("W_dp").is_not_null() & pl.col("W_within").is_not_null())
    dp_eq_within = d_ok.filter(pl.col("W_dp") == pl.col("W_within")).height
    print(f"{p}: fas rows {s.height} | code==within {code_eq_within}/{n_c} (code != within {n_c-code_eq_within}) | "
          f"dp==within {dp_eq_within}/{d_ok.height}")
# non-first-after-swap rounds in post period, for contrast
nf = D.filter(pl.col("period") == "post").filter(~pl.col("round").is_in(fas["round"].unique().to_list()) | True)
print("\n== post-period code != within by fas status ==")
Dp = D.filter((pl.col("period") == "post") & pl.col("W_code").is_not_null() & pl.col("W_within").is_not_null())
print(Dp.group_by(["is_fas", pl.col("W_code") != pl.col("W_within")]).len().sort(["is_fas"]).rows())
print("(columns: is_fas, code_differs_from_within, count)")

# bomb-outcome cross-tab (side-absolute)
print("\n== bomb outcome vs labels (player_count==20, bomb side present) ==")
B = R_all.filter((pl.col("player_count") == 20) & pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0))
for p in ["pre", "post"]:
    s = B.filter(pl.col("period") == p)
    for rb in ["r13", "r25", "other"]:
        t = s.filter(pl.col("rbin") == rb)
        if t.height == 0:
            continue
        nc = t.filter(pl.col("W_code").is_not_null()).height
        cd = t.filter(pl.col("W_code").is_not_null() & (pl.col("W_code") != pl.col("W_bomb"))).height
        ww = t.filter(pl.col("W_within").is_not_null())
        wd = ww.filter(ww["W_within"] != ww["W_bomb"]).height
        dd = t.filter(pl.col("W_dp").is_not_null())
        dn = dd.filter(dd["W_dp"] != dd["W_bomb"]).height
        print(f"{p:4s} {rb:5s} bomb rows {t.height:5d} | code!=bomb {cd:4d}/{nc} | within!=bomb {wd:4d}/{ww.height} | dp!=bomb {dn:4d}/{dd.height}")
print("bomb conflicts (both defused and exploded) in table:", R_all.filter(pl.col("W_bomb") < 0).height)
print("round_state bomb vs bomb_state bomb, where both present (should agree):",
      R_all.filter(pl.col("W_rs_bomb").is_not_null() & pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0)).filter(pl.col("W_rs_bomb") == pl.col("W_bomb")).height,
      "agree of", R_all.filter(pl.col("W_rs_bomb").is_not_null() & pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0)).height)
print("rows with bomb_state outcome but no round_state bomb event:", R_all.filter(pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0) & pl.col("W_rs_bomb").is_null()).height)

# mismatch-type breakdown (bomb side vs code, post period, player_count==20)
print("\n== mismatch types (code != bomb), all periods, player_count==20 ==")
M = B.filter(pl.col("W_code") != pl.col("W_bomb"))
print(M.group_by(["period", "W_bomb", "W_code", "reason_msg"]).len().sort(["period", "W_bomb"]).rows())

# reason message vs code consistency (every row)
print("\n== reason message vs winner_team_code (all rows, player_count==20) ==")
Q = R_all.filter(pl.col("player_count") == 20).with_columns(
    pl.when(pl.col("reason").is_in([1, 9, 17]) | (pl.col("reason_msg").str.contains("Terrorists"))).then(2)
      .when(pl.col("reason").is_in([7, 8, 18]) | (pl.col("reason_msg").str.contains("CTs"))).then(3)
      .otherwise(None).alias("reason_side"))
print("reason-side defined rows:", Q.filter(pl.col("reason_side").is_not_null()).height,
      "| reason_side == W_code:", Q.filter(pl.col("reason_side") == pl.col("W_code")).height)
print(Q.filter(pl.col("reason_side").is_not_null() & (pl.col("reason_side") != pl.col("W_code"))).group_by(["period", "reason_msg", "W_code"]).len().rows())

# per-match header total test: SUPERSEDED. It filtered rounds by player_count, which drops rounds and breaks the
# contiguity check. The header test is redone on the full round sequence in verify_D7_5_40_parity.py and
# verify_D7_5_90_header_subset.py. Stop here.
sys.exit(0)
print("\n== header total test (tF = t_starters_score_final, cF = ct_starters_score_final) ==")
def header_test(df, source, last_from_within=True):
    res = []
    for mk, g in df.sort("round").group_by("match_key", maintain_order=True):
        mk = mk[0] if isinstance(mk, tuple) else mk
        rs = g["round"].to_list()
        N = max(rs)
        if rs != list(range(1, N + 1)):
            res.append((mk, "gap", None, None, None)); continue
        h = hdr[mk]
        orient = g["orient"].to_list()
        if any(o is None for o in orient):
            res.append((mk, "no_orient", None, None, None)); continue
        if source == "dp":
            sides = g["W_dp"].to_list()
            if last_from_within:
                sides[-1] = g["W_within"].to_list()[-1]
        else:
            sides = g[source].to_list()
        if any(s is None for s in sides):
            res.append((mk, "missing", None, None, None)); continue
        A = sum(1 for s, o in zip(sides, orient) if (s == 2 and o == 0) or (s == 3 and o == 1))
        Bt = N - A
        tF, cF = h["t_starters_score_final"], h["ct_starters_score_final"]
        if A == tF and Bt == cF:
            verdict = "pass_tF_cF"
        elif A == cF and Bt == tF:
            verdict = "pass_swapped"
        else:
            verdict = "fail"
        res.append((mk, verdict, A, N, (tF, cF)))
    return res

for label, df in [("player_count==20", R_all.filter(pl.col("player_count") == 20))]:
    per = {}
    pmap = dict(zip(df["match_key"].to_list(), df["period"].to_list()))
    for src in ["W_code", "W_within", "W_dp"]:
        res = header_test(df, src)
        cnt = Counter((pmap.get(mk), v) for mk, v, *_ in res)
        print(f"{label} source={src:9s}:", sorted(cnt.items(), key=lambda x: (str(x[0][0]), x[0][1])))
        per[src] = res
    # matches where the code fails but the within source passes, and the reverse
    code_fail = {mk for mk, v, *_ in per["W_code"] if v == "fail"}
    within_fail = {mk for mk, v, *_ in per["W_within"] if v == "fail"}
    dp_fail = {mk for mk, v, *_ in per["W_dp"] if v == "fail"}
    print("code fail:", len(code_fail), "| within fail:", len(within_fail), "| dp fail:", len(dp_fail))
    print("code fail and within pass:", len(code_fail - within_fail - {mk for mk, v, *_ in per['W_within'] if v not in ('pass_tF_cF','pass_swapped')}))
    print("within fail count by period:", Counter(pmap.get(mk) for mk in within_fail) if within_fail else {})
    print("code fail count by period:", Counter(pmap.get(mk) for mk in code_fail) if code_fail else {})
    print("sample code-fail matches (key, N, A, tF,cF):", [x for x in per["W_code"] if x[1] == "fail"][:8])
    print("sample within-fail matches:", [x for x in per["W_within"] if x[1] == "fail"][:8])
    print("sample dp-fail matches:", [x for x in per["W_dp"] if x[1] == "fail"][:8])
    print("verdict N==tF+cF (rounds equal header total?) sample over all matches:",
          Counter((df.filter(pl.col("match_key") == mk).height == (tF + cF)) for mk, tF, cF in
                  [(k, v["t_starters_score_final"], v["ct_starters_score_final"]) for k, v in list(hdr.items())[:300]]))
