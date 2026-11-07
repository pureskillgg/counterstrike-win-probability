import glob, os
import polars as pl
pi_files = sorted(glob.glob("E:/csds_final/csds/2026/09/*/*/player_info"))
sample = pi_files[::max(1, len(pi_files) // 40)][:40]
CT, T, TICK = 3, 2, 64
tot = {"5v5": dict(ticks=0, eq=0, status_lt_alive=0, status_gt_alive=0, side_empty_alive=0, rounds=0, rounds_mismatch=0),
       "non5v5": dict(ticks=0, eq=0, status_lt_alive=0, status_gt_alive=0, side_empty_alive=0, rounds=0, rounds_mismatch=0)}
ex = []
for pif in sample:
    d = os.path.dirname(pif)
    pi = pl.read_parquet(pif).filter(pl.col("team_code").is_in([2, 3]))
    ps = pl.read_parquet(os.path.join(d, "player_status"), columns=["tick", "round", "player_id"])
    rs = pl.read_parquet(os.path.join(d, "round_state"), columns=["round", "tick", "event_type"])
    re_ = pl.read_parquet(os.path.join(d, "round_end"), columns=["round", "tick"])
    pdth = pl.read_parquet(os.path.join(d, "player_death"), columns=["round", "tick", "player_id", "player_team_code"])
    frz = dict(rs.filter(pl.col("event_type") == "round_freeze_end").group_by("round").agg(pl.col("tick").min()).iter_rows())
    end = dict(re_.group_by("round").agg(pl.col("tick").min()).iter_rows())
    ros = {}
    for r in pi.select(["round", "team_code", "player_id"]).iter_rows():
        ros.setdefault(r[0], {}).setdefault(r[1], set()).add(r[2])
    side_of = {(r, p): s for r, teams in ros.items() for s, pids in teams.items() for p in pids}
    # status rows -> side via roster, as serving does (rows with no roster side are dropped)
    cnt = {}
    for tick, rnd, pid in ps.iter_rows():
        s = side_of.get((rnd, pid))
        if s is None:
            continue
        cnt[(rnd, tick, s)] = cnt.get((rnd, tick, s), 0) + 1
    dd = {}
    for rnd, tick, pid, side in pdth.iter_rows():
        if side is None:
            continue
        dd.setdefault(rnd, []).append((int(tick), pid, int(side)))
    for rnd, f in frz.items():
        e = end.get(rnd)
        if e is None or e <= f or rnd not in ros:
            continue
        rct = len(ros[rnd].get(CT, set())); rt = len(ros[rnd].get(T, set()))
        cls = "5v5" if (rct == 5 and rt == 5) else "non5v5"
        deaths = sorted(x for x in dd.get(rnd, []) if f <= x[0] <= e)
        seen = set(); dct = dt = 0; di = 0
        mm_round = False
        tot[cls]["rounds"] += 1
        for k in range(0, 10000):
            t = f + TICK * k
            if t >= e:
                break
            while di < len(deaths) and deaths[di][0] <= t:
                _, pid, side = deaths[di]; di += 1
                if pid in seen: continue
                seen.add(pid)
                if side == CT: dct += 1
                elif side == T: dt += 1
            a_ct = max(0, rct - dct); a_t = max(0, rt - dt)
            for alive, s in ((a_ct, CT), (a_t, T)):
                n = cnt.get((rnd, t, s), 0)
                tot[cls]["ticks"] += 1
                if n == alive:
                    tot[cls]["eq"] += 1
                else:
                    mm_round = True
                    if n == 0 and alive > 0:
                        tot[cls]["side_empty_alive"] += 1
                    elif n < alive:
                        tot[cls]["status_lt_alive"] += 1
                    else:
                        tot[cls]["status_gt_alive"] += 1
                    if len(ex) < 8:
                        ex.append((os.path.basename(os.path.dirname(d)), rnd, t, s, alive, n))
        tot[cls]["rounds_mismatch"] += int(mm_round)
for k, v in tot.items():
    print(k, v)
print("sample mismatch examples (match prefix, round, tick, side, serving alive, status rows):", ex)
