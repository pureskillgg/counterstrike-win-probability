"""Second pass for notes/spec/feature_feasibility.md (read-only).

- bomb_state.site_code: codes per match and per map.
- Bomb timers: plant to explode and plant to defuse, in seconds (tick / 64).
- Round length on Target_Saved (code 12) rounds, split by header.is_wingman.
"""
import collections
import glob
import os
import statistics

import pyarrow.parquet as pq

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")


def table_dir(table):
    base = table if table == "header" else "basic_" + table
    for name in os.listdir(TOME):
        if name.split(".")[0] == base:
            return os.path.join(TOME, name)
    raise FileNotFoundError(base)


def column_values(table, col):
    out = []
    for p in sorted(glob.glob(os.path.join(table_dir(table), "dataframe_*"))):
        out.extend(pq.read_table(p, columns=[col]).column(col).to_pylist())
    return out


def summary(label, xs):
    xs = sorted(xs)
    if not xs:
        print(label, "none")
        return
    q = statistics.quantiles(xs, n=20)
    print(f"{label}: n={len(xs)} min={xs[0]:.2f} 5%={q[0]:.2f} median={statistics.median(xs):.2f} "
          f"95%={q[-1]:.2f} max={xs[-1]:.2f}")


def main():
    key_map = dict(zip(column_values("header", "key"), column_values("header", "map_name")))
    wing = dict(zip(column_values("header", "key"), column_values("header", "is_wingman")))

    mk = column_values("bomb_state", "match_key")
    rnd = column_values("bomb_state", "round")
    tick = column_values("bomb_state", "tick")
    ev = column_values("bomb_state", "event_type")
    sc = column_values("bomb_state", "site_code")

    print("== bomb_state.site_code ==")
    codes_by_match = collections.defaultdict(set)
    for a, b in zip(mk, sc):
        if b is not None:
            codes_by_match[a].add(b)
    print("matches with site codes:", len(codes_by_match))
    print("distinct codes per match:", collections.Counter(len(v) for v in codes_by_match.values()).most_common(8))
    per_map = collections.defaultdict(collections.Counter)
    for a, b in zip(mk, sc):
        if b is not None:
            per_map[key_map.get(a)][b] += 1
    for m in sorted(per_map, key=lambda x: str(x)):
        c = per_map[m]
        print(m, "distinct", len(c), "has281", 281 in c, "has282", 282 in c, "top4", c.most_common(4))

    print("\n== bomb timers (seconds from plant) ==")
    events = collections.defaultdict(dict)
    for a, r, t, e in zip(mk, rnd, tick, ev):
        events[(a, r)][e] = t
    boom, defuse = [], []
    for d in events.values():
        if "bomb_planted" in d and "bomb_exploded" in d:
            boom.append((d["bomb_exploded"] - d["bomb_planted"]) / 64)
        if "bomb_planted" in d and "bomb_defused" in d:
            defuse.append((d["bomb_defused"] - d["bomb_planted"]) / 64)
    summary("plant to explode", boom)
    summary("plant to defuse", defuse)

    print("\n== round length on Target_Saved (code 12) rounds, by is_wingman ==")
    live = {}
    for a, r, e, s in zip(column_values("round_state", "match_key"), column_values("round_state", "round"),
                          column_values("round_state", "event_type"), column_values("round_state", "second")):
        if e == "round_freeze_end":
            live[(a, r)] = s
    rl = collections.Counter()
    for a, r, s, c in zip(column_values("round_end", "match_key"), column_values("round_end", "round"),
                          column_values("round_end", "second"), column_values("round_end", "win_reason_code")):
        if c == 12 and (a, r) in live:
            rl[(wing.get(a), round(s - live[(a, r)]))] += 1
    print(rl.most_common(10))


if __name__ == "__main__":
    main()
