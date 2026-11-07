# verify_A2_5_recount.py
# Independent recount of the 2024 header-level duplicate groups claimed in finding A2-5.
# Reads only the 2024 header parquet (projected columns). Output goes to stdout only.
import os, collections
import pyarrow.parquet as pq

BASE = r"G:\pureskill\csds-cs2"
HDR = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds", "header", "dataframe_00000")
COLS = ["key", "match_id", "match_date", "map_name", "t_starters_score_final",
        "ct_starters_score_final", "number_of_points", "build_num", "tick_rate",
        "match_type", "platform", "providence", "rushb_version", "ppp_version",
        "t_starters_avg_rank", "t_starters_avg_wins", "ct_starters_avg_rank", "ct_starters_avg_wins"]
EXTRA = COLS[7:]  # header fields that should be identical between true copies of one demo


def load_rows():
    rows = pq.read_table(HDR, columns=COLS).to_pylist()
    for r in rows:
        parts = r["key"].split("/")
        r["day"] = parts[3]
        r["uuid"] = parts[4]
    return rows


def sig_claim(r):
    # claimed signature: full match_date timestamp | map | final scores | number_of_points
    return "|".join(str(r[c]) for c in ["match_date", "map_name", "t_starters_score_final",
                                        "ct_starters_score_final", "number_of_points"])


def sig_date_only(r):
    # variant used by A2_final_checks.py: date-only match_date
    return "|".join([str(r["match_date"])[:10], str(r["map_name"]), str(r["t_starters_score_final"]),
                     str(r["ct_starters_score_final"]), str(r["number_of_points"])])


def dup_groups(rows, sigf):
    g = collections.defaultdict(list)
    for r in rows:
        g[sigf(r)].append(r)
    return {s: m for s, m in g.items() if len(m) > 1}


def report(name, dg):
    nrows = sum(len(m) for m in dg.values())
    excess = nrows - len(dg)
    pairs = sum(len(m) * (len(m) - 1) // 2 for m in dg.values())
    print(f"[{name}] groups={len(dg)} rows={nrows} excess={excess} within-group pairs={pairs}")
    print(f"[{name}] size distribution (size, count):", sorted(collections.Counter(len(m) for m in dg.values()).items()))
    cross = sum(1 for m in dg.values() if len({x["day"] for x in m}) > 1)
    print(f"[{name}] groups spanning >1 folder day: {cross}; all in one folder day: {len(dg) - cross}")
    return excess


def main():
    rows = load_rows()
    print("header rows:", len(rows))
    print("distinct key:", len({r["key"] for r in rows}), "| distinct match_id:", len({r["match_id"] for r in rows}))
    print("key uuid == match_id for all rows:", all(r["uuid"] == r["match_id"] for r in rows))
    print("match_date string lengths (count):", sorted(collections.Counter(len(str(r["match_date"])) for r in rows).items()))
    print("folder-day row counts:", sorted(collections.Counter(r["day"] for r in rows).items()))
    print()

    dg = dup_groups(rows, sig_claim)
    ex_ts = report("claim sig (timestamp|map|scores|nop)", dg)
    dg_d = dup_groups(rows, sig_date_only)
    ex_d = report("date-only sig (date|map|scores|nop)", dg_d)
    print("excess identical under timestamp vs date-only signature:", ex_ts == ex_d)
    print()

    # Do the members of each claimed group agree on every other header field?
    agree, disagree = 0, []
    for s, m in dg.items():
        diffs = [c for c in EXTRA if len({str(x[c]) for x in m}) > 1]
        if diffs:
            disagree.append((s, [x["uuid"] for x in m], diffs))
        else:
            agree += 1
    print(f"claimed groups where all {len(EXTRA)} extra header fields are identical: {agree}")
    print(f"claimed groups with at least one differing extra header field: {len(disagree)}")
    for d in disagree[:10]:
        print("   ", d)
    print()

    # Discriminating power of the match timestamp alone (how often do distinct matches collide on it?)
    by_ts = collections.defaultdict(list)
    for r in rows:
        by_ts[str(r["match_date"])].append(r)
    shared = {t: m for t, m in by_ts.items() if len(m) > 1}
    pair_n, pair_same, pair_diff = 0, 0, 0
    for t, m in shared.items():
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                pair_n += 1
                a, b = m[i], m[j]
                same = all(str(a[c]) == str(b[c]) for c in ["map_name", "t_starters_score_final",
                                                            "ct_starters_score_final", "number_of_points"])
                if same:
                    pair_same += 1
                else:
                    pair_diff += 1
    print(f"timestamps shared by >1 match: {len(shared)}; pairs sharing a timestamp: {pair_n}")
    print(f"  of those pairs, identical map+scores+number_of_points (claim signature): {pair_same}; differing: {pair_diff}")
    print()

    # Rows in claimed groups by folder day (to see where the copies sit)
    members = collections.Counter(x["day"] for m in dg.values() for x in m)
    print("claimed-group member rows by folder day:", sorted(members.items()))
    print("ASSUMPTION: excess copies = rows minus groups (keep one per group), matching the claim's definition.")
    print("DONE recount")


if __name__ == "__main__":
    main()
