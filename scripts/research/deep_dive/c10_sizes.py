# -*- coding: utf-8 -*-
"""
C10 - size per channel for CS2 match folders, E:\\csds_final\\csds\\2026\\10\\01..08.

Metadata only. No file is opened for reading.
  * logical bytes = st_size from os.scandir / DirEntry.stat
  * on-disk bytes = st_size rounded up to whole clusters. The cluster size is read
    from the volume with GetDiskFreeSpaceW (a volume query). GetCompressedFileSizeW
    was tried first and returned the logical size on this exFAT volume, so it is not used.

Channel = each entry directly inside a match folder. The data has one extensionless
file per channel, not one subfolder per channel. A directory entry would also count
as a channel (its files are summed recursively); none were found.
Names containing a dot (for example player_vector.334aDCEe) are stray files. They are
reported separately and excluded from per-channel averages and from the estimates.

Writes the same text to LOG_PATH and prints it. Output is ASCII only.
"""
import ctypes
import os
import statistics
import sys
from collections import Counter
from ctypes import wintypes

ROOT = r"E:\csds_final\csds\2026\10"
VOLUME_ROOT = "E:\\"
DAYS = ["01", "02", "03", "04", "05", "06", "07", "08"]
LOG_PATH = r"G:\pureskill\csds-cs2\logs\deep_dive\c10_sizes.log"
SEP_DAYS = 15  # last 15 days of September

# Per-tick channels: dropped in estimate (b). player_status is kept in (c).
PER_TICK = {
    "tick": "per-tick clock channel (by name)",
    "player_vector": "per-tick player position vector (by name; task rule)",
    "player_status": "per-tick / 1 Hz player status (task rule; kept in estimate c)",
    "grenade_vector": "per-tick grenade position vector (by name and size)",
    "world_item_vector": "per-tick world item position vector (by name and size)",
}
META = {"header", "csds"}  # match-level files, not per-tick
KEY_CHANNELS = ["player_vector", "player_status", "tick", "grenade_vector", "world_item_vector"]
CASES = [
    ("a", "(a) all channels"),
    ("b", "(b) event channels only"),
    ("c", "(c) event + player_status"),
]

CLUSTER = 0  # set in main() from the volume


def cluster_size(root):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    f = k32.GetDiskFreeSpaceW
    f.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    f.restype = wintypes.BOOL
    spc, bps, free, total = wintypes.DWORD(), wintypes.DWORD(), wintypes.DWORD(), wintypes.DWORD()
    ok = f(root, ctypes.byref(spc), ctypes.byref(bps), ctypes.byref(free), ctypes.byref(total))
    if not ok:
        raise OSError(ctypes.get_last_error(), "GetDiskFreeSpaceW failed")
    return spc.value * bps.value


def on_disk(logical):
    if logical <= 0:
        return 0
    return ((logical + CLUSTER - 1) // CLUSTER) * CLUSTER


def measure_entry(entry):
    """Return (logical, on_disk, n_files, n_subdirs) for a file, or for a directory tree."""
    if entry.is_file(follow_symlinks=False):
        lg = entry.stat(follow_symlinks=False).st_size
        return lg, on_disk(lg), 1, 0
    lg_t = dk_t = nf = nd = 0
    for dirpath, dirnames, filenames in os.walk(entry.path):
        nd += len(dirnames)
        for fn in filenames:
            lg = os.stat(os.path.join(dirpath, fn), follow_symlinks=False).st_size
            lg_t += lg
            dk_t += on_disk(lg)
            nf += 1
    return lg_t, dk_t, nf, nd


def gb(n):
    return n / 1e9


def in_case(case, name):
    if case == "a":
        return True
    if case == "b":
        return name not in PER_TICK
    return (name not in PER_TICK) or name == "player_status"


def main():
    global CLUSTER
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    CLUSTER = cluster_size(VOLUME_ROOT)

    ch_matches = Counter()  # channel -> matches containing it
    ch_logical = Counter()  # channel -> total logical bytes
    ch_disk = Counter()     # channel -> total on-disk bytes
    ch_is_dir = Counter()
    day_stats = {d: {"matches": 0, "logical": 0, "disk": 0} for d in DAYS}
    day_case = {d: {k: [0, 0] for k, _ in CASES} for d in DAYS}  # [logical, on_disk] per case
    day_ch = {d: Counter() for d in DAYS}                        # logical bytes per channel per day
    channel_sets = Counter()
    match_ids = Counter()
    strays = []             # (day, match, name, logical, on_disk)
    anomalies = []
    n_matches = 0
    n_empty = 0
    n_subdirs = 0
    n_files = 0

    for d in DAYS:
        day_path = os.path.join(ROOT, d)
        if not os.path.isdir(day_path):
            anomalies.append(f"day folder missing: {d}")
            continue
        with os.scandir(day_path) as it:
            day_entries = list(it)
        for de in day_entries:
            if not de.is_dir(follow_symlinks=False):
                anomalies.append(f"non-folder entry at day level: {d}\\{de.name}")
                continue
            n_matches += 1
            day_stats[d]["matches"] += 1
            match_ids[de.name] += 1
            with os.scandir(de.path) as it:
                entries = list(it)
            if not entries:
                n_empty += 1
            present = set()
            m_lg = m_dk = 0
            for ce in entries:
                if ce.is_symlink():
                    anomalies.append(f"symlink/junction skipped: {d}\\{de.name}\\{ce.name}")
                    continue
                lg, dk, nf, nd = measure_entry(ce)
                n_files += nf
                n_subdirs += nd
                if "." in ce.name:
                    strays.append((d, de.name, ce.name, lg, dk))
                    continue
                present.add(ce.name)
                ch_matches[ce.name] += 1
                ch_logical[ce.name] += lg
                ch_disk[ce.name] += dk
                if not ce.is_file(follow_symlinks=False):
                    ch_is_dir[ce.name] += 1
                m_lg += lg
                m_dk += dk
                day_ch[d][ce.name] += lg
                for k, _ in CASES:
                    if in_case(k, ce.name):
                        day_case[d][k][0] += lg
                        day_case[d][k][1] += dk
            channel_sets[frozenset(present)] += 1
            day_stats[d]["logical"] += m_lg
            day_stats[d]["disk"] += m_dk

    N = n_matches
    if N == 0:
        out("No match folders found. Check ROOT.")
        return

    names = sorted(ch_logical, key=lambda c: -ch_logical[c])
    tot_lg = sum(ch_logical[c] for c in names)
    tot_dk = sum(ch_disk[c] for c in names)
    stray_lg = sum(s[3] for s in strays)
    stray_dk = sum(s[4] for s in strays)

    out("C10 - size per channel, CS2 match folders, October 01-08 2026")
    out(f"ROOT: {ROOT}")
    out("Method: metadata only (os.scandir, DirEntry.stat, GetDiskFreeSpaceW). No file content opened.")
    out(f"Cluster size of {VOLUME_ROOT} (GetDiskFreeSpaceW): {CLUSTER:,} bytes")
    out("Units: bytes. GB = 1e9 bytes. logical = st_size. on-disk = st_size rounded up to whole clusters.")
    out("")

    # Matches per day
    out("== Matches per day folder ==")
    out(f"{'day':<6}{'matches':>9}{'logical GB':>13}{'on-disk GB':>13}{'avg logical/match':>19}{'avg on-disk/match':>19}")
    for d in DAYS:
        s = day_stats[d]
        m = s["matches"]
        avg_l = s["logical"] / m if m else 0
        avg_d = s["disk"] / m if m else 0
        out(f"{d:<6}{m:>9,}{gb(s['logical']):>13,.2f}{gb(s['disk']):>13,.2f}{avg_l:>19,.1f}{avg_d:>19,.1f}")
    avg_per_day = N / len(DAYS)
    out(f"{'TOTAL':<6}{N:>9,}{gb(sum(s['logical'] for s in day_stats.values())):>13,.2f}"
        f"{gb(sum(s['disk'] for s in day_stats.values())):>13,.2f}")
    out(f"Average matches per day ({N:,} / {len(DAYS)} day folders): {avg_per_day:.2f}")
    counts = [day_stats[d]["matches"] for d in DAYS]
    big = max(DAYS, key=lambda d: day_stats[d]["matches"])
    out(f"Largest day: {big} with {day_stats[big]['matches']:,} matches "
        f"({100 * day_stats[big]['matches'] / N:.1f}% of all). Median day: {statistics.median(counts):.1f}.")
    out("")

    # Per-day key channels (logical bytes per match)
    out("== Per-day average logical bytes per match (key channels) ==")
    hdr = f"{'day':<6}" + "".join(f"{c[:14]:>16}" for c in KEY_CHANNELS) + f"{'(b) event only':>18}"
    out(hdr)
    for d in DAYS:
        m = day_stats[d]["matches"]
        if not m:
            continue
        row = f"{d:<6}" + "".join(f"{day_ch[d][c] / m:>16,.0f}" for c in KEY_CHANNELS)
        row += f"{day_case[d]['b'][0] / m:>18,.0f}"
        out(row)
    out("")

    # Structure checks
    out("== Structure checks ==")
    out(f"  Match folders: {N:,}; with no entries: {n_empty:,}")
    out(f"  Channel folders (directories inside a match): {sum(ch_is_dir.values()):,}; "
        f"subfolders inside them: {n_subdirs:,}")
    out(f"  Files counted (all entries, recursive): {n_files:,}")
    dup_ids = [k for k, v in match_ids.items() if v > 1]
    out(f"  Match folder IDs appearing in more than one day: {len(dup_ids):,}")
    out(f"  Anomalies recorded: {len(anomalies):,}")
    for a in anomalies[:20]:
        out(f"    - {a}")
    out(f"  Distinct canonical channel sets: {len(channel_sets)}")
    if channel_sets:
        ref_set, ref_n = channel_sets.most_common(1)[0]
        out(f"  Most common set: {len(ref_set)} channels, in {ref_n:,} of {N:,} matches")
        off = [(s, c) for s, c in channel_sets.items() if s != ref_set]
        if off:
            missing = Counter()
            for s, c in off:
                for x in ref_set - s:
                    missing[x] += c
            out(f"  Matches whose channel set differs from the most common set: {sum(c for _, c in off):,}")
            for x, c in sorted(missing.items()):
                out(f"    missing {x}: {c:,} matches")
        else:
            out("  All matches have the same canonical channel set.")
    out("")

    # Stray files
    out("== Stray entries (name contains a dot; excluded from channel rows and estimates) ==")
    if strays:
        for d, mid, name, lg, dk in strays:
            out(f"  day {d}  match {mid}  file {name}  logical {lg:,}  on-disk {dk:,}")
        out(f"  Stray total: {len(strays)} files, logical {stray_lg:,} bytes, on-disk {stray_dk:,} bytes")
        out("  These look like leftover temp or duplicate files from an interrupted write. Not verified.")
    else:
        out("  none")
    out("")

    # Per-channel table
    def cls(name):
        if name in PER_TICK:
            if name == "player_status":
                return "per-tick/1Hz (drop in b, keep in c)"
            return "per-tick (drop in b and c)"
        if name in META:
            return "match-level meta (keep)"
        return "event (keep)"

    out("== Per channel (canonical names, sorted by logical bytes) ==")
    out(f"{'channel':<20}{'class':<36}{'matches':>8}{'total_logical':>16}{'total_on_disk':>16}"
        f"{'avg_logical/match':>19}{'avg_on_disk/match':>19}")
    for c in names:
        out(
            f"{c:<20}{cls(c):<36}{ch_matches[c]:>8,}{ch_logical[c]:>16,}{ch_disk[c]:>16,}"
            f"{ch_logical[c] / N:>19,.1f}{ch_disk[c] / N:>19,.1f}"
        )
    out(f"{'TOTAL (canonical)':<20}{'':<36}{'':>8}{tot_lg:>16,}{tot_dk:>16,}"
        f"{tot_lg / N:>19,.1f}{tot_dk / N:>19,.1f}")
    out("")

    # Per-match totals by case (all October matches)
    per_case = {}
    for k, _ in CASES:
        lg = sum(ch_logical[c] for c in names if in_case(k, c)) / N
        dk = sum(ch_disk[c] for c in names if in_case(k, c)) / N
        per_case[k] = (lg, dk)

    def pooled(days):
        m = sum(day_stats[d]["matches"] for d in days)
        return {k: (sum(day_case[d][k][0] for d in days) / m,
                    sum(day_case[d][k][1] for d in days) / m) for k, _ in CASES}

    drop_b = [c for c in names if c in PER_TICK and c != "player_status"]
    out("== Per-match totals (canonical channels; strays excluded) ==")
    for k, label in CASES:
        lg, dk = per_case[k]
        out(f"  {label:<28} logical {lg:>14,.1f} bytes/match   on-disk {dk:>14,.1f} bytes/match")
    out(f"  dropped in (b): {', '.join(drop_b)}")
    out(f"  stray files (not in any case): {stray_lg / N:,.1f} logical, {stray_dk / N:,.1f} on-disk bytes/match")
    out("")

    # September estimates
    primary_m = SEP_DAYS * avg_per_day
    early = pooled(["01", "02", "03", "04", "05"])
    late = pooled(["06", "07", "08"])
    scen = [
        (f"PRIMARY: {SEP_DAYS} x mean of 8 days ({avg_per_day:.2f}/day), October per-match sizes",
         primary_m, per_case),
        (f"SENS 1: {SEP_DAYS} x mean excl. largest day ({day_stats[big]['matches']} removed), October per-match sizes",
         SEP_DAYS * (N - day_stats[big]["matches"]) / (len(DAYS) - 1), per_case),
        (f"SENS 2: {SEP_DAYS} x median day ({statistics.median(counts):.1f}/day), October per-match sizes",
         SEP_DAYS * statistics.median(counts), per_case),
        (f"SENS 3: primary match count, per-match sizes of days 01-05 pooled", primary_m, early),
        (f"SENS 4: primary match count, per-match sizes of days 06-08 pooled", primary_m, late),
    ]
    out("== September estimates (last 15 days) ==")
    for title, sep_m, pm in scen:
        out(f"  {title}")
        out(f"    matches = {sep_m:,.0f}")
        out(f"    {'case':<30}{'GB logical':>14}{'GB on-disk':>14}")
        for k, label in CASES:
            lg, dk = pm[k]
            out(f"    {label:<30}{gb(lg * sep_m):>14,.2f}{gb(dk * sep_m):>14,.2f}")
    out("")
    out("Assumption (brief): September per-match, per-channel sizes equal the October 01-08 averages.")
    out("GB = 1e9 bytes. Estimates exclude stray files (about 20 KB logical per match on average).")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"log written: {LOG_PATH}")


if __name__ == "__main__":
    sys.exit(main())
