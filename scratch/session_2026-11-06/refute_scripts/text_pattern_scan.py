import sys, os, re, collections
PATS = {
 "steamid64": r"7656119\d{10}",
 "steam_legacy": r"STEAM_[0-5]:[01]:\d+",
 "email_dotted": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}",
 "email_bracket_any": r"<[^<>\s]+@[^<>\s]+>",
 "ipv4_public_like": r"\b(?!10\.|127\.|0\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b",
 "win_user_path": r"(?i)[A-Z]:\+Users\+[^\\s\"']+",
 "unix_user_path": r"/(?:home|Users)/[^/\s\"']+",
 "token_like": r"(ghp_|github_pat_|AKIA[0-9A-Z]{12,}|sk-[A-Za-z0-9]{20,}|xox[bp]-)",
 "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
 "name_key_field": r"(?i)\"(player_?name|nickname|username|real_?name|display_?name|email|ip|ip_?address|isp|lat|lon)\"\s*:",
}
CR = {k: re.compile(v) for k, v in PATS.items()}
tot = collections.Counter(); files_hit = collections.Counter(); nfiles = 0
per_file = []
for root, _, files in os.walk(sys.argv[1]):
    for fn in files:
        p = os.path.join(root, fn); nfiles += 1
        try:
            s = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        hits = {}
        for k, rx in CR.items():
            n = len(rx.findall(s))
            if n:
                hits[k] = n; tot[k] += n; files_hit[k] += 1
        if hits:
            per_file.append((os.path.relpath(p, sys.argv[1]).replace(os.sep, "/"), hits))
print("files scanned:", nfiles)
print("pattern totals (matches) and files affected:")
for k in PATS:
    print("  %-18s matches=%-6d files=%d" % (k, tot[k], files_hit[k]))
print("files with any hit (path, pattern counts):")
for pth, h in per_file:
    print("  ", pth, h)
print("total files with any hit:", len(per_file))
