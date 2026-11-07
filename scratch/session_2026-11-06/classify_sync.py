"""Classify the workspace files missing from the repo, and scan them for secrets and player identifiers."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SCR = Path(r"C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad")
WS = Path(r"G:/pureskill/csds-cs2")
REPO = Path(r"C:/Users/13144/Documents/pureskill/counterstrike-win-probability")
rows = json.loads((SCR / "sync_inventory.json").read_text(encoding="utf-8"))
ws = [r for r in rows if r["origin"] == "workspace" and r["status"] in ("missing", "differs", "moved-differs")]

ext = Counter(Path(r["rel"]).suffix.lower() for r in ws)
print("extensions:", dict(ext.most_common()))
sub = Counter("/".join(r["rel"].split("/")[:2]) for r in ws)
print("subfolders (top 30):", dict(sub.most_common(30)))
print("largest 15:")
for r in sorted(ws, key=lambda r: -r["size"])[:15]:
    print(f"  {r['size']:>9} {r['rel']}")

SECRET = re.compile(r"AKIA[0-9A-Z]{16}|aws_secret_access_key\s*[=:]\s*\S{20,}|rpa_[A-Za-z0-9]{20,}|BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|ghp_[A-Za-z0-9]{30,}|xox[bp]-[A-Za-z0-9-]{20,}|sk-[A-Za-z0-9]{32,}|(?i:api[_-]?key)\s*[=:]\s*['\"][A-Za-z0-9_\-]{20,}['\"]")
STEAM = re.compile(r"7656119\d{10}|STEAM_[01]:[01]:\d+|\[U:1:\d+\]")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
sec_hits, steam_hits, email_hits = [], [], Counter()
for r in ws:
    p = WS / r["rel"]
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    if SECRET.search(t):
        sec_hits.append(r["rel"])
    n = len(STEAM.findall(t))
    if n:
        steam_hits.append((r["rel"], n))
    for e in set(EMAIL.findall(t)):
        email_hits[e.split("@")[1].lower()] += 1
print("\nsecret-pattern hits:", len(sec_hits), sec_hits[:20])
print("steam-id-pattern hits:", len(steam_hits), steam_hits[:20])
print("email domains seen (count of files):", dict(email_hits.most_common(10)))

print("\ndiffers / moved-differs:")
for r in ws:
    if r["status"] != "missing":
        print("  ", r["status"], r["rel"], r["repo_matches"])

print("\n== G: models folder sizes ==")
tot = 0
for p in sorted((WS / "models").rglob("*")):
    if p.is_file() and ".meridian-outputs-cache" not in p.parts:
        tot += p.stat().st_size
print(f"total models (excluding cache): {tot/1e6:.1f} MB")
by = Counter()
for p in (WS / "models").rglob("*"):
    if p.is_file() and ".meridian-outputs-cache" not in p.parts:
        by[p.relative_to(WS / "models").parts[0] if len(p.relative_to(WS / "models").parts) > 1 else "(root)"] += p.stat().st_size
print({k: f"{v/1e6:.1f} MB" for k, v in by.most_common()})
print("root model files:")
for p in sorted((WS / "models").glob("*")):
    if p.is_file():
        print(f"  {p.stat().st_size:>9} {p.name}")
