"""Per-day digest of the live conversation transcript (251 MB): user messages and assistant updates, grouped by UTC day.

Read-only. Output: scratchpad/digests/digest_live_<day>.md for Oct 30 .. Nov 4 2026. Caps keep each file readable.
"""
import json
import os
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
SRC = r"C:\Users\13144\.claude\projects\C--Users-13144-Documents-pureskill\047ee5fa-b35b-4b25-97b1-beaf96834ff2.jsonl"
OUT = r"C:\Users\13144\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad\digests"
DAYS = ["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04"]
SECRET = re.compile(r"(AKIA[0-9A-Z]{12,}|aws_secret\w*\s*[=:]\s*\S+|sk-[A-Za-z0-9_-]{16,}|(?i:api[_-]?key|token|password)\s*[=:]\s*\S+|rpa_[A-Za-z0-9]{20,})")
MAX_USER_PER_DAY = 160
MAX_ASSIST_PER_DAY = 120


def clean(s):
    return SECRET.sub("[masked]", s).replace("\n", " ").strip()


def text_of(content):
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in (content or []) if isinstance(b, dict) and b.get("type") == "text")


users = defaultdict(list)
assist = defaultdict(list)
tools = defaultdict(lambda: defaultdict(int))
with open(SRC, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if '"timestamp"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        ts = d.get("timestamp", "")
        day = ts[:10]
        if day not in DAYS:
            continue
        msg = d.get("message") or {}
        if d.get("type") == "user" and not d.get("isMeta"):
            content = msg.get("content")
            if isinstance(content, str):
                users[day].append((ts, clean(content)[:200]))
            elif isinstance(content, list) and not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                t = clean(text_of(content))
                if t:
                    users[day].append((ts, t[:200]))
        elif d.get("type") == "assistant":
            for b in msg.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    tools[day][b.get("name", "?")] += 1
                elif b.get("type") == "text":
                    t = clean(b.get("text", ""))
                    if len(t) > 80:
                        assist[day].append((ts, t[:150]))

for day in DAYS:
    with open(os.path.join(OUT, f"digest_live_{day}.md"), "w", encoding="utf-8") as fh:
        tl = ", ".join(f"{k}x{v}" for k, v in sorted(tools[day].items(), key=lambda x: -x[1])[:12])
        fh.write(f"# Live conversation, {day} (UTC)\n- tool calls by name: {tl or 'none'}\n- user messages: {len(users[day])} (first {MAX_USER_PER_DAY} shown)\n\n")
        n = len(users[day])
        step = max(1, n // MAX_USER_PER_DAY)
        for ts, u in users[day][::step][:MAX_USER_PER_DAY]:
            fh.write(f"- USER {ts[11:16]}: {u}\n")
        fh.write("\n## assistant updates (sampled)\n")
        m = len(assist[day])
        step = max(1, m // MAX_ASSIST_PER_DAY)
        for ts, a in assist[day][::step][:MAX_ASSIST_PER_DAY]:
            fh.write(f"- asst {ts[11:16]}: {a}\n")
    print(day, "user", len(users[day]), "assist", len(assist[day]), "tools", sum(tools[day].values()), flush=True)
