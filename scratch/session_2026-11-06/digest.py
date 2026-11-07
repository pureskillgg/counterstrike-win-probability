"""Build one digest per day (Oct 30 .. Nov 4 2026) from the session transcripts. Read-only on the transcripts.

Per session: id, start/end (UTC), cwd, number of user turns, tool-call counts, the user's own messages (trimmed),
and short assistant text lines. Secrets-looking strings are masked. Output: scratchpad/digests/digest_<day>.md
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
SRC = "C:/Users/13144/.claude/projects/C--Users-13144-Documents-pureskill"
OUT = r"C:\Users\13144\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad\digests"
os.makedirs(OUT, exist_ok=True)
DAYS = ["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04"]
SECRET = re.compile(r"(AKIA[0-9A-Z]{12,}|aws_secret\w*\s*[=:]\s*\S+|sk-[A-Za-z0-9_-]{16,}|(?i:api[_-]?key|token|password)\s*[=:]\s*\S+|rpa_[A-Za-z0-9]{20,})")
MAX_USER = 220     # chars per user message
MAX_ASSIST = 160   # chars per assistant text line
MAX_ASSIST_LINES = 25  # per session


def clean(s):
    return SECRET.sub("[masked]", s).replace("\n", " ").strip()


def text_of(content):
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
    return " ".join(out)


sessions = defaultdict(lambda: {"start": None, "end": None, "cwd": None, "users": [], "tools": Counter(), "assist": [], "files": set()})
for path in glob.glob(os.path.join(SRC, "*.jsonl")):
    fname = os.path.basename(path)
    if os.path.getsize(path) > 50_000_000:  # the live conversation file is handled by its own pass (too big for a digest)
        continue
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts = d.get("timestamp")
            sid = d.get("sessionId") or fname[:-6]
            if not ts:
                continue
            s = sessions[sid]
            if s["start"] is None or ts < s["start"]:
                s["start"] = ts
            if s["end"] is None or ts > s["end"]:
                s["end"] = ts
            s["cwd"] = s["cwd"] or d.get("cwd")
            msg = d.get("message") or {}
            if d.get("type") == "user" and not d.get("isMeta"):
                content = msg.get("content")
                if isinstance(content, str):
                    s["users"].append((ts, clean(content)[:MAX_USER]))
                elif isinstance(content, list):
                    has_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
                    t = clean(text_of(content))
                    if t and not has_result:
                        s["users"].append((ts, t[:MAX_USER]))
            if d.get("type") == "assistant":
                for b in msg.get("content") or []:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_use":
                        name = b.get("name", "?")
                        s["tools"][name] += 1
                        inp = b.get("input") or {}
                        fp = inp.get("file_path") or inp.get("url") or inp.get("path")
                        if fp and len(s["files"]) < 40:
                            s["files"].add(str(fp)[:160])
                    if b.get("type") == "text" and len(s["assist"]) < 400:
                        t = clean(b.get("text", ""))
                        if len(t) > 60:
                            s["assist"].append((ts, t[:MAX_ASSIST]))

by_day = defaultdict(list)
for sid, s in sessions.items():
    if not s["start"]:
        continue
    day = s["start"][:10]
    if day in DAYS:
        by_day[day].append((sid, s))

for day in DAYS:
    rows = sorted(by_day.get(day, []), key=lambda x: x[1]["start"])
    with open(os.path.join(OUT, f"digest_{day}.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Sessions starting {day} (UTC): {len(rows)}\n\n")
        for sid, s in rows:
            fh.write(f"## {sid[:8]}  {s['start'][11:16]} to {s['end'][11:16]}  cwd={s['cwd']}\n")
            tools = ", ".join(f"{k}x{v}" for k, v in s["tools"].most_common(8))
            fh.write(f"- user turns: {len(s['users'])}; tools: {tools or 'none'}\n")
            if s["files"]:
                fh.write(f"- files touched (sample): {', '.join(sorted(s['files'])[:12])}\n")
            for ts, u in s["users"][:30]:
                fh.write(f"- USER {ts[11:16]}: {u}\n")
            for ts, a in s["assist"][:MAX_ASSIST_LINES]:
                fh.write(f"- asst {ts[11:16]}: {a}\n")
            fh.write("\n")
    print(day, "sessions", len(rows), "bytes", os.path.getsize(os.path.join(OUT, f"digest_{day}.md")), flush=True)
