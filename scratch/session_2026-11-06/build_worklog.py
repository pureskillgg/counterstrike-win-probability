"""Build the work-log page from worklog.json (the six day results of the digest Workflow).

Output: scratchpad/worklog.html. Text is escaped; nothing is added to the items beyond layout.
"""
import html
import json
import os
from collections import Counter

SCR = r"C:\Users\<user>\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad"
days = json.load(open(os.path.join(SCR, "worklog.json"), encoding="utf-8"))

DAY_LABEL = {
    "2026-10-30": "Friday 30 October",
    "2026-10-31": "Saturday 31 October",
    "2026-11-01": "Sunday 1 November",
    "2026-11-02": "Monday 2 November",
    "2026-11-03": "Tuesday 3 November",
    "2026-11-04": "Wednesday 4 November (today, to about 23:00 UTC)",
}
STATUSES = ["done", "partial", "failed", "open", "unclear"]
esc = lambda s: html.escape(str(s or ""), quote=True)

total = Counter()
n_items = 0
for d in days:
    for it in d["items"]:
        total[it["status"]] += 1
        n_items += 1

stat_cells = "".join(
    f'<div class="stat"><span class="num">{total[s]}</span><span class="lab s-{s}">{s}</span></div>' for s in STATUSES
)
filter_buttons = '<button type="button" class="chip-btn is-on" data-filter="all">all ' + str(n_items) + "</button>" + "".join(
    f'<button type="button" class="chip-btn" data-filter="{s}"><span class="dot s-{s}"></span>{s} {total[s]}</button>' for s in STATUSES
)

sections = []
for d in days:
    items_html = []
    for it in d["items"]:
        items_html.append(
            f'<li class="item" data-status="{esc(it["status"])}">'
            f'<time class="when">{esc(it["time"])}</time>'
            f'<div class="body">'
            f'<div class="head"><h3>{esc(it["topic"])}</h3><span class="tag s-{esc(it["status"])}">{esc(it["status"])}</span></div>'
            f'<p class="what">{esc(it["what"])}</p>'
            f'<p class="res"><span class="k">Result</span> {esc(it["result"])}</p>'
            f"</div></li>"
        )
    opens = "".join(f"<li>{esc(x)}</li>" for x in d.get("open_threads", [])) or "<li>None listed.</li>"
    issues = "".join(f"<li>{esc(x)}</li>" for x in d.get("issues", [])) or "<li>None listed.</li>"
    sections.append(
        f'<section class="day" id="{esc(d["day"])}">'
        f'<header class="day-head"><h2>{esc(DAY_LABEL.get(d["day"], d["day"]))}</h2>'
        f'<p class="count">{len(d["items"])} items · {len(d.get("open_threads", []))} open threads · {len(d.get("issues", []))} issues recorded</p></header>'
        f'<ol class="items">{"".join(items_html)}</ol>'
        f'<details class="side"><summary>Open threads at end of day</summary><ul>{opens}</ul></details>'
        f'<details class="side"><summary>Issues the transcripts record</summary><ul>{issues}</ul></details>'
        f"</section>"
    )

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="claude:viewport-fit-sides" content="cover">
<title>Work Log, 30 Oct to 4 Nov</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a log read top to bottom. A fixed time column on the left, items stacked by day, status as a small tag. */
:root {
  --bg: #f3f5f2; --panel: #ffffff; --ink: #1b2420; --muted: #56655e; --rule: #d3dbd6;
  --accent: #0c675f; --done: #2c7649; --partial: #8f5b00; --failed: #a1352a; --open: #2456a0; --unclear: #687370;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1613; --panel: #16201c; --ink: #e3eae6; --muted: #98a9a1; --rule: #2a3732;
    --accent: #5bc2b6; --done: #6fc392; --partial: #e5b35e; --failed: #f0917f; --open: #8ebbef; --unclear: #a2aca8;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1613; --panel: #16201c; --ink: #e3eae6; --muted: #98a9a1; --rule: #2a3732;
  --accent: #5bc2b6; --done: #6fc392; --partial: #e5b35e; --failed: #f0917f; --open: #8ebbef; --unclear: #a2aca8;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 400 15px/1.55 "IBM Plex Sans", system-ui, sans-serif;
  padding-block: 24px 56px;
  padding-inline: max(16px, env(safe-area-inset-left, 0px)) max(16px, env(safe-area-inset-right, 0px));
}
main { max-width: 48rem; margin: 0 auto; }
h1, h2, h3 { font-family: "Source Serif 4", Georgia, serif; font-weight: 600; text-wrap: balance; margin: 0; }
h1 { font-size: 2.1rem; line-height: 1.15; letter-spacing: -0.01em; }
.lede { color: var(--muted); margin: 0.6rem 0 0; max-width: 40rem; }
.stats { display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; margin: 1.4rem 0 0; padding: 1rem 0; border-block: 1px solid var(--rule); }
.stat { display: flex; align-items: baseline; gap: 0.4rem; }
.num { font: 500 1.35rem "IBM Plex Mono", monospace; }
.lab { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.filters { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 1.1rem 0 0.4rem; }
.chip-btn {
  font: inherit; font-size: 0.85rem; color: var(--ink); background: var(--panel);
  border: 1px solid var(--rule); border-radius: 999px; padding: 0.3rem 0.75rem; cursor: pointer;
  display: inline-flex; align-items: center; gap: 0.4rem;
}
.chip-btn.is-on { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
.chip-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.dot { width: 0.55rem; height: 0.55rem; border-radius: 50%; background: currentColor; display: inline-block; }
.s-done { color: var(--done); } .s-partial { color: var(--partial); } .s-failed { color: var(--failed); }
.s-open { color: var(--open); } .s-unclear { color: var(--unclear); }
.day { margin-top: 2.6rem; }
.day-head { border-bottom: 2px solid var(--ink); padding-bottom: 0.4rem; }
.day-head h2 { font-size: 1.45rem; }
.count { color: var(--muted); font-size: 0.88rem; margin: 0.3rem 0 0; }
.items { list-style: none; margin: 0; padding: 0; }
.item {
  display: grid; grid-template-columns: 3.6rem minmax(0, 1fr); gap: 0.9rem;
  padding: 0.95rem 0; border-bottom: 1px solid var(--rule);
}
.when { font: 400 0.85rem "IBM Plex Mono", monospace; color: var(--muted); padding-top: 0.15rem; }
.head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.35rem 0.7rem; justify-content: space-between; }
.head h3 { font-size: 1.04rem; }
.tag {
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600;
  color: currentColor; border: 1px solid currentColor; border-radius: 4px; padding: 0.08rem 0.42rem; white-space: nowrap;
}
.what { margin: 0.35rem 0 0; color: var(--ink); }
.res { margin: 0.35rem 0 0; color: var(--muted); font-size: 0.93rem; }
.res .k { font-weight: 600; color: var(--ink); margin-right: 0.25rem; }
.side { margin-top: 0.75rem; font-size: 0.92rem; }
.side summary { cursor: pointer; color: var(--accent); font-weight: 500; }
.side ul { margin: 0.5rem 0 0; padding-left: 1.2rem; color: var(--muted); }
.side li { margin: 0.3rem 0; }
.note { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--rule); color: var(--muted); font-size: 0.88rem; max-width: 44rem; }
.item[hidden], .day[hidden] { display: none !important; }
@media (max-width: 30rem) {
  .item { grid-template-columns: minmax(0, 1fr); gap: 0.2rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
<main>
<h1>Work log, Friday 30 October to Wednesday 4 November</h1>
<p class="lede">Every dated item the session transcripts record for those six days, grouped by day. Two strands run through it: model experiments on the local GPU, and the CS2 data pipeline on SageMaker and RunPod from 1 November on. Times are UTC.</p>
<div class="stats">__STATS__</div>
<div class="filters" role="group" aria-label="Filter by status">__FILTERS__</div>
__SECTIONS__
<p class="note">How this was made: a Haiku agent read a digest of each day's session transcripts and returned the items above. Digests keep every user message (capped per day) and sample the long live conversation. Statuses and numbers are as the transcripts state them; they have not been re-run or re-checked against the outputs. Spot-check any figure before relying on it.</p>
</main>
<script>
(function () {
  var buttons = document.querySelectorAll('.chip-btn');
  var items = document.querySelectorAll('.item');
  buttons.forEach(function (b) {
    b.addEventListener('click', function () {
      var f = b.getAttribute('data-filter');
      buttons.forEach(function (x) { x.classList.toggle('is-on', x === b); });
      items.forEach(function (it) {
        it.hidden = !(f === 'all' || it.getAttribute('data-status') === f);
      });
      document.querySelectorAll('.day').forEach(function (day) {
        var visible = day.querySelectorAll('.item:not([hidden])').length;
        day.hidden = visible === 0;
      });
    });
  });
})();
</script>
</body>
</html>
"""

out = PAGE.replace("__STATS__", stat_cells).replace("__FILTERS__", filter_buttons).replace("__SECTIONS__", "\n".join(sections))
open(os.path.join(SCR, "worklog.html"), "w", encoding="utf-8").write(out)
print("items", n_items, dict(total), "bytes", len(out.encode("utf-8")))
