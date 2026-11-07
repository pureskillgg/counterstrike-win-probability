"""Local preview of slide HTML files: each <x-embed> becomes an <iframe srcdoc> at the same position; each <section> is shown as a
1920x1080 box scaled to fit. Usage: python preview_slides.py <slides_dir> <out.html> [slide ids...]"""
import html
import os
import re
import sys

src, out = sys.argv[1], sys.argv[2]
ids = sys.argv[3:]
files = sorted(f for f in os.listdir(src) if f.endswith(".html"))
if ids:
    files = [f for f in files if f[:-5] in ids]
parts = ['<!doctype html><meta charset="utf-8"><style>body{margin:0;background:#888;font-family:Arial}.wrap{width:960px;height:540px;margin:12px auto;position:relative;overflow:hidden;background:#fff}'
         '.inner{width:1920px;height:1080px;transform:scale(.5);transform-origin:0 0;position:absolute;left:0;top:0}section{position:relative;width:1920px;height:1080px;box-sizing:border-box}'
         'table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #DDD9CE;padding:12px 10px;text-align:left}aside{display:none}h1,h2,h3,p,ul,ol{margin:0}ul,ol{padding-left:0}li{margin:6px 0 6px 28px}</style>']
for f in files:
    h = open(os.path.join(src, f), encoding="utf-8").read()

    def repl(m):
        attrs, inner = m.group(1), m.group(2)
        return f'<iframe {attrs} srcdoc="{html.escape(inner, quote=True)}" style="border:0;position:absolute;{re.search(r"style=.([^\"]*)", attrs).group(1) if re.search(r"style=.([^\"]*)", attrs) else ""}"></iframe>'

    h2 = re.sub(r"<x-embed([^>]*)>(.*?)</x-embed>", repl, h, flags=re.S)
    parts.append(f'<div class="wrap" id="{f[:-5]}"><div class="inner">{h2}</div></div>')
open(out, "w", encoding="utf-8").write("\n".join(parts))
print("wrote", out, len(files), "slides")
