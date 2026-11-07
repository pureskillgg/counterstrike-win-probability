"""HTML helpers for the findings deck (Claude Slides project: one <section> per slide file, <x-embed> for charts).
House style copied from the published slides: 1920x1080, Source Serif 4 headings, IBM Plex Sans body, blue #1D4ED8 accent,
amber #B45309 for cautions, cards #FDFCF9 with a #DDD9CE border. Everything is inline CSS; charts are inline SVG built by a
small script inside <x-embed>.
"""
import html
import json

BG, INK, BLUE, AMBER, MUTED, CARD, RULE = "#F6F5F0", "#16202E", "#1D4ED8", "#B45309", "#6A7179", "#FDFCF9", "#DDD9CE"
SANS = "'IBM Plex Sans', Arial, sans-serif"
SERIF = "'Source Serif 4', Georgia, serif"


def esc(s):
    return html.escape(str(s), quote=False)


def slide(sid, eyebrow, title, body, source=None, notes="", dark=False, h2=None, eyebrow_color=None):
    if h2 is None:
        h2 = 64 if len(title) <= 46 else (56 if len(title) <= 88 else 50)
    bg, fg = ("#16202E", "#F6F5F0") if dark else (BG, INK)
    ec = eyebrow_color or (("#93B4FF") if dark else BLUE)
    parts = [f'<section id="{sid}" data-transition="fade" style="background:{bg};color:{fg};font-family:{SANS};padding:128px 128px 160px;display:flex;flex-direction:column;gap:32px">',
             f'<p style="font-size:24px;letter-spacing:2px;text-transform:uppercase;color:{ec};font-weight:600">{esc(eyebrow)}</p>',
             f'<h2 style="font-family:{SERIF};font-size:{h2}px;font-weight:600;line-height:1.1">{esc(title)}</h2>',
             body]
    if source:
        parts.append(f'<p style="position:absolute;left:128px;bottom:64px;width:1664px;font-size:24px;color:{MUTED if not dark else "#9AA7BC"}">{esc(source)}</p>')
    if notes:
        parts.append(f"<aside>{esc(notes)}</aside>")
    parts.append("</section>\n")
    return "\n".join(parts)


def cards(items, big=72, caption=26):
    """items: (value, caption, color) with color 'blue' | 'amber'."""
    out = ['<div style="display:flex;gap:24px">']
    for v, c, col in items:
        colr = AMBER if col == "amber" else BLUE
        out.append(f'<div style="flex:1;display:flex;flex-direction:column;gap:8px;background:{CARD};padding:26px 28px;border:1px solid {RULE};border-radius:16px">'
                   f'<p style="font-family:{SERIF};font-size:{big}px;font-weight:600;line-height:1.05;color:{colr}">{esc(v)}</p>'
                   f'<p style="font-size:{caption}px;color:#3B4656;line-height:1.3">{esc(c)}</p></div>')
    out.append("</div>")
    return "\n".join(out)


def bullets(items, size=28, color=INK, gap=None):
    lis = "\n".join(f"<li>{esc(i)}</li>" for i in items)
    return f'<ul style="font-size:{size}px;line-height:1.42;color:{color}">\n{lis}\n</ul>'


def table(headers, rows, widths=None, size=26):
    n = len(headers)
    widths = widths or [round(100 / n)] * n
    th = "".join(f'<th style="width:{w}%">{esc(h)}</th>' for h, w in zip(headers, widths))
    trs = "\n".join("<tr>" + "".join(f'<td style="width:{w}%">{esc(c)}</td>' for c, w in zip(r, widths)) + "</tr>" for r in rows)
    return f'<table style="font-size:{size}px;color:{INK}">\n<tr>{th}</tr>\n{trs}\n</table>'


def callout(text, size=28):
    return f'<p style="font-size:{size}px;line-height:1.4;color:{INK};background:#E8EEFB;padding:22px 30px;border-radius:12px">{esc(text)}</p>'


def embed(js_body, left=128, top=300, width=1100, height=580):
    """js_body runs with helpers: E(tag, attrs, text) appends an SVG element; s is the <svg>."""
    return (f'<x-embed style="position:absolute;left:{left}px;top:{top}px;width:{width}px;height:{height}px"><style>body{{margin:0;overflow:hidden}}</style>'
            f'<svg id="s" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" font-family="Arial, Helvetica, sans-serif"></svg><script>\n'
            "var s=document.getElementById('s'),N='http://www.w3.org/2000/svg';\n"
            "function E(t,a,x){var e=document.createElementNS(N,t);for(var k in a)e.setAttribute(k,a[k]);if(x!==undefined)e.textContent=x;s.appendChild(e);return e}\n"
            f"{js_body}\n</script></x-embed>")


def side_panel(inner, left=1260, top=300, width=532):
    return f'<div style="position:absolute;left:{left}px;top:{top}px;width:{width}px;display:flex;flex-direction:column;gap:18px">{inner}</div>'


def stat_box(value, caption, color="blue", big=56, cap=25):
    colr = AMBER if color == "amber" else BLUE
    return (f'<div style="display:flex;flex-direction:column;gap:6px;background:{CARD};padding:22px 28px;border:1px solid {RULE};border-radius:14px">'
            f'<p style="font-family:{SERIF};font-size:{big}px;font-weight:600;color:{colr};line-height:1.05">{esc(value)}</p>'
            f'<p style="font-size:{cap}px;line-height:1.3;color:#3B4656">{esc(caption)}</p></div>')


def note(text, size=24):
    return f'<p style="font-size:{size}px;line-height:1.35;color:{MUTED}">{esc(text)}</p>'


def forest_js(rows, lo, hi, ticks, axis_label, left=370, right=1030, top=20, bottom=470, pitch=88, highlight=None, fmt="toFixed(4)", label_size=26, show_value=True, hline=None):
    """Interval chart. rows: [label, value, ci_lo, ci_hi]; highlight: index drawn in blue (others slate)."""
    data = json.dumps(rows)
    tk = json.dumps(ticks)
    hl = "null" if highlight is None else str(highlight)
    hline_js = "" if hline is None else (f"E('line',{{x1:X({hline}),x2:X({hline}),y1:T,y2:B,stroke:'#64748B','stroke-width':3,'stroke-dasharray':'8 6'}});")
    return (f"var D={data},L={left},R={right},T={top},B={bottom},lo={lo},hi={hi},HL={hl};\n"
            "function X(v){return L+(v-lo)/(hi-lo)*(R-L)}\n"
            f"{tk}.forEach(function(v){{E('line',{{x1:X(v),x2:X(v),y1:T,y2:B,stroke:'#D8DEE8'}});E('text',{{x:X(v),y:B+34,'font-size':24,'text-anchor':'middle',fill:'#475569'}},String(v))}});\n"
            f"{hline_js}\n"
            f"E('text',{{x:(L+R)/2,y:B+78,'font-size':24,'text-anchor':'middle',fill:'#475569'}},{json.dumps(axis_label)});\n"
            f"D.forEach(function(d,i){{var y=T+50+i*{pitch};var c=(HL===null||i==HL)?'#1D4ED8':'#475569';"
            "E('line',{x1:X(d[2]),x2:X(d[3]),y1:y,y2:y,stroke:c,'stroke-width':6});E('line',{x1:X(d[2]),x2:X(d[2]),y1:y-12,y2:y+12,stroke:c,'stroke-width':4});"
            "E('line',{x1:X(d[3]),x2:X(d[3]),y1:y-12,y2:y+12,stroke:c,'stroke-width':4});E('circle',{cx:X(d[1]),cy:y,r:11,fill:c});"
            f"E('text',{{x:L-18,y:y+9,'font-size':{label_size},'text-anchor':'end',fill:'#16202E'}},d[0]);"
            + (f"var tx=(X(d[3])>R-90)?X(d[2])-16:X(d[3])+16;E('text',{{x:tx,y:y+9,'font-size':24,'text-anchor':(X(d[3])>R-90)?'end':'start',fill:c}},d[1].{fmt});" if show_value else "") + "});")


def reliability_js(points, width=780, height=670):
    """points: [pred, observed, lo, hi]."""
    return ("var P=" + json.dumps(points) + ";\nvar L=100,R=750,T=20,B=560;\n"
            "function X(v){return L+v*(R-L)}function Y(v){return B-v*(B-T)}\n"
            "for(var v=0;v<=1.001;v+=0.2){E('line',{x1:X(v),x2:X(v),y1:T,y2:B,stroke:'#D8DEE8'});E('line',{x1:L,x2:R,y1:Y(v),y2:Y(v),stroke:'#D8DEE8'});"
            "E('text',{x:X(v),y:B+34,'font-size':24,'text-anchor':'middle',fill:'#475569'},v.toFixed(1));E('text',{x:L-14,y:Y(v)+8,'font-size':24,'text-anchor':'end',fill:'#475569'},v.toFixed(1))}\n"
            "E('line',{x1:X(0),y1:Y(0),x2:X(1),y2:Y(1),stroke:'#94A3B8','stroke-width':3,'stroke-dasharray':'10 8'});\n"
            "E('text',{x:(L+R)/2,y:B+74,'font-size':24,'text-anchor':'middle',fill:'#475569'},'Predicted probability that CT wins (10 bins)');\n"
            "E('text',{x:34,y:(T+B)/2,'font-size':24,'text-anchor':'middle',fill:'#475569',transform:'rotate(-90 34 '+((T+B)/2)+')'},'Observed CT win rate');\n"
            "var d='';P.forEach(function(p,i){d+=(i?'L':'M')+X(p[0])+' '+Y(p[1])});E('path',{d:d,fill:'none',stroke:'#1D4ED8','stroke-width':3,'stroke-opacity':0.6});\n"
            "P.forEach(function(p){E('line',{x1:X(p[0]),x2:X(p[0]),y1:Y(p[2]),y2:Y(p[3]),stroke:'#1D4ED8','stroke-width':4});E('circle',{cx:X(p[0]),cy:Y(p[1]),r:9,fill:'#2563EB',stroke:'#FFFFFF','stroke-width':2})});\n"
            "E('text',{x:X(0.62),y:Y(0.1),'font-size':24,fill:'#64748B'},'dashed = perfect');")


def bars_js(rows, hi, left=380, right=1030, top=20, pitch=64, color="#1D4ED8", unit="", label_size=24, fmt="toFixed(1)"):
    """Horizontal bars. rows: [label, value]."""
    return ("var D=" + json.dumps(rows) + f",L={left},R={right},T={top},hi={hi};\n"
            "function X(v){return L+v/hi*(R-L)}\n"
            f"D.forEach(function(d,i){{var y=T+i*{pitch};E('rect',{{x:L,y:y,width:X(d[1])-L,height:{pitch - 18},fill:'{color}'}});"
            f"E('text',{{x:L-14,y:y+{(pitch - 18) // 2 + 8},'font-size':{label_size},'text-anchor':'end',fill:'#16202E'}},d[0]);"
            f"E('text',{{x:X(d[1])+12,y:y+{(pitch - 18) // 2 + 8},'font-size':24,fill:'#475569'}},d[1].{fmt}+'{unit}')}});")
