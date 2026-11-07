"""Inline SVG charts for the deck: a forest plot (point estimate and 95% interval per row) and a bar chart with intervals.

Values are passed in the unit the slide shows (for example 1e-4 log loss), so the axis labels are the numbers a reader sees.
Colours come from the deck palette: ink for text and intervals, rust for a worse result, slate for a better one.
"""

INK = "#1A1A1A"
RUST = "#9A3412"
SLATE = "#3B5B7A"
GREY = "#6A7179"
PAPER = "#FBFBF8"


def _fmt(v, d=2):
    return f"{v:+.{d}f}" if v != 0 else f"{0:.{d}f}"


def forest(rows, x0, x1, ticks, width=1664, row_h=72, label_w=560, value_w=420, axis_title="", zero_label="no change"):
    """rows: list of dicts with label, value, lo, hi (same unit), color (optional).
    Returns an <svg> string. Right of zero is worse when the value is a loss change."""
    top = 36
    bottom = 92
    height = top + row_h * len(rows) + bottom
    left = label_w
    right = width - value_w
    pw = right - left

    def X(v):
        return left + (v - x0) / (x1 - x0) * pw

    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" aria-label="{axis_title}">']
    if x0 < 0 < x1:
        zx = X(0)
        out.append(f'<line x1="{zx:.1f}" y1="{top - 12}" x2="{zx:.1f}" y2="{top + row_h * len(rows)}" stroke="{GREY}" stroke-width="2" stroke-dasharray="8 6"/>')
        out.append(f'<text x="{zx:.1f}" y="{top - 16}" font-size="20" fill="{GREY}" text-anchor="middle">{zero_label}</text>')
    for i, r in enumerate(rows):
        cy = top + row_h * i + row_h / 2
        color = r.get("color", INK)
        out.append(f'<text x="0" y="{cy + 8:.1f}" font-size="26" fill="{INK}">{r["label"]}</text>')
        lo, hi = X(r["lo"]), X(r["hi"])
        out.append(f'<line x1="{lo:.1f}" y1="{cy:.1f}" x2="{hi:.1f}" y2="{cy:.1f}" stroke="{INK}" stroke-width="3"/>')
        for xx in (lo, hi):
            out.append(f'<line x1="{xx:.1f}" y1="{cy - 10:.1f}" x2="{xx:.1f}" y2="{cy + 10:.1f}" stroke="{INK}" stroke-width="3"/>')
        out.append(f'<circle cx="{X(r["value"]):.1f}" cy="{cy:.1f}" r="10" fill="{color}"/>')
        out.append(f'<text x="{width}" y="{cy + 8:.1f}" font-size="24" fill="{INK}" text-anchor="end">'
                   f'{_fmt(r["value"], r.get("d", 2))} [{_fmt(r["lo"], r.get("d", 2))}, {_fmt(r["hi"], r.get("d", 2))}]</text>')
    ay = top + row_h * len(rows) + 14
    out.append(f'<line x1="{left}" y1="{ay}" x2="{right}" y2="{ay}" stroke="{INK}" stroke-width="2"/>')
    for t in ticks:
        tx = X(t)
        out.append(f'<line x1="{tx:.1f}" y1="{ay}" x2="{tx:.1f}" y2="{ay + 10}" stroke="{INK}" stroke-width="2"/>')
        out.append(f'<text x="{tx:.1f}" y="{ay + 40}" font-size="22" fill="{INK}" text-anchor="middle">{_fmt(t, 1) if t else "0"}</text>')
    out.append(f'<text x="{left + pw / 2:.1f}" y="{height - 4}" font-size="22" fill="{GREY}" text-anchor="middle">{axis_title}</text>')
    out.append("</svg>")
    return "".join(out)


def bars(items, vmax, width=1664, bar_h=56, label_w=560, value_w=420, axis_title="", ticks=None, tick_fmt="{:.0f}"):
    """items: list of dicts with label, value, lo, hi (same unit), color. Horizontal bars starting at zero with the interval drawn on top."""
    top = 20
    height = top + bar_h * 1.7 * len(items) + 70
    left = label_w
    right = width - value_w
    pw = right - left

    def X(v):
        return left + v / vmax * pw

    out = [f'<svg viewBox="0 0 {width} {height:.0f}" width="{width}" height="{height:.0f}" xmlns="http://www.w3.org/2000/svg" aria-label="{axis_title}">']
    for i, r in enumerate(items):
        cy = top + bar_h * 1.7 * i + bar_h / 2
        color = r.get("color", INK)
        out.append(f'<text x="0" y="{cy + 9:.1f}" font-size="26" fill="{INK}">{r["label"]}</text>')
        out.append(f'<rect x="{left}" y="{cy - bar_h / 2:.1f}" width="{X(r["value"]) - left:.1f}" height="{bar_h}" fill="{color}" opacity="0.85"/>')
        lo, hi = X(r["lo"]), X(r["hi"])
        ci_y = cy + bar_h / 2 + 12
        out.append(f'<line x1="{lo:.1f}" y1="{ci_y:.1f}" x2="{hi:.1f}" y2="{ci_y:.1f}" stroke="{INK}" stroke-width="3"/>')
        for xx in (lo, hi):
            out.append(f'<line x1="{xx:.1f}" y1="{ci_y - 7:.1f}" x2="{xx:.1f}" y2="{ci_y + 7:.1f}" stroke="{INK}" stroke-width="3"/>')
        out.append(f'<text x="{width}" y="{cy + 9:.1f}" font-size="24" fill="{INK}" text-anchor="end">'
                   f'{r.get("text", _fmt(r["value"], r.get("d", 3)))}</text>')
    ay = top + bar_h * 1.7 * len(items) + 8
    out.append(f'<line x1="{left}" y1="{ay}" x2="{right}" y2="{ay}" stroke="{INK}" stroke-width="2"/>')
    if ticks is None:
        ticks = [vmax / 4 * k for k in range(5)]
    for t in ticks:
        tx = X(t)
        out.append(f'<line x1="{tx:.1f}" y1="{ay}" x2="{tx:.1f}" y2="{ay + 10}" stroke="{INK}" stroke-width="2"/>')
        out.append(f'<text x="{tx:.1f}" y="{ay + 36}" font-size="22" fill="{INK}" text-anchor="middle">{tick_fmt.format(t)}</text>')
    out.append(f'<text x="{left + pw / 2:.1f}" y="{height - 2:.0f}" font-size="22" fill="{GREY}" text-anchor="middle">{axis_title}</text>')
    out.append("</svg>")
    return "".join(out)
