"""Build the 6 Nov 2026 finished-models deck on the existing short deck's master and theme.

Keeps the source master, layouts and theme (Public Sans, same palette). Removes the source slides,
reuses three of its charts (XGBoost explainability), and adds new slides with native tables and shape-drawn bars.
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

SRC = Path(r"C:/Users/<user>/Downloads/CS2 Win-Probability Short Version.pptx")
OUT = Path(r"C:/Users/<user>/Documents/pureskill/CS2 Win-Probability Finished Models 6 Nov 2026.pptx")
SCR = Path(r"C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad")

INK = RGBColor(0x1B, 0x24, 0x30)     # theme dk1
MUTED = RGBColor(0x52, 0x58, 0x60)   # theme dk2
BLUE = RGBColor(0x2F, 0x6F, 0x8F)    # theme accent1
SLATE = RGBColor(0x4A, 0x55, 0x63)   # theme accent3
MIST = RGBColor(0xD6, 0xDE, 0xE6)    # theme accent4
ROW = RGBColor(0xEE, 0xF2, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def extract_images():
    """Pull three XGBoost explainability charts from the source deck (the user's own figures)."""
    want = {2: "shap", 3: "gdt_groups", 7: "calibration"}
    prs = Presentation(SRC)
    paths = {}
    for n, slide in enumerate(prs.slides, 1):
        if n not in want:
            continue
        for sh in slide.shapes:
            if sh.shape_type == 13:  # PICTURE
                p = SCR / f"{want[n]}.{sh.image.ext}"
                p.write_bytes(sh.image.blob)
                paths[want[n]] = p
                break
    return paths


def open_base():
    prs = Presentation(SRC)
    lst = prs.slides._sldIdLst
    for sid in list(lst):
        prs.part.drop_rel(sid.rId)
        lst.remove(sid)
    return prs


def new_slide(prs, title, size=28, top=0.55, h=0.95, notes=None, width=11.9):
    s = prs.slides.add_slide(prs.slide_layouts[1])
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx != 0:
            ph._element.getparent().remove(ph._element)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.89), Inches(top), Inches(width), Inches(h)
    tf = t.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.BOTTOM
    tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = INK
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def text(slide, x, y, w, h, lines, size=16, color=INK, bold=False, space_after=8, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if isinstance(lines, str):
        lines = [lines]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        segs = ln if isinstance(ln, list) else [(ln, bold)]
        for seg_text, seg_bold in segs:
            r = p.add_run()
            r.text = seg_text
            r.font.size = Pt(size)
            r.font.bold = seg_bold
            r.font.color.rgb = color
    return box


def stat(slide, x, y, w, big, label, big_size=36, label_size=15):
    text(slide, x, y, w, 0.7, big, size=big_size, color=BLUE, bold=True, space_after=0)
    text(slide, x, y + 0.78, w, 0.9, label, size=label_size, color=INK, space_after=0)


def source(slide, msg):
    text(slide, 0.89, 6.86, 11.9, 0.3, msg, size=11, color=MUTED, space_after=0)


def card(slide, x, y, w, h):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = WHITE
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def table(slide, x, y, col_w, rows, aligns=None, size=14, row_h=0.42):
    nrows, ncols = len(rows), len(rows[0])
    gs = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * nrows))
    tbl = gs.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE if i == 0 else (WHITE if i % 2 else ROW)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (aligns[j] if aligns else PP_ALIGN.LEFT) if i else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = val
            r.font.size = Pt(size)
            r.font.bold = i == 0
            r.font.color.rgb = WHITE if i == 0 else INK
    return gs


def pic(slide, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def manual_bars(slide, x0, y0, w, h, cats, series, colors, vmin, vmax, ticks):
    """Grouped bars drawn as shapes: renders the same in every viewer, including signs on negative values."""
    def ypos(v):
        return y0 + (vmax - v) / (vmax - vmin) * h

    for t in ticks:
        yy = ypos(t)
        ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(yy), Inches(x0 + w), Inches(yy))
        ln.line.color.rgb = MUTED if t == 0 else MIST
        ln.line.width = Pt(1.25 if t == 0 else 0.75)
        label = "0" if t == 0 else f"{t:+d}"
        text(slide, x0 - 0.7, yy - 0.11, 0.6, 0.25, label, size=11, color=MUTED, align=PP_ALIGN.RIGHT, space_after=0)

    n = len(cats)
    gw = w / n
    bw = 0.42
    for i, cat in enumerate(cats):
        xc = x0 + (i + 0.5) * gw
        for k, (name, vals) in enumerate(series):
            v = vals[i]
            bx = xc - bw - 0.03 if k == 0 else xc + 0.03
            top, bottom = ypos(max(v, 0.0)), ypos(min(v, 0.0))
            rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(bx), Inches(top), Inches(bw), Inches(bottom - top))
            rect.fill.solid()
            rect.fill.fore_color.rgb = colors[k]
            rect.line.fill.background()
            rect.shadow.inherit = False
            label_y = top - 0.27 if v >= 0 else bottom + 0.03
            text(slide, bx - 0.25, label_y, bw + 0.5, 0.25, f"{v:+.2f}", size=12, color=INK,
                 align=PP_ALIGN.CENTER, space_after=0)
        text(slide, x0 + i * gw, y0 + h + 0.12, gw, 0.3, cat, size=12, color=INK, align=PP_ALIGN.CENTER, space_after=0)

    lx = x0
    for k, (name, _) in enumerate(series):
        sq = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(lx), Inches(y0 + h + 0.6), Inches(0.18), Inches(0.18))
        sq.fill.solid()
        sq.fill.fore_color.rgb = colors[k]
        sq.line.fill.background()
        sq.shadow.inherit = False
        text(slide, lx + 0.28, y0 + h + 0.57, 3.3, 0.25, name, size=12, color=INK, space_after=0)
        lx += 3.6


def build():
    imgs = extract_images()
    prs = open_base()

    # 1. Title
    s = new_slide(prs, "Finished models: what we ran and what it shows", size=36, top=1.6, h=1.9, width=7.2,
                  notes="Covers the last two days: sequence-model training on the GDT objective and on log loss, "
                        "the regularisation test, the XGBoost comparison, explainability of the XGBoost model, and the status of weights and data.")
    text(s, 0.89, 3.75, 7.2, 0.8, "Training, GDT scoring and explainability, as of 6 November 2026", size=18, color=INK, space_after=0)
    text(s, 0.89, 4.7, 7.2, 0.8,
         "Builds on the CS2 win-probability short deck (XGBoost explainability). Validation results unless marked.",
         size=14, color=MUTED, space_after=0)
    card(s, 8.6, 1.7, 3.9, 5.1)
    stat(s, 8.95, 1.95, 3.3, "47", "GDT-objective arm-seeds, weights saved")
    stat(s, 8.95, 3.5, 3.3, "0 of 6", "sequence arms pass the pinned GDT bar")
    stat(s, 8.95, 5.05, 3.3, "+0.0011", "log-loss gain of the GRU at 25k steps, 5 of 5 seeds", label_size=14)

    # 2. Summary
    s = new_slide(prs, "No arm passes the GDT bar; 25k steps help the GRU",
                  notes="The 0 of 6 is the pinned GDT bar: the whole 95% interval above zero and a positive situation-weighted estimate. "
                        "MinGRU and xLSTM sigmoid-gate have positive point estimates, but their intervals include zero. "
                        "The +0.0011 is the log-loss GRU at 25k steps against the 30k reference, a stopping-point effect on the GRU only.")
    stat(s, 0.89, 1.9, 3.7, "0 of 6", "sequence arms pass the GDT bar against the GRU control")
    stat(s, 4.95, 1.9, 3.7, "+0.0011", "log-loss gain of the GRU when it stops at 25k steps (5 of 5 seeds)")
    stat(s, 9.0, 1.9, 3.7, "0.45025", "XGBoost 83-input validation log loss, in the same range as the GRU")
    text(s, 0.89, 4.15, 11.9, 2.5, [
        [("Weights. ", True), ("The 47 GDT-objective arm-seeds have weights saved in S3 and locally. The log-loss runs saved none, so their weights cannot be exported without retraining.", False)],
        [("Explainability. ", True), ("Done for the XGBoost model in the explainability report: SHAP, group ablation, calibration. As far as the records show, attribution has not been run for the sequence models.", False)],
        [("Holdout. ", True), ("Read once, for XGBoost. No candidate passes the pinned rules, and no choice used the holdout.", False)],
    ], size=17, space_after=14)
    source(s, "Sources: REPORT_2026-11-06 sections 1, 3.2 to 3.6; regularisation result a3ceeda7 (Meridian); pins 0d97f8f6, 56316f69.")

    # 3. What was run
    s = new_slide(prs, "Two days of runs: what was done",
                  notes="All pods are closed (REPORT section 5). The GDT scoring is a verbatim copy of the intern's grader, so it is not a re-implementation. "
                        "The capacity transformer ran without the 500-step warm-up its pin specified; the result is reported as run.")
    cards = [
        ("47", "GDT-objective arm-seeds, each retrained once with weights saved"),
        ("5", "seeds for the round-2 arms, one for the rest; each paired with the GRU on the same seed"),
        ("10", "regularisation runs: A (30k, dropout 0.2, weight decay 1e-4); B (25k, none)"),
        ("43 of 47", "retrained runs match their original validation loss within 0.0001"),
    ]
    for i, (big, lab) in enumerate(cards):
        x = 0.89 + i * 3.0
        card(s, x, 1.9, 2.75, 2.25)
        stat(s, x + 0.25, 2.05, 2.3, big, lab, big_size=30, label_size=14)
    text(s, 0.89, 4.5, 11.9, 2.3, [
        [("GDT scoring: ", True), ("a verbatim copy of the intern's grader (score_gdt.py), not re-derived.", False)],
        [("Reproduction: ", True), ("the four TCN seeds differ from their originals by up to 0.0006, likely GPU nondeterminism (not yet confirmed).", False)],
        [("Deviation: ", True), ("the capacity transformer ran without the 500-step warm-up its pin specified.", False)],
        [("Compute: ", True), ("RunPod RTX 4090 pods on network volume z7yvc40ykx. All pods are closed.", False)],
    ], size=15, space_after=10)
    source(s, "Sources: REPORT_2026-11-06 sections 4 and 5; weights manifest (47 rows); regularisation metrics per seed on the network volume (results/reg_A, reg_B).")

    # 4. Log loss table
    s = new_slide(prs, "On log loss, the five-seed arms span about 0.001",
                  notes="Five seeds each, final-step scoring on validation. Seed-to-seed spread is about 0.0005, so gaps smaller than that are within noise. "
                        "The GRU control is the reference in both objectives.")
    rows = [
        ["Arm", "Log loss, log-loss objective", "Log loss, GDT objective", "Note"],
        ["xLSTM 7B-style block", "0.449940", "0.450746", "best on log loss"],
        ["xLSTM closed-form", "0.450123", "0.450441", ""],
        ["MinGRU", "0.450138", "0.450274", "best on GDT objective"],
        ["RWKV-7", "0.450168", "0.450730", ""],
        ["xLSTM sigmoid-gate", "0.450337", "0.450474", ""],
        ["TCN", "0.450735", "0.450480", ""],
        ["GRU control", "0.451061", "0.450392", "reference"],
    ]
    R, L = PP_ALIGN.RIGHT, PP_ALIGN.LEFT
    table(s, 0.89, 1.95, [3.6, 2.9, 2.9, 2.5], rows, aligns=[L, R, R, L], size=15, row_h=0.46)
    text(s, 0.89, 5.95, 11.9, 0.8,
         "On log loss the GRU control is the worst of the seven five-seed arms. On GDT-trained log loss it is second of seven. Gaps below about 0.0005 are within seed noise.",
         size=16, color=INK, space_after=0)
    source(s, "Validation, final-step, 1,515,228 seconds, 1,168 matches. Five seeds per arm. Source: REPORT_2026-11-06 sections 3.1 and 3.2.")

    # 5. GDT table
    s = new_slide(prs, "On GDT, no sequence arm passes the bar",
                  notes="The bar needs the whole 95% interval above zero and a positive situation-weighted estimate. None of the six arms clears it. "
                        "The earlier pick of xLSTM sigmoid-gate came from scoring the log-loss-trained models on GDT; the GDT-trained models, scored the same way, do not reproduce it (+0.00019).")
    rows = [
        ["Arm (vs GRU, same seed)", "GDT difference", "95% interval", "Situation-weighted", "Bar met"],
        ["MinGRU", "+0.00039", "-0.00014 to +0.00093", "+0.00012", "no"],
        ["xLSTM sigmoid-gate", "+0.00019", "-0.00054 to +0.00091", "+0.00024", "no"],
        ["xLSTM closed-form", "-0.00006", "-0.00070 to +0.00067", "+0.00023", "no"],
        ["TCN", "-0.00000", "-0.00062 to +0.00068", "-0.00045", "no"],
        ["RWKV-7", "-0.00059", "-0.00138 to +0.00027", "+0.00013", "no"],
        ["xLSTM 7B-style block", "-0.00094", "-0.00184 to +0.00002", "-0.00075", "no"],
    ]
    table(s, 0.89, 1.95, [3.4, 2.0, 3.3, 2.3, 0.9], rows, aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.CENTER], size=14, row_h=0.44)
    card(s, 0.89, 5.2, 11.9, 1.4)
    text(s, 1.15, 5.36, 11.4, 1.1, [
        [("Correction to an earlier pick. ", True),
         ("Pin c807884d chose xLSTM sigmoid-gate (+0.00131) by scoring the log-loss-trained models on GDT. The GDT-trained models give +0.00019, so that candidate is unconfirmed.", False)],
    ], size=15, space_after=0)
    source(s, "GDT on the validation proxy population (297 matches, per the 6 Nov report). Pooled over five seeds; match-cluster bootstrap, 1,000 resamples.")

    # 6. Regularisation chart
    s = new_slide(prs, "Stopping at 25k steps helps the GRU on 5 of 5 seeds",
                  notes="The reference is the GRU at 30k steps on the log-loss objective (metrics record steps = 30000; the script default is 12,000). It reproduces the pinned 0.451061. "
                        "Arm A (dropout 0.2, weight decay 1e-4, 30k) is not a lever: mean 0.451266, 2 of 5 seeds better. "
                        "Arm B (no regularisation, stopped at 25k) has mean 0.449989. The gain is a stopping-point effect on the GRU only; "
                        "the other arms were scored at 30k steps, so this does not rank architectures.")
    text(s, 0.89, 1.85, 7.4, 0.35, "Gain over the 30k GRU, by seed (thousandths of log loss; positive = better)",
         size=14, color=INK, bold=True, space_after=0)
    manual_bars(s, 1.6, 2.45, 6.7, 3.3,
                ["Seed 0", "Seed 1", "Seed 2", "Seed 3", "Seed 4"],
                [("Dropout 0.2 + weight decay, 30k", (0.94, -1.22, 0.19, -0.31, -0.61)),
                 ("No regularisation, 25k", (1.68, 0.93, 0.48, 1.81, 0.46))],
                [SLATE, BLUE], vmin=-1.5, vmax=2.0, ticks=[-1, 0, 1, 2])
    stat(s, 8.75, 1.95, 3.9, "+0.00107", "mean gain of the GRU at 25k steps over its 30k reference: 0.449989 against 0.451061")
    stat(s, 8.75, 3.95, 3.9, "-0.0002", "dropout 0.2 and weight decay 1e-4 at 30k: 0.451266, 2 of 5 seeds better")
    text(s, 8.75, 5.75, 3.9, 0.8, "GRU only. The other arms were scored at 30k steps, so this does not rank architectures.", size=14, color=MUTED, space_after=0)
    source(s, "Validation, log-loss objective, final-step. Arm settings from per-seed metrics on the network volume (reg_A: 30000 steps, dropout 0.2, weight decay 1e-4; reg_B: 25000 steps, none).")

    # 7. XGBoost vs GRU
    s = new_slide(prs, "XGBoost is in the same range on validation",
                  notes="XGBoost 83-input: three seeds, mean 0.450252. The GDT difference against the GRU is measured on seed 0 only. "
                        "The holdout number is the single pinned read (Oct 26 to 31, 3,315 matches). It is not comparable to the validation figures and was not used for any choice. "
                        "The 83-input model file was not found in the local models folder (see slide 12).")
    stat(s, 0.89, 1.9, 3.7, "0.45025", "XGBoost 83-input, validation log loss, mean of 3 seeds")
    stat(s, 4.95, 1.9, 3.7, "0.45106", "GRU at 30k steps, validation log loss, mean of 5 seeds")
    stat(s, 9.0, 1.9, 3.7, "0.44999", "GRU at 25k steps, validation log loss, mean of 5 seeds")
    text(s, 0.89, 4.15, 11.9, 2.4, [
        [("GDT, seed 0. ", True), ("XGBoost against the GRU: -0.00071, 95% interval -0.00269 to +0.00131. Not distinguishable.", False)],
        [("Holdout, one pinned read. ", True), ("Log loss 0.452295 on Oct 26 to 31, 3,315 matches, interval 0.449288 to 0.454983. Not used for any choice.", False)],
        [("57-input XGBoost. ", True), ("The GRU beat it on validation on every seed (pin 921fa3dd). The 83-input model is the one the explainability report covers.", False)],
    ], size=17, space_after=14)
    source(s, "Sources: REPORT_2026-11-06 sections 3.5 to 3.7; pins 1b7c86fd, 56316f69, 921fa3dd.")

    # 8. Explainability: SHAP
    s = new_slide(prs, "Loadout carries most of the XGBoost model's signal",
                  notes="Mean absolute TreeSHAP effect on P(CT wins), validation sample, from the explainability report. "
                        "The position total is across 24 inputs, not one. These are attributions for the XGBoost model only.")
    pic(s, imgs["shap"], 0.89, 1.75, h=5.0)
    stat(s, 6.9, 1.95, 4.9, "13.3", "points per second from the loadout difference, CT minus T")
    stat(s, 6.9, 3.45, 4.9, "3.1", "points per second from round state: alive counts, bomb, clock")
    stat(s, 6.9, 4.95, 4.9, "3.5", "points in total from the 24 position inputs")
    source(s, "Mean absolute TreeSHAP effect, validation sample. Source: XGB explainability report, Table 2 (short deck slide 2).")

    # 9. GDT by group
    s = new_slide(prs, "Loadout alone gives about 40% of the full model's GDT",
                  notes="GDT is the gain over the lookup table. The full model scores 0.0254. Loadout alone scores 0.0098, which is about 39% of the full model. "
                        "Round state alone scores below the table. The population here (295 matches) differs from the 297 in the 6 November report; to reconcile.")
    pic(s, imgs["gdt_groups"], 0.89, 1.85, w=8.2)
    text(s, 9.5, 2.0, 3.3, 4.2, [
        [("Full model: ", True), ("0.0254 GDT.", False)],
        [("Loadout alone: ", True), ("0.0098.", False)],
        [("Round state alone: ", True), ("below the lookup table.", False)],
    ], size=16, space_after=14)
    source(s, "GDT on 295 matches, mean of three seeds (explainability deck). The report uses 297; to reconcile. Source: XGB explainability report, short deck slide 3.")

    # 10. Calibration
    s = new_slide(prs, "XGBoost is well calibrated by decile on validation",
                  notes="Calibration is checked per second, in deciles. The round-level check in 1-versus-5 states (1.48 against 1, over 1,033 rounds) is in the explainability report, "
                        "but its unit and sample are not confirmed in the records, so it is left off this slide until they are. "
                        "Wilson intervals treat seconds as independent, so they are too narrow.")
    pic(s, imgs["calibration"], 0.89, 1.75, h=4.9)
    stat(s, 6.6, 2.1, 5.2, "0.009", "largest gap between predicted and observed CT-win rate, in any decile")
    text(s, 6.6, 4.0, 5.2, 1.6,
         "The round-level check in 1-versus-5 states is not shown here: its unit and sample need confirming against the explainability report first.",
         size=15, color=MUTED, space_after=0)
    source(s, "83-input XGBoost booster on all 1,515,228 validation seconds. Source: XGB explainability report (short deck slide 7).")

    # 11. Failure analysis
    s = new_slide(prs, "No arm fails to beat the table in any segment",
                  notes="Computed by seg_check.py on the GDT-trained proxy rows (297 matches, validation only). Segments are planted state, time-in-round thirds, alive counts and economy levels; "
                        "only segments with at least 20,000 seconds are counted (12 in total). The spread is the range across the five-seed GDT arms of (GRU loss minus arm loss) in one segment. "
                        "Attribution for the sequence models is not covered.")
    cards = [
        ("Beats the table everywhere", "All 19 arm variants beat the lookup table in all 12 segments with at least 20,000 seconds."),
        ("Spread between GDT arms", "Five-seed GDT arms against the GRU: median spread 0.0011 across segments, widest 0.0016 (planted = 1)."),
        ("Not done yet", "Attribution for the sequence models. This analysis covers scenario-level gains only."),
    ]
    for i, (head, body) in enumerate(cards):
        x = 0.89 + i * 4.05
        card(s, x, 2.0, 3.75, 3.4)
        text(s, x + 0.3, 2.3, 3.2, 0.9, head, size=19, color=BLUE, bold=True, space_after=0)
        text(s, x + 0.3, 3.35, 3.2, 1.9, body, size=16, color=INK, space_after=0)
    text(s, 0.89, 5.75, 11.9, 0.8,
         "Log-loss-trained five-seed means beat the GRU by up to 0.0011 (xLSTM 7B-style block). Source: REPORT section 3.1.",
         size=16, color=INK, space_after=0)
    source(s, "Source: seg_check.py on the GDT-trained proxy rows (297 matches, validation only); segments with at least 20,000 seconds.")

    # 12. Weights, data, PII
    s = new_slide(prs, "Weights, data and PII status",
                  notes="Weights status is from the inventory of S3 and local copies on 6 Nov 2026. The PII audit was still running when this slide was written, so its row is pending. "
                        "An automated pattern and schema scan is not proof of absence, and the slide does not describe the dataset as cleared.")
    rows = [
        ["Item", "Where", "Status"],
        ["GDT-objective weights, 47 arm-seeds", "S3 results/weights_b1-b8, weights_x1-x3; models/gdt_objective; local master", "saved; sha256 matches across all three copies"],
        ["Log-loss weights", "none: pos24 non-GDT folders and local_laptop (S3)", "not saved; retraining needed to export"],
        ["Regularisation runs, A and B", "S3 results/reg_A, reg_B", "metrics and predictions only; no weights"],
        ["TCN extra seeds (0, 1, 3, 4)", "S3 results/weights_tcnr0, 1, 3, 4", "4 weight files, not in the local master"],
        ["XGBoost 83-input model file", "not found in local models or diagnostics", "to locate before presenting; served model is 58-input v8"],
        ["Dataset personal data", "S3 volume and repo", "PII review pending; do not treat as cleared"],
        ["Network volume pureskill-train-euro1", "z7yvc40ykx, EU-RO-1", "kept until final training and backup"],
    ]
    table(s, 0.89, 1.95, [3.5, 4.7, 3.7], rows, size=13, row_h=0.56)
    source(s, "Sources: weights inventory and S3 listings (6 Nov 2026); weights_manifest.csv (sha256, 47 rows). PII audit still running at time of writing.")

    # 13. Next steps
    s = new_slide(prs, "Decisions and next steps",
                  notes="The 25k re-run and the volume clear-out are proposals, not decisions. The 25k re-run needs a new pin before any run.")
    text(s, 0.89, 1.95, 11.9, 4.6, [
        [("1. Re-run the five-seed arms at 25k steps (proposed). ", True), ("Tests whether the step count changes the ranking. Needs a new pin first.", False)],
        [("2. Holdout. ", True), ("No candidate passes the pinned rules. Decide: keep the holdout on hold, or pre-register a different rule before any read.", False)],
        [("3. Dr Freeman's GDT confirmation. ", True), ("Recorded from your report of his instruction. His written confirmation is still needed.", False)],
        [("4. Population. ", True), ("The explainability deck gives 295 matches for its GDT figures; the 6 November report gives 297. Reconcile before presenting.", False)],
        [("5. XGBoost 83-input model file. ", True), ("Not found locally; locate it, or correct the 83-input references.", False)],
        [("6. Volumes (proposed). ", True), ("Back up, then clear pureskill-train-euro1 after final training. Needs your explicit confirmation before any delete.", False)],
    ], size=15, space_after=10)
    source(s, "Open items for 6 November 2026.")

    # 14. Appendix pins
    s = new_slide(prs, "Appendix: pins behind the decisions",
                  notes="Pins as cited in the 6 Nov report. a3ceeda7 was created today in Meridian for the regularisation result. c1f31a9d is recorded in the working notes and has not been re-checked in Meridian. "
                        "Result pins for the GDT-trained runs and for the round-2 log-loss test are not recorded.")
    text(s, 0.89, 1.95, 5.7, 3.6, [
        [("64b9d52d  ", True), ("validation only for every choice", False)],
        [("064d46c76  ", True), ("GDT objective (PROVENANCE writes it as 64d46c76)", False)],
        [("079f7189  ", True), ("amendment on the GDT objective", False)],
        [("c807884d  ", True), ("earlier xLSTM sigmoid pick on log-loss models (did not reproduce on GDT)", False)],
        [("0d97f8f6  ", True), ("holdout candidate, on hold", False)],
        [("31f6825b  ", True), ("round-2 log-loss rule", False)],
    ], size=15, space_after=10)
    text(s, 6.9, 1.95, 5.9, 3.6, [
        [("1b7c86fd  ", True), ("XGBoost 83-input, three-seed validation", False)],
        [("56316f69  ", True), ("XGBoost holdout read (single)", False)],
        [("921fa3dd  ", True), ("GRU beats the 57-input XGBoost on every seed", False)],
        [("c1f31a9d  ", True), ("regularisation test: decision rule", False)],
        [("a3ceeda7  ", True), ("regularisation test: result (created 6 Nov 2026 in Meridian)", False)],
    ], size=15, space_after=10)
    text(s, 0.89, 5.7, 11.9, 0.9,
         "Not recorded: result pins for the GDT-trained runs and for the round-2 log-loss test (PROVENANCE). c1f31a9d is recorded in the working notes and not yet re-checked in Meridian.",
         size=14, color=MUTED, space_after=0)

    prs.core_properties.title = "CS2 win-probability: finished models, as of 6 November 2026"
    prs.save(str(OUT))
    print("saved", OUT, "slides:", len(prs.slides))


if __name__ == "__main__":
    build()
