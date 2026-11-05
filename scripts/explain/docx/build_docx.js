const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, ImageRun,
  AlignmentType, LevelFormat, HeadingLevel, Footer, PageNumber, BorderStyle,
} = require("docx");

const DATA = process.argv[2];
const OUT = process.argv[3];
const d = JSON.parse(fs.readFileSync(DATA, "utf8"));
const USABLE = 10080; // Letter width 12240 minus 0.75 in margins each side

const FONT = "Arial";
const BODY = 20; // 10 pt
const SMALL = 16; // 8 pt in tables
const NAVY = "1F3864";

function run(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: opts.size || BODY, bold: !!opts.bold, italics: !!opts.italics, color: opts.color });
}
function para(text, opts = {}) {
  return new Paragraph({
    spacing: { before: opts.before ?? 60, after: opts.after ?? 100, line: 276 },
    alignment: opts.align,
    children: Array.isArray(text) ? text : [run(text, opts)],
  });
}
function bullet(text) {
  const parts = Array.isArray(text) ? text : [run(text)];
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80, line: 276 }, children: parts });
}
function heading(text, level) {
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: FONT, size: level === 1 ? 28 : 23, bold: true, color: NAVY })],
    spacing: { before: level === 1 ? 300 : 200, after: 120 },
    keepNext: true,
  });
}
function caption(text) {
  return new Paragraph({ spacing: { before: 40, after: 220 }, children: [run(text, { size: 16, italics: true, color: "555555" })] });
}
function cell(text, width, header, numeric) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new Paragraph({
      alignment: numeric && !header ? AlignmentType.RIGHT : AlignmentType.LEFT,
      spacing: { after: 0 },
      children: [new TextRun({ text: String(text ?? ""), font: FONT, size: SMALL, bold: !!header })],
    })],
  });
}
function table(headers, rows, widths, numericCols = []) {
  const total = widths.reduce((a, b) => a + b, 0);
  if (Math.abs(total - USABLE) > 2) throw new Error(`column widths ${total} != ${USABLE}`);
  const head = new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], true, false)) });
  const body = rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, widths[i], false, numericCols.includes(i))) }));
  return new Table({
    width: { size: USABLE, type: WidthType.DXA },
    columnWidths: widths,
    rows: [head, ...body],
  });
}
function figure(f, maxW = 640) {
  const scale = Math.min(1, maxW / f.w);
  const w = Math.round(f.w * scale), h = Math.round(f.h * scale);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    keepNext: true,
    spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(f.path), transformation: { width: w, height: h },
      altText: { title: path.basename(f.path), description: path.basename(f.path), name: path.basename(f.path) } })],
  });
}

const body = [];

body.push(new Paragraph({
  spacing: { before: 0, after: 80 },
  children: [new TextRun({ text: "What drives the XGBoost win-probability model, input by input", font: FONT, size: 36, bold: true, color: NAVY })],
}));
body.push(para("83-input candidate (explained) and 57-input submission model (v6). Validation seconds only. 5 November 2026.",
  { color: "555555", after: 200 }));

// 1 Summary
body.push(heading("1. Summary", 1));
body.push(bullet([run("Economy drives most of the movement. ", { bold: true }),
  run("The loadout-difference inputs (team HP, equipment, armor, rifles) move P(CT wins) by about 13 pp per second on average. The round-state group (alive counts, clock, bomb) is about 3 pp, and the position inputs together are about 3.5 pp.")]));
body.push(bullet([run("Per-unit effects, from the linear slope of each input's effect: ", { bold: true }),
  run("each 100 HP of team HP difference (CT minus T) is worth about 3.7 pp; each armored player of difference about 1.9 pp; each extra rifle of difference about 0.75 pp; each $1,000 of equipment difference about 0.37 pp; each alive player of CT minus T about 1.6 pp.")]));
body.push(bullet([run("Position: ", { bold: true }),
  run("defuse slack (about +0.14 pp per second of slack, when the bomb is planted) and bomb distance matter most. Site geometry is smaller, the VIEW inputs are small, and the BLIND inputs have no measurable effect.")]));
body.push(bullet([run("Read these with care. ", { bold: true }),
  run("The explained model is a CPU refit that matches the sprint's log loss but not its per-row predictions. The loadout inputs probably also carry the alive-count signal, because dead players lose HP and equipment. Permutation ranks are distorted for the missing-value, sentinel and difference inputs.")]));

// 2 What was explained
body.push(heading("2. What was explained, and how", 1));
body.push(bullet("Model: 59 base inputs plus GEO (16), VIEW (4) and BLIND (4). Depth 4, eta 0.03, min_child_weight 20, subsample 0.8, early stop 100, seed 0. Best iteration 2,695."));
body.push(bullet("Explained on 100,000 random validation seconds out of 1,515,228. The holdout (Oct 26 to 31) and the frozen test were not used."));
body.push(bullet("TreeSHAP (XGBoost pred_contribs) gives each input's contribution to the log-odds. Its additivity check passed (max error 3.5e-5)."));
body.push(bullet("Effect in pp per second for an input: 100 × [sigmoid(z) − sigmoid(z − φ)], where z is the row's log-odds and φ the input's SHAP value. Mean absolute effect is the average over rows. Signed effect shows direction."));
body.push(bullet("Permutation importance (black box) shuffles an input (or a bundle) over all 1,515,228 validation seconds and records the rise in log loss."));
body.push(bullet("The CPU refit is not the GPU model file. Its validation log loss is 0.450210 against 0.450228 for the sprint, but per-row predictions differ by up to 0.25 (mean 0.0043)."));

// 3 Economy and health inputs
body.push(heading("3. Economy and health inputs, precisely", 1));
body.push(para("Slope = change in effect (pp) per one unit of the input. Change P10 to P90 = mean effect of the top 10% of values minus the bottom 10%. For the difference inputs, a positive slope means the CT side has more of the thing."));
body.push(table(
  ["Input", "Unit", "Mean |effect| (pp/s)", "Slope (pp/unit)", "Change P10 to P90 (pp)", "Perm. Δ log loss"],
  d.loadout_rows,
  [2050, 3230, 1300, 1300, 1400, 800],
  [2, 3, 4],
));
body.push(caption("Table 1. Economy and health inputs. Units from the cache definitions: HP and armor points are team sums; equipment is dollars."));
body.push(figure(d.figs.curves_loadout));
body.push(caption("Figure 1. Signed effect on P(CT wins) against each input's value, binned (validation sample). Orange background = flagged in the team's notes."));

// 4 Bundles
body.push(heading("4. What each bundle adds", 1));
body.push(para("Bundles sum the SHAP values of their member inputs, so the bundle effect is exact for the model. Compare bundles with the mean absolute effect; the permutation column is a black-box check and is distorted for the bomb and difference bundles."));
body.push(table(
  ["Bundle", "Inputs", "Mean |effect| (pp/s)", "Mean signed (pp/s)", "Perm. Δ log loss"],
  d.bundle_rows,
  [3280, 900, 1900, 1900, 2100],
  [1, 2, 3, 4],
));
body.push(caption("Table 2. Bundle effects, 83-input model. ALL_LOADOUT and ALL_POSITION are sums across their members; they have no permutation value."));
body.push(figure(d.figs.bundles, 600));
body.push(caption("Figure 2. What each bundle adds to P(CT wins), mean absolute effect in pp per second."));

// 5 Top 24 inputs
body.push(heading("5. The 24 largest individual inputs", 1));
body.push(table(
  ["Input", "Bundle", "Mean |effect| (pp/s)", "Slope (pp/unit)", "Change P10 to P90 (pp)", "Perm. Δ", "SHAP rank"],
  d.top_rows.map((r) => [r[0], r[1], r[3], r[4], r[5], r[6], r[7]]),
  [2100, 2800, 1050, 1150, 1200, 1000, 780],
  [2, 3, 4, 5, 6],
));
body.push(caption("Table 3. Largest inputs by mean absolute SHAP effect. Units are in Table 1 and the figures. Perm. Δ is the permutation log-loss increase; SHAP rank is the position among the 83 inputs."));
body.push(figure(d.figs.curves_top_a));
body.push(caption("Figure 3a. Signed effect against value for the largest inputs, ranks 1 to 12 (binned)."));
body.push(figure(d.figs.curves_top_b));
body.push(caption("Figure 3b. Signed effect against value for the largest inputs, ranks 13 to 24 (binned)."));

// 6 SHAP vs permutation
body.push(heading("6. Where SHAP and permutation disagree, and why", 1));
body.push(table(
  ["Input", "SHAP rank", "Mean |effect| (pp/s)", "Perm. Δ log loss", "Why the two differ"],
  d.disagree_rows,
  [1800, 900, 1500, 1500, 4380],
  [1, 2, 3],
));
body.push(caption("Table 4. Inputs where the two methods disagree most."));
body.push(figure(d.figs.shap_vs_perm, 520));
body.push(caption("Figure 4. Mean absolute SHAP effect against permutation log-loss increase, per input."));

// 7 Flags and ablation
body.push(heading("7. Flags from the team's notes, and retrain evidence", 1));
body.push(table(
  ["Input", "Note", "Mean |effect| (pp/s)", "Perm. Δ log loss", "Retrain ablation (sprint, log loss)"],
  d.flags,
  [2300, 3000, 1500, 1600, 1680],
  [2, 3],
));
body.push(caption("Table 5. Flagged inputs. The retrain column is from the sprint at depth 4, eta 0.05, 3 seeds, relative to base (59 inputs)."));
body.push(table(
  ["Feature set", "Δ log loss vs base", "Note"],
  d.ablation,
  [3600, 2600, 3880],
  [1],
));
body.push(caption("Table 6. Retrain ablation of position groups. Negative = better than base."));

// 8 57-input model
body.push(heading("8. The 57-input submission model", 1));
if (d.cmp57_rows) {
  const m = d.meta57;
  body.push(para(`Model winprob_v6_drop_heavy: 57 inputs, no position inputs, heavy weapons dropped. Best iteration ${m.best_iteration}. Validation log loss ${Number(m.validation_logloss_all_rows).toFixed(6)} on all ${Number(m.validation_rows).toLocaleString("en-US")} validation seconds. It was explained on the same ${Number(m.shap_rows).toLocaleString("en-US")} seconds as the 83-input model, and the bundle columns below compare the two on identical rows.`));
  body.push(bullet("Without the position inputs, round state takes more of the signal: 4.7 pp against 3.1 pp, and its permutation cost is 0.125 log loss against 0.043."));
  body.push(bullet("The bomb-timing inputs take over what defuse slack and bomb distance did in the 83 model. seconds_since_plant rises from 0.25 pp to 1.79 pp, and bomb_planted from 0.12 pp to 1.23 pp. Their permutation values are inflated by the -1 sentinel (Section 6)."));
  body.push(bullet("The economy inputs barely change: LOADOUT_DIFF is 13.3 pp in the 83 model and 14.1 pp in the 57 model."));
  body.push(bullet("Validation log loss is 0.453557 for the 57 model against 0.450210 for the 83 refit. The position inputs are worth about 0.0033 log loss, close to the sprint's 0.0034."));
  body.push(table(
    ["Bundle", "83-input |effect| (pp/s)", "57-input |effect| (pp/s)", "83-input perm. Δ", "57-input perm. Δ"],
    d.cmp57_rows,
    [3080, 1800, 1800, 1700, 1700],
    [1, 2, 3, 4],
  ));
  body.push(caption("Table 7. Bundle effects, 83-input vs 57-input model, on the same validation seconds."));
  body.push(table(
    ["Input", "Bundle", "Mean |effect| (pp/s)", "Mean signed (pp/s)", "Perm. Δ log loss"],
    d.top57_rows,
    [2600, 2600, 1800, 1800, 1280],
    [2, 3, 4],
  ));
  body.push(caption("Table 8. The 20 largest inputs in the 57-input model."));
} else {
  body.push(para("The 57-input explanation was still running when this report was built."));
}

// 9 Next steps
body.push(heading("9. Next steps", 1));
body.push(bullet("Retrain ablations on the WSL GPU (about 2 minutes per fit, against 83 minutes for the CPU refit): drop BLIND, drop VIEW, drop the armor sums, and remove the bomb inputs from GEO."));
body.push(bullet("Check the alive-count question: fit a loadout-only and a round-state-only model, and see how much alive count adds once HP and equipment are present."));
body.push(bullet("Make the feature changes from the notes (rebuild VIEW, BLIND and defuse slack; drop armor sums), then rerun this explanation on the new feature set."));
body.push(bullet("Nothing is committed or pushed."));

const doc = new Document({
  creator: "Claude",
  title: "What drives the XGBoost win-probability model",
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: NAVY }, paragraph: { spacing: { before: 300, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 23, bold: true, color: NAVY }, paragraph: { spacing: { before: 200, after: 120 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 360, hanging: 260 } } } }] }],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], font: FONT, size: 16, color: "777777" })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, buf.length, "bytes");
});
