// Builds the explainability report as a paper: abstract, body, then appendices with the detailed tables.
// Usage: node build_docx.js <report_data.json> <out.docx>   (report_data.json is written by prep_docx_data.py)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, ImageRun,
  AlignmentType, LevelFormat, HeadingLevel, Footer, PageNumber, PageBreak,
} = require("docx");

const DATA = process.argv[2];
const OUT = process.argv[3];
const d = JSON.parse(fs.readFileSync(DATA, "utf8"));
const K = d.key;
const USABLE = 10080; // Letter width 12240 minus 0.75 in margins each side

const FONT = "Arial";
const BODY = 20; // 10 pt
const SMALL = 16; // 8 pt in tables
const NAVY = "1F3864";
const DETAILED_DECK = "https://claude.ai/artifact/<artifact-id>";
const SHORT_DECK = process.env.SHORT_DECK_URL || "";

// ---- number formatting -------------------------------------------------------------------------------------------
const f = (v, n) => v.toFixed(n);
function sgn(v, n = 4) {
  if (Math.abs(v) < 0.5 * 10 ** -n) return (0).toFixed(n);
  return (v > 0 ? "+" : "−") + Math.abs(v).toFixed(n);
}
const rng = (lo, hi, n) => `${sgn(lo, n)} to ${sgn(hi, n)}`;

// ---- building blocks -----------------------------------------------------------------------------------------------
function run(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: opts.size || BODY, bold: !!opts.bold, italics: !!opts.italics, color: opts.color });
}
function para(text, opts = {}) {
  return new Paragraph({
    spacing: { before: opts.before ?? 60, after: opts.after ?? 110, line: 276 },
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
  return new Table({ width: { size: USABLE, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}
function figure(fig, maxW = 640) {
  if (!fig) return para("[figure not available in this build]", { color: "888888" });
  const scale = Math.min(1, maxW / fig.w);
  const w = Math.round(fig.w * scale), h = Math.round(fig.h * scale);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    keepNext: true,
    spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(fig.path), transformation: { width: w, height: h },
      altText: { title: path.basename(fig.path), description: path.basename(fig.path), name: path.basename(fig.path) } })],
  });
}
function abstractBox(paragraphs) {
  return new Table({
    width: { size: USABLE, type: WidthType.DXA },
    columnWidths: [USABLE],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: USABLE, type: WidthType.DXA },
      shading: { fill: "F3F6FA", type: ShadingType.CLEAR, color: "auto" },
      margins: { top: 140, bottom: 140, left: 200, right: 200 },
      children: paragraphs.map((t) => new Paragraph({ spacing: { after: 110, line: 288 }, children: [run(t, { size: 19 })] })),
    })] })],
  });
}
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

// ---- the numbers the prose uses -----------------------------------------------------------------------------------
const full = K.full, np = K.no_position, c77 = K.checklist_77, mrs = K.minus_round_state;
const bundleByName = Object.fromEntries(d.bundle_rows.map((r) => [r[0], r]));
const loadoutDiff = bundleByName["LOADOUT_DIFF"];
const position = d.bundle_rows.find((r) => r[0] === "ALL_POSITION (GEO+VIEW+BLIND)");
const slope = Object.fromEntries(d.loadout_rows.map((r) => [r[0], r]));

// ---- front matter ----------------------------------------------------------------------------------------------
const body = [];
body.push(new Paragraph({
  spacing: { before: 0, after: 80 },
  children: [new TextRun({ text: "Which inputs carry a CS2 round win-probability model", font: FONT, size: 36, bold: true, color: NAVY })],
}));
body.push(para("An explanation of an 83-input model, and a test of its input groups on the team's metric, GDT.",
  { size: 24, color: "333333", after: 60 }));
body.push(para("Working draft, 6 November 2026. Companion slides: " +
  (SHORT_DECK || "(short deck to follow)") + ".",
  { color: "555555", after: 200, size: 18 }));

body.push(heading("Abstract", 1));
body.push(abstractBox([
  "A win-probability model estimates, second by second, how likely the counter-terrorist (CT) side is to win a Counter-Strike 2 round. " +
  "We explain an 83-input gradient-boosted model of this probability and test which groups of inputs it needs. TreeSHAP attributes each " +
  "prediction to its inputs, and permutation importance gives a black-box check. The loadout difference between the two teams carries most " +
  `of the signal, averaging ${f(loadoutDiff[2] * 1, 1)} percentage points of probability per second. The 24 position inputs total ${f(parseFloat(position[2]), 1)} points.`,
  "We retrained the model without each group and scored every retrain with the team's evaluation metric, GDT: the gain over a fixed lookup table, " +
  `on 295 matches. The full model scores ${f(full.gdt, 4)}. The loadout inputs alone score ${f(K.loadout_only.gdt, 4)}. ` +
  `Removing the 24 position inputs gives ${f(np.gdt, 4)} (${sgn(np.dgdt)} against the full model); site and spread alone costs ${sgn(K.minus_site_spread.dgdt)}, and contact alone nothing measurable.` +
  `The checklist's 77-input fallback gives ${f(c77.gdt, 4)} (${sgn(c77.dgdt)}), and log loss ${sgn(c77.dlogloss, 5)} against the full model. ` +
  `Dropping the alive counts, the armour sums or the blindness inputs changes GDT by 0.0002 or less. Dropping the VIEW or bomb-geometry group changes it by about 0.0005 each.`,
  "Calibration holds: predicted and observed CT-win rates agree within 0.01 in every decile over 1.5 million validation seconds, and in the " +
  "1-versus-5 state the expected and observed CT round wins agree. Several inputs remain flagged for rebuild, and the design matrix has not " +
  "been regenerated from raw data, so the reproduction is incomplete.",
]));

// ---- 1 introduction ----------------------------------------------------------------------------------------------------
body.push(heading("1. Introduction", 1));
body.push(para("A round win-probability model gives, at every second of a Counter-Strike 2 round, the chance that the CT side wins. " +
  "The team uses such a model to read live rounds and to judge which game-state inputs are worth their cost. Two questions follow. " +
  "Which inputs drive the prediction, and by how much? And does each input group earn its place when the model is judged on the metric " +
  "the team actually uses?"));
body.push(para("The model is trained to minimise log loss on every second of the training matches. The team scores models with GDT, " +
  "the gain over a fixed lookup table of CT-win rates by coarse situation (who is alive, the bomb, the time left, and each side's gear). " +
  "GDT is computed on sampled datapoints, one per round per situation, so the two metrics weight the seconds of a round differently. " +
  "Section 2.4 explains the difference, and why it matters for the input decisions."));
body.push(para("This report answers five questions: (1) which inputs carry the signal, and how much; (2) what the position inputs add beyond the " +
  "non-positional inputs; (3) whether the 83-input model is preferable to the 77-input checklist fallback; (4) whether single predictions and " +
  "example matches on GDT look sensible; and (5) whether the model is calibrated. Appendices hold the detailed tables."));

// ---- 2 data, model and metric ------------------------------------------------------------------------------------------
body.push(heading("2. Data, model and metric", 1));
body.push(heading("2.1 Data and splits", 2));
body.push(para("The training pool is 14,804 matches dated 2 August to 13 October 2026. The validation sample holds 1,168 matches and " +
  "1,515,228 seconds. Explanations use a sample of 100,000 validation seconds. GDT is scored on a population of 295 matches dated 1 September " +
  "to 8 October, restricted to rounds that start five against five: 5,239 rounds and 354,519 seconds. The Oct 26 to 31 holdout and the frozen " +
  "test were not read for this work. The team's handoff and the cache disagree on the match count for 9 and 10 October; Appendix G lists the " +
  "open reconciliation."));
body.push(heading("2.2 Inputs", 2));
body.push(para("The 83 inputs fall into 59 non-positional inputs (loadout, round state and map) and 24 position inputs (where the players are " +
  "and what they can see). Table 1 lists the groups."));
body.push(table(
  ["Group", "Inputs", "What it holds", "Positional?"],
  [
    ["Loadout difference (CT minus T)", "6", "Team HP, equipment, rifles, SMGs, AWPs and armoured players, as CT minus T", "No"],
    ["Loadout, each side", "40", "Per-side equipment, weapons, health and armour, grenades and kit", "No"],
    ["Round state", "6", "Alive counts, the alive difference, bomb planted, round clock, seconds since the plant", "No"],
    ["Map", "7", "Indicator for each of the seven maps", "No"],
    ["Bomb geometry", "4", "Distance to the nearest bomb spot, bomb-area counts, and defuse slack", "Yes"],
    ["Site and spread geometry", "9", "Distances to the two sites, spread of each side, centroid distance", "Yes"],
    ["Contact", "3", "Nearest enemy distance and in-contact flags", "Yes"],
    ["VIEW", "4", "Sees-enemy and exposed flags for each side (flagged in the team's notes)", "Yes"],
    ["BLIND", "4", "Blind counts and sums for each side (flagged in the team's notes)", "Yes"],
  ],
  [2900, 900, 5180, 1100],
  [1],
));
body.push(caption("Table 1. The 83 inputs by group. Non-positional = 59 inputs; positional = 24 inputs. The counts add to 83."));
body.push(heading("2.3 Model", 2));
body.push(bullet("XGBoost with the binary logistic objective, log-loss evaluation, depth 4, learning rate 0.03, minimum child weight 20, " +
  "subsample 0.8 and 256 histogram bins. Early stopping after 100 rounds on a 10% split, with a 3,000-round cap. Monotone constraints on the alive counts."));
body.push(bullet(`The shipped booster is a CPU refit with best iteration 2,695. Its validation log loss is 0.450210. The sprint's GPU fit gave 0.450228. The file holds 2,796 boosted rounds: the best iteration plus 100 early-stopping rounds. Predictions and explanations must use the first 2,696 rounds (iteration_range=(0, 2696)). A plain predict uses all rounds and differs by up to 0.02.`));
body.push(bullet(`The ablations in this report refit the full model in one environment (GPU, three seeds). Their full-model log loss is ${f(full.logloss, 6)} on average, so the ablations are compared with each other, not with the shipped file.`));
body.push(heading("2.4 The metric, and why training on log loss is not the same", 2));
body.push(para("GDT is the table's log loss minus the model's log loss, averaged over sampled datapoints. The scorer draws one second per round " +
  "per situation, at random, and the table's loss is fixed. So a GDT difference between two models is exactly a difference in their log losses on " +
  "those draws. Training on every second instead weights a long-lived situation by its length: thirty seconds of the same situation count thirty " +
  "times. GDT counts it once. The objectives therefore differ in weighting, not in direction. We did not train on GDT. Section 5.1 lists the " +
  "retraining with sample weights as the next experiment."));
body.push(heading("2.5 Candidate models compared", 2));
body.push(para("Five models answer the questions in the introduction. All use the same matrix, split and fit path, so differences come from " +
  "the inputs alone."));
body.push(bullet(`Full model, 83 inputs: the model that is explained.`));
body.push(bullet(`Non-positional model, 59 inputs: the full model without the 24 position inputs.`));
body.push(bullet(`Checklist fallback, 77 inputs: the full model without the armour sums and the VIEW inputs. The team's checklist asked for these to be removed, and this is the disclosed fallback.`));
body.push(bullet(`Minus round state, 77 inputs: the full model without the six round-state inputs. The count is the same as the fallback but the inputs differ.`));
body.push(bullet(`Loadout only, 46 inputs, and round state only, 6 inputs: single-group models for reference.`));
body.push(para("A 57-input submission model (v6) drops the heavy weapons and the position inputs. It was fitted separately and is reported " +
  "in Appendix D only, for log loss and explanation, because it was not refitted under this report's settings."));

// ---- 3 methods ------------------------------------------------------------------------------------------------------------
body.push(heading("3. Methods", 1));
body.push(heading("3.1 Explanations", 2));
body.push(para("TreeSHAP gives each input's contribution to the log-odds, and the contributions add up to the prediction exactly. The effect of " +
  "an input in percentage points per second is 100 times the sigmoid of the row's log-odds minus the sigmoid of the same log-odds without that " +
  "contribution. Its mean absolute value summarises importance on the probability scale. Bundles add the contributions of their members, so bundle " +
  "effects are exact. Permutation importance shuffles one input (or a bundle) over all 1.5 million validation seconds and records the rise in " +
  "log loss. It is a black-box check, and it is distorted for inputs that are missing or sentinel-coded or that are exact functions of others."));
body.push(heading("3.2 Retraining and scoring", 2));
body.push(para("Each candidate is refitted with the same settings and three seeds. The GDT scorer is the team's own (score_gdt.py). It is run on the same " +
  "datapoints for every model, so the differences are paired. The paired difference for a seed is the candidate's GDT minus the full model's GDT on " +
  "that seed's datapoints. The table reports the mean of three seeds and the range of the paired difference over seeds."));
body.push(heading("3.3 Calibration", 2));
body.push(para("Calibration uses the shipped booster on all validation seconds, in deciles of predicted probability. Seconds within a round are not " +
  "independent, so the round-level check sums each round's mean predicted probability in the 1-versus-5 state and compares it with the number of " +
  "rounds the CT side won."));
body.push(heading("3.4 Example matches", 2));
body.push(para("Example matches are chosen by fixed rules, not by eye. Among matches with at least 60 sampled datapoints, we take the match that the " +
  "full model beats the table on most, the one it does worst on, and the two with the largest and smallest difference between the full and " +
  "non-positional models. One round from the largest-difference match is drawn second by second."));

// ---- 4 results ------------------------------------------------------------------------------------------------------------
body.push(heading("4. Results", 1));
body.push(heading("4.1 Which inputs carry the signal", 2));
body.push(para(`The loadout difference carries most of the signal, with a mean absolute effect of ${loadoutDiff[2]} percentage points per second. ` +
  `Round state follows at ${bundleByName["ROUND_STATE"][2]} points, and the bomb-distance and defuse-slack bundle at ${bundleByName["BOMB_GEO"][2]}. ` +
  `The 24 position inputs total ${position[2]} points: bomb geometry, site geometry and contact, with visibility at 0.63 and blindness at 0.03. ` +
  "Table 2 gives the bundles and Figure 1 shows them."));
body.push(table(
  ["Bundle", "Inputs", "Mean |effect| (pp per second)", "Mean signed (pp per second)", "Permutation: rise in log loss"],
  d.bundle_rows.filter((r) => !r[0].startsWith("ALL_")),
  [3280, 900, 1900, 1900, 2100],
  [1, 2, 3, 4],
));
body.push(caption("Table 2. Bundle effects for the 83-input model. Combined position and loadout totals are in Appendix A."));
body.push(figure(d.figs.bundles, 600));
body.push(caption("Figure 1. Mean absolute effect on P(CT wins) by bundle, in percentage points per second (validation sample)."));

body.push(heading("4.2 Economy and health, in per-unit terms", 2));
body.push(para(`A 100-HP difference in team health (CT minus T) moves the probability by about ${f(parseFloat(slope["st_hp_diff"][3]) * 100, 1)} points, ` +
  "and the slope is steady across the middle of its range. Each armoured player of difference is worth about " +
  `${f(parseFloat(slope["st_armor_n_diff"][3]), 1)} points, each rifle about ${f(parseFloat(slope["st_rifle_diff"][3]), 2)} points, ` +
  `and each $1,000 of equipment difference about ${f(parseFloat(slope["st_equip_diff"][3]) * 1000, 2)} points. ` +
  "Defuse slack, when the bomb is planted, adds about 0.14 points per second of slack. Appendix A.1 gives every loadout input."));

body.push(heading("4.3 What the team's metric says", 2));
body.push(para(`On GDT the full model scores ${f(full.gdt, 4)} on 295 matches (scorer interval ${d.gdt_ci}). ` +
  "Table 3 gives every drop-group and single-group retrain, paired with the full model. Negative changes mean worse than the full model, and a " +
  "negative GDT means worse than the lookup table."));
body.push(table(
  ["Feature set", "Inputs", "GDT (mean of 3 seeds)", "GDT change vs full", "Seed range of change", "Log-loss change vs full"],
  d.gdt_rows,
  [3200, 900, 1500, 1500, 1900, 1080],
  [1, 2, 3, 4, 5],
));
body.push(caption(`Table 3. Retrain ablations scored on GDT. The scorer's interval for the full model is ${d.gdt_ci}, so compare the changes, not the levels.`));
body.push(figure(d.figs.gdt_ablation, 640));
body.push(caption("Figure 2. GDT for each feature set, mean of three seeds. Whiskers show the range over seeds. Bars below zero score worse than the lookup table."));
body.push(bullet([run("Alive counts add nothing on GDT. ", { bold: true }),
  run(`Dropping them changes GDT by ${sgn(K.minus_alive.dgdt)}, while log loss rises by ${sgn(K.minus_alive.dlogloss, 5)}. This is the one set where the two metrics disagree in direction.`)]));
body.push(bullet([run("Armour sums and blindness are not measurable on GDT. ", { bold: true }),
  run(`Dropping the armour sums changes GDT by ${sgn(K.minus_armor_sums.dgdt)}, and dropping the BLIND inputs by ${sgn(K.minus_blind.dgdt)}. Neither is distinguishable from zero across seeds.`)]));
body.push(bullet([run("View and bomb geometry are small but consistent. ", { bold: true }),
  run(`Each changes GDT by about ${sgn(K.minus_view.dgdt)}. The bomb-geometry set includes defuse slack, so defuse slack is not tested alone.`)]));
body.push(bullet([run("Loadout is most of the model. ", { bold: true }),
  run(`The 46 loadout inputs alone score ${f(K.loadout_only.gdt, 4)}. Round state alone scores ${sgn(K.round_state_only.gdt)}, below the lookup table.`)]));

body.push(heading("4.4 Positional and non-positional inputs", 2));
body.push(para(`Table 4 sets the full model beside the non-positional model. The non-positional model has the 59 inputs that are not position inputs and is fitted in the same way. ` +
  `On log loss it is worse by ${sgn(np.dlogloss, 5)}, and on GDT it scores ${f(np.gdt, 4)}, a change of ${sgn(np.dgdt)} ` +
  `(seed range ${rng(np.dgdt_min, np.dgdt_max)}). The position inputs matter for the metric as a whole.` +
  "Single-group drops are smaller than the block: site and spread costs −0.0020, bomb geometry and VIEW about 0.0005 each, and contact nothing measurable. These do not add up to the block, because the groups overlap. The 57-input submission model also has no position inputs, and its validation log loss is " +
  `${f(K.v6_57.logloss, 6)} (Appendix D).`));
body.push(table(
  ["Model", "Inputs", "Validation log loss (mean of 3 seeds)", "Log loss vs full", "GDT (mean)", "GDT vs full (seed range)"],
  d.comparison_rows,
  [3080, 800, 1700, 1300, 1200, 2000],
  [1, 2, 3, 4],
));
body.push(caption("Table 4. Positional and non-positional models, and the checklist fallback, on log loss and GDT. The 57-input submission model is not scored on GDT here."));

body.push(heading("4.5 The 83-input model against the 77-input variants", 2));
body.push(para(`The team's checklist asked for the armour sums and the VIEW inputs to be removed. The 83-input model keeps them, and the disclosed fallback removes them, giving 77 inputs. ` +
  `The fallback changes log loss by ${sgn(c77.dlogloss, 5)} and GDT by ${sgn(c77.dgdt)} (seed range ${rng(c77.dgdt_min, c77.dgdt_max)}). ` +
  `A second 77-input set removes round state instead. It changes log loss by ${sgn(mrs.dlogloss, 5)} and GDT by ${sgn(mrs.dgdt)} (seed range ${rng(mrs.dgdt_min, mrs.dgdt_max)}).`));
body.push(para(`Against the checklist fallback, the 83-input model is better on both measures, and the sign holds in every seed. Its GDT is higher by ${f(-c77.dgdt, 4)} (the fallback is negative in all three seeds) and its log loss is lower by ${f(c77.dlogloss, 5)} (positive in all three seeds). Most of the GDT loss from the fallback comes from removing the VIEW inputs, partly offset by removing the armour sums, which are close to zero on their own. The margin is small. If VIEW is rebuilt and kept, the 83-input model is the better choice on the evidence; if VIEW is removed for reliability, the fallback costs about ${f(-c77.dgdt, 4)} GDT. Against minus round state, GDT cannot separate the two models (the difference changes sign across seeds), while log loss favours the 83-input model in every seed. The round-state group therefore adds no measurable GDT over the other inputs, but it does improve log loss, and that is the reason to keep it.`));

body.push(heading("4.6 Example matches on GDT", 2));
body.push(para(`Where does the full model beat the lookup table? Figure 3 averages the gain by alive count. The full model beats the table in ${d.gdt_summary ? (d.gdt_summary.share_beats_table * 100).toFixed(0) : "most"}% of the ${d.gdt_summary ? d.gdt_summary.matches : 295} matches. Figure 4 shows the spread across matches, and the paired difference between the full and non-positional models. Table 5 names four example matches chosen by the rules in Section 3.4.`));
body.push(figure(d.figs.gdt_by_alive, 520));
body.push(caption("Figure 3. Mean gain over the lookup table by alive count, full model, seed-0 datapoints. Blue beats the table, red does worse; the count under each value is the number of datapoints."));
body.push(figure(d.figs.gdt_per_match, 640));
body.push(caption("Figure 4. Per-match gain over the table for the full model (left), and the full model minus the non-positional model (right)."));
if (d.figs.gdt_example_round) {
  body.push(figure(d.figs.gdt_example_round, 640));
  body.push(caption("Figure 5. One round from the match where the position inputs help most, second by second. The table, the non-positional model and the full model are shown with the round's outcome."));
}
body.push(table(
  ["Example", "Match", "Date", "Sampled datapoints", "Full model gain", "Non-positional gain", "Full minus non-positional"],
  d.examples_rows,
  [2200, 1400, 1080, 1100, 1500, 1500, 1300],
  [3, 4, 5, 6],
));
body.push(caption("Table 5. Example matches on GDT (seed-0 draw). A gain is the table's log loss minus the model's, per sampled datapoint, averaged over the match."));

body.push(heading("4.7 One prediction", 2));
body.push(para(`The largest-loss second in the validation sample is round 2 of a match on 9 October. The model gave CT a 0.11% chance, and CT won the round. ` +
  "Figure 6 walks the prediction down from the average value, input by input. Team HP difference (CT has 100 HP, T has 370) and CT alive minus T alive account for most of the push against CT. Appendix E gives the four example seconds and two contrasts."));
body.push(figure(d.figs.waterfall, 640));
body.push(caption("Figure 6. Contributions to the log-odds of CT winning for the largest-loss second. The base value is the average over the sample; the bars sum to the prediction."));

body.push(heading("4.8 Explaining whole matches", 2));
{
  const pct = (v) => (100 * v).toFixed(0) + "%";
  body.push(para("The waterfall in Section 4.7 explains one prediction. Matches are what people watch, so we also explain every row of three validation matches: the largest-loss match, the match where the position inputs help most on GDT, and the match where the full model does worst against the table. For each row, TreeSHAP splits the log-odds into the ten bundles of Table 1, and the bundles sum exactly to the prediction. Figure 7 shows the largest-loss match, and Appendix E.4 the other two."));
  body.push(figure(d.figs.match_largest_loss, 760));
  body.push(caption("Figure 7. P(CT wins) over the largest-loss match (top), and each bundle's contribution to the log-odds for every row (bottom). Rows are the seconds the matrix holds, in round order. The bar on the top panel is the round's outcome: blue for a CT win, red for a T win."));
  body.push(table(["Round", "Seconds", "P(CT wins)", "Change", "Biggest mover (log-odds)", "Second mover (log-odds)", "Outcome"],
    d.match.swing_rows, [800, 1200, 1500, 1200, 2300, 2300, 780], []));
  body.push(caption("Table 6. The five largest moves in P(CT wins) between consecutive rows of the largest-loss match, with the two bundles that moved most."));
  body.push(table(["Bundle", "Largest-loss match", "Position inputs help most", "Worst against the table"],
    d.match.share_rows, [3480, 2200, 2200, 2200], [1, 2, 3]));
  body.push(caption("Table 7. Share of the absolute log-odds contribution by bundle, for each match, summed over its rows."));
  body.push(para(`The loadout bundles account for ${pct(d.match.loadout_total.largest_loss)}, ${pct(d.match.loadout_total.positional_help)} and ${pct(d.match.loadout_total.worst_vs_table)} of the absolute contribution in the three matches. The largest single move in the largest-loss match is from 6% to 54% between seconds 62 and 63 of round 2. The explanation puts most of it on the CT loadout, with bomb geometry second. A move that size in one second should be checked against the raw seconds before it is read as a game event.`));
}
body.push(heading("4.9 Calibration", 2));
body.push(para("Across 1,515,228 validation seconds, the predicted probability matches the observed CT-win rate to within 0.01 in every decile " +
  "(Figure 7; Appendix C.1). In the 1-versus-5 state, the model expected 1.48 CT round wins over 1,033 rounds, and CT won one. The one win is the " +
  "round of the largest per-second miss, so the data do not show a calibration problem in that state."));
body.push(figure(d.figs.calibration, 420));
body.push(caption("Figure 7. Calibration by decile, shipped booster, all validation seconds. The intervals treat seconds as independent, so they are too narrow."));

// ---- 5 discussion --------------------------------------------------------------------------------------------------
body.push(heading("5. Discussion", 1));
body.push(heading("5.1 What the results suggest for the model", 2));
body.push(bullet("The armour sums can be removed at no measurable GDT cost, as the team's notes ask. This is the cheapest change."));
body.push(bullet("The BLIND inputs have no measurable effect on GDT. Rebuild them from the player blind events before deciding whether to keep them."));
body.push(bullet("Site and spread is the largest single position group (−0.0020 GDT alone). Rebuild it first, and then retest it. Contact has no measurable effect on either metric, so it is a candidate to drop, as the BLIND inputs are."));
body.push(bullet("The VIEW inputs are flagged as unreliable, and they carry a small GDT gain. Rebuild them first, and then retest the group."));
body.push(bullet("Defuse slack is flagged as crude. It is in the bomb-geometry group, and that group's drop test cannot separate it from its neighbours. A rebuild should be tested alone."));
body.push(bullet("Retraining with weights that equalise each round's situations would align the training loss with GDT. This is the next experiment, and it is not yet run."));
body.push(heading("5.2 What the explanations do not show", 2));
body.push(para("Explanations describe how the model uses its inputs. They do not show that an input causes the outcome. TreeSHAP splits credit between " +
  "correlated inputs, so the alive counts get little direct credit while the loadout inputs carry the signal. The drop tests are the stronger " +
  "evidence for whether a group is needed, and they say little about any one input inside a group."));

// ---- 6 limitations ------------------------------------------------------------------------------------------------------
body.push(heading("6. Limitations", 1));
body.push(bullet("GDT differences are small. The paired change for a set varies by about 0.0002 to 0.0003 across seeds, so changes of that size or less should not be read as real."));
body.push(bullet("The position effect is not additive across groups. The single-group drops (site and spread −0.0020, VIEW and bomb geometry −0.0005 each, contact about zero) sum to about 0.003, against 0.0049 for all 24 together. The overlap between groups is not measured."));
body.push(bullet("GDT here uses the 295-match validation window. It is not the frozen test days, and it is not comparable to scores the scorer's README gives on other populations."));
body.push(bullet("The shipped booster is a CPU refit, and its per-row predictions differ from the sprint's GPU run by up to 0.25 (mean 0.004), though their log losses agree closely."));
body.push(bullet("The design matrix has not been regenerated from raw data, and frozen_test_keys.json was not found. The matrix is the Nov 3 build."));
body.push(bullet("The October 9 and 10 match counts disagree between the handoff and the cache, and the choice affects the training pool."));

// ---- 7 conclusion ---------------------------------------------------------------------------------------------------
body.push(heading("7. Conclusion", 1));
body.push(para(`The loadout difference drives most of the model's signal, and round state and position inputs add to it. On GDT, the full model scores ${f(full.gdt, 4)}. ` +
  "The position inputs add about 0.005 GDT as a group, mostly through site and spread; the checklist's fallback costs a little on both measures, and the round-state group adds nothing measurable on GDT but improves log loss (Section 4.5). " +
  "The model is calibrated over the validation window. The remaining work is in the reproduction and the input rebuilds listed in Appendix G."));

// ---- appendices -------------------------------------------------------------------------------------------------------------
body.push(pageBreak());
body.push(heading("Appendices", 1));
body.push(para("The detailed tables that support the body. Each one is labelled with the section it supports.", { color: "444444" }));

body.push(heading("Appendix A. Loadout and bundle detail", 1));
body.push(heading("A.1 Economy and health inputs", 2));
body.push(para("Slope is the change in effect (percentage points) per one unit of the input. Change P10 to P90 is the mean effect of the top 10% of values minus the bottom 10%. For difference inputs, a positive slope means the CT side has more of the thing."));
body.push(table(
  ["Input", "Unit", "Mean |effect| (pp/s)", "Slope (pp/unit)", "Change P10 to P90 (pp)", "Perm. Δ log loss"],
  d.loadout_rows,
  [2050, 3230, 1300, 1300, 1400, 800],
  [2, 3, 4],
));
body.push(caption("Table A1. Economy and health inputs. Units are from the cache definitions: HP and armour are team sums; equipment is in dollars."));
body.push(figure(d.figs.curves_loadout, 640));
body.push(caption("Figure A1. Signed effect on P(CT wins) against each input's value, binned (validation sample). Flagged inputs are marked in the team's notes."));
body.push(heading("A.2 Bundle totals", 2));
body.push(table(
  ["Bundle", "Inputs", "Mean |effect| (pp/s)", "Mean signed (pp/s)", "Perm. Δ log loss"],
  d.bundle_rows,
  [3280, 900, 1900, 1900, 2100],
  [1, 2, 3, 4],
));
body.push(caption("Table A2. All bundles, including the combined position and loadout totals, which have no permutation value."));

body.push(heading("Appendix B. Ablation detail", 1));
body.push(heading("B.1 Per-seed results", 2));
body.push(table(
  ["Feature set", "Seed", "Inputs", "Validation log loss", "Log-loss change vs full (same seed)", "Best iteration"],
  d.per_seed_rows,
  [2700, 800, 900, 1800, 2380, 1500],
  [1, 2, 3, 4, 5],
));
body.push(caption("Table B1. Each retrain on each seed. The log-loss change is paired with the full model on the same seed."));
body.push(heading("B.2 Earlier sprint run (log loss only)", 2));
body.push(table(
  ["Feature set", "Δ log loss vs base", "Note"],
  d.ablation,
  [3600, 2600, 3880],
  [1],
));
body.push(caption("Table B2. Earlier sprint run of the position groups (eta 0.05, base of 59 inputs, three seeds). Superseded for decisions by Table 3; kept for the record. Negative means better than base."));

body.push(heading("Appendix C. Calibration detail", 1));
body.push(heading("C.1 Deciles", 2));
body.push(table(
  ["Decile", "Seconds", "Mean predicted P(CT wins)", "Observed CT win rate", "Observed 95% interval", "Observed minus predicted"],
  d.calib_rows,
  [1000, 1300, 2000, 1800, 2280, 1700],
  [1, 2, 3, 4, 5],
));
body.push(caption("Table C1. Calibration deciles for the shipped booster. The largest gap is 0.009, in decile 6. Wilson intervals treat seconds as independent, so they are too narrow."));
body.push(heading("C.2 Round level, 1-versus-5 states", 2));
body.push(para("Per round: expected is the sum over rounds of each round's mean predicted probability; observed is the number of rounds CT won."));
body.push(table(
  ["1-versus-5 state", "Rounds", "Matches", "Expected CT round wins", "Observed CT round wins"],
  d.onev5_rows,
  [4080, 1500, 1500, 1500, 1500],
  [1, 2, 3, 4],
));
body.push(caption("Table C2. Round-level 1-versus-5 check. The one CT win in the planted state is the same round as the largest per-second miss. A round can count in both bomb states, so the rows do not add to the first row."));

body.push(heading("Appendix D. The 57-input submission model", 1));
if (d.cmp57_rows) {
  const m = d.meta57;
  body.push(para(`Model winprob_v6_drop_heavy: 57 inputs, no position inputs, heavy weapons dropped. Best iteration ${m.best_iteration}. Validation log loss ${Number(m.validation_logloss_all_rows).toFixed(6)} on all ${Number(m.validation_rows).toLocaleString("en-US")} validation seconds. It was explained on the same ${Number(m.shap_rows).toLocaleString("en-US")} seconds as the 83-input model, so the bundle columns compare the two on identical rows.`));
  body.push(bullet("Without the position inputs, round state takes more of the signal: 4.7 points against 3.1, and its permutation cost is 0.125 log loss against 0.043."));
  body.push(bullet("The bomb-timing inputs take over what defuse slack and bomb distance did in the 83-input model. Their permutation values are inflated by the −1 sentinel used when no bomb is planted."));
  body.push(bullet("The economy inputs barely change: LOADOUT_DIFF is 13.3 points in the 83-input model and 14.1 in the 57-input model."));
  body.push(table(
    ["Bundle", "83-input |effect| (pp/s)", "57-input |effect| (pp/s)", "83-input perm. Δ", "57-input perm. Δ"],
    d.cmp57_rows,
    [3080, 1800, 1800, 1700, 1700],
    [1, 2, 3, 4],
  ));
  body.push(caption("Table D1. Bundle effects, 83-input and 57-input models, on the same validation seconds."));
  body.push(table(
    ["Input", "Bundle", "Mean |effect| (pp/s)", "Mean signed (pp/s)", "Perm. Δ log loss"],
    d.top57_rows,
    [2600, 2600, 1800, 1800, 1280],
    [2, 3, 4],
  ));
  body.push(caption("Table D2. The 20 largest inputs in the 57-input model."));
} else {
  body.push(para("The 57-input explanation was still running when this report was built."));
}

body.push(heading("Appendix E. Local explanations", 1));
body.push(heading("E.1 Four example seconds", 2));
body.push(para("Four seconds from the 100,000-second sample, chosen by fixed rules (recorded in local/local_explain_meta.json). The waterfall for each is in local/."));
body.push(table(
  ["Example", "Round", "Second", "CT won (1 = yes)", "P(CT wins)", "Logit z", "Log loss"],
  d.local_rows.map((r) => [r[0], r[4], r[5], r[6],
    ((x) => (x * 100 < 0.001 ? (x * 100).toExponential(2) : (x * 100).toFixed(4)) + "%")(parseFloat(r[7])),
    parseFloat(r[8]).toFixed(2), parseFloat(r[10]).toFixed(3)]),
  [2200, 1000, 1000, 1400, 1600, 1480, 1400],
  [],
));
body.push(caption("Table E1. The example seconds with their log-loss and probability fields."));
body.push(heading("E.2 Contrasts between the two confident calls", 2));
body.push(para("The two confident calls are P = 99.999% for CT and P = 0.00002% for T. The logit gap is 27.1, and the per-input differences sum to it exactly."));
body.push(table(
  ["Rank", "Input", "Bundle", "Value (CT call)", "Value (T call)", "Difference in logit"],
  d.contrast_rows,
  [800, 2400, 2200, 1700, 1700, 1280],
  [0, 3, 4, 5],
));
body.push(caption("Table E2. The six largest per-input contributions to the logit gap between the two calls."));

body.push(heading("E.3 Interactions between inputs", 2));
body.push(para("Pairwise SHAP interaction values on 2,000 validation seconds, for the eight largest pairs. The interactions are small beside the main effects: the largest pair, seconds since the plant with defuse slack, averages 0.039 log-odds. The standard total pair effect is twice the value in the table."));
body.push(table(["Rank", "Input A", "Input B", "Mean |interaction| (log-odds)", "Mean |SHAP| A", "Mean |SHAP| B"],
  d.interaction_rows, [700, 2600, 2600, 2000, 1090, 1090], [0, 3, 4, 5]));
body.push(caption("Table E3. Largest pairwise SHAP interactions in the validation sample (local/interactions_top_pairs.csv). The file reports xgboost's off-diagonal entry."));
body.push(heading("E.4 The other two matches", 2));
body.push(figure(d.figs.match_positional_help, 760));
body.push(caption("Figure E4. The match where the position inputs help most on GDT: each bundle's contribution, row by row."));
body.push(figure(d.figs.match_worst_vs_table, 760));
body.push(caption("Figure E5. The match where the full model does worst against the table."));

body.push(heading("Appendix F. Example matches and the GDT datapoints", 1));
body.push(para(`The full population has ${d.gdt_summary ? d.gdt_summary.matches : 295} matches and ${d.gdt_summary ? d.gdt_summary.datapoints.toLocaleString("en-US") : ""} sampled datapoints for seed 0. The median match gains ${d.gdt_summary ? sgn(d.gdt_summary.median_gain, 4) : ""} GDT per sampled datapoint.`));
body.push(table(
  ["Example", "Match", "Date", "Sampled datapoints", "Full model gain", "Non-positional gain", "Full minus non-positional"],
  d.examples_rows,
  [2200, 1400, 1080, 1100, 1500, 1500, 1300],
  [3, 4, 5, 6],
));
body.push(caption("Table F1. The example matches in Table 5, with every field."));
body.push(para("The plain mean of each model's gain over the sampled datapoints equals the scorer's own GDT to four decimal places. The check is run in gdt_examples.py against summary.json for seed 0."));

body.push(heading("Appendix G. Open items and reconciliation", 1));
body.push(bullet("Reproduce the design matrix from the raw tomes. sprint_prepare.py has not been rerun, frozen_test_keys.json is not on this machine, and the builder of cache_pos_2026_trainctr.parquet is not identified."));
body.push(bullet("Reconcile October 9 and 10. The handoff gives 2,598 matches for these days, and the training cache has 4,865."));
body.push(bullet("Rebuild the flagged inputs from the team's notes: VIEW (sees-enemy and exposed), BLIND (from player blind events) and defuse slack (time to reach and defuse). Then rerun the explanation."));
body.push(bullet("The Oct 26 to 31 holdout has one recorded read from an earlier candidate and no new scoring. Any new read needs a pre-registration pin and an explicit go-ahead."));
body.push(bullet("The deep model has a handoff only (handover/deep-model/README.md). No training has run."));

body.push(heading("Appendix H. Reproduction", 1));
body.push(para("The data is not in git. Where each input lives is in handover/explainability/README.md. The commands, in order:"));
body.push(bullet("Explanation: xgb83_explain.py (train, explain, report), feature_curves.py, feature_curve_figs.py, xgb83_local.py, xgb57_explain.py, xgb83_calibration.py."));
body.push(bullet("Ablations (WSL, GPU): xgb83_drop_ablation.py, which now also fits no_position and checklist_77."));
body.push(bullet("GDT: xgb83_gdt.py, then xgb83_ablation_report.py, then gdt_examples.py."));
body.push(bullet("Figures and report: readme_figs.py, then prep_docx_data.py, then build_docx.js."));

const doc = new Document({
  creator: "FPS Critic, Inc.",
  title: "Which inputs carry a CS2 round win-probability model",
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
