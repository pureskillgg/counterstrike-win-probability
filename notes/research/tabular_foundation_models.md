# Tabular foundation models on CPU: feasibility for CS2 round win probability

Date: 2026-10-12. Scope: ~100k to 1M training rows, ~40 features, CPU only, no package installs, no changes to tomes or E:\.

Method: primary sources only (GitHub and Hugging Face raw files, docs.priorlabs.ai Models page, arXiv full text). Search summaries were used only to locate sources, and anything they alone support is marked UNVERIFIED. Full citations for every source are in `tabular_foundation_models.bib`.

Quoting: this note paraphrases source text and gives locators, because verbatim excerpts are limited to one short quote per response. The raw fetched files are in the session scratchpad (`C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\7308a8e7-a47e-4034-b140-5c511acb0aa4\scratchpad\tabpfn\`) for line-level checking.

---

## Bottom line (Q4)

**No, not on the documented evidence.** No model here has a published CPU run at roughly 300k rows, and no vendor documents CPU inference at that size.

- The TabPFN family is capped on CPU (1,000 samples for v2 to v2.6, 5,000 for v3 and later), and every version from v2.5 onward has non-commercial weights.
- TabICLv2 (BSD-3-Clause) and TabDPT (Apache-2.0) are the two permissively licensed candidates with a plausible CPU path. Their CPU timings at this size are UNVERIFIED and need a measurement on the target machine.

---

## Q1. TabPFN versions: row, feature and CPU limits

Source of the row and column numbers: docs.priorlabs.ai/models (Models table). CPU caps: GitHub PriorLabs/TabPFN README, GPU Recommended note and the `TABPFN_ALLOW_CPU_LARGE_DATASET` entry.

| Version | Documented max training rows | Documented max columns | CPU support (documented) | Source |
|---|---|---|---|---|
| TabPFN v2 (TabPFNv2) | 10,000 | 500 | CPU cap 1,000 samples (README, older versions) | docs Models; README |
| TabPFN-2.5 (Base, Plus) | 100,000 (report design target: 50,000, not strict) | 2,000 | CPU cap 1,000 samples; report says CPU suits only small datasets (sec. 5) | docs Models; arXiv 2511.08667 |
| TabPFN-2.6 | 100,000 | 2,000 | CPU cap 1,000 samples (README, older versions) | docs Models; README |
| TabPFN-3 | 1,000,000 | 2,000 (report validates 200 features at 1M rows; 2,000 at 100k rows) | CPU cap 5,000 samples (README). Report gives no CPU runtime for the full model; only distilled MLP or tree artifacts are described as running on CPU | docs Models; arXiv 2605.13986 (abstract, Fig. 4); README |
| TabPFN-3.5 (default), 3.5-Fast (alpha) | 1,000,000 | 20,000 (docs recommend up to 6,000) | CPU cap 5,000 samples (README). All published timings are GPU runs (RTX Pro 6000, plus 24 CPU cores on the node) | docs Models; arXiv 2609.17895; README |

Notes on the CPU figures:
- Caps are enforced by the package, not by the model. Setting `TABPFN_ALLOW_CPU_LARGE_DATASET` to true overrides them. The README warns that large datasets can still be slow on CPU. No CPU timing above the cap is published for any version.
- The README also says TabPFN is slow on CPU and recommends a GPU.
- The 1M-row figures are GPU results on one H100 (TabPFN-3 report) or the docs' limits, which do not state the hardware.

## Conflicts resolved

1. **1,000 vs 5,000 CPU rows.** Both are correct for different versions. The README gives 5,000 for TabPFN-3, 3.5 and 3.5-Fast, and 1,000 for older versions (v2, 2.5, 2.6). They are not contradictory.
2. **10,000 samples and 500 features.** This is the TabPFN v2 maximum (docs Models row; TabPFN-3 report Fig. 4). It is not a CPU limit.
3. **TabPFN-2.5 rows: 50,000 vs 100,000.** The 2.5 report says 50,000 is its design target and not strict, with benchmarks up to 100,000 training samples. The docs list 100,000 as the maximum.
4. **TabPFN-3 features: 2,000 vs 200.** The docs list 2,000 columns as the maximum. The TabPFN-3 report's validated envelope is 200 features at 1M rows, and 2,000 features at 100k rows.

---

## Q2. Licence (commercial use)

| Version | Weights licence | Commercial or production use | Source |
|---|---|---|---|
| TabPFN v2 | Prior Labs License v1.1: an Apache-2.0 derivative with an added attribution clause (Paragraph 10) | Not excluded by the licence text (legal reading UNVERIFIED) | HF Prior-Labs/TabPFN-v2-clf LICENSE.txt; README |
| TabPFN-2.5 | TABPFN-2.5 License v1.0 (report) or v1.1 (docs); non-commercial | Not permitted for model, derivatives or outputs | arXiv 2511.08667 sec. 6; docs Models |
| TabPFN-2.6 | TABPFN-2.6 License v1.0; non-commercial per README | Not permitted; licence text NOT FETCHED | docs Models; README |
| TabPFN-3 | TABPFN-3.0 License v1.0; non-commercial | Not permitted for model, derivatives or outputs | arXiv 2605.13986 sec. 5 |
| TabPFN-3.5, 3.5-Fast | TABPFN-3.5 License v1.0; non-commercial and non-production | Not permitted. One short verbatim excerpt: "non-commercial and non-production use" | HF Prior-Labs/tabpfn_3_5 LICENSE; arXiv 2609.17895 sec. 4 |

Code licences: Apache-2.0 per the README, and per the docs for TabPFN-3.5. The docs model table gives a Prior Labs License code licence for TabPFN-2.5 and TabPFN-3, which conflicts with the README. The weights licence governs commercial use either way.

Output and distillation: the docs FAQ states that using TabPFN or its outputs to train, fine-tune or distill a commercial model is not allowed without a commercial agreement. This matters for the CPU workaround of distilling to an MLP or tree ensemble.

Practical reading: only TabPFN v2 has weights whose licence text does not bar commercial use. Every later TabPFN version needs a commercial agreement for production.

---

## Q3. Other tabular foundation models (2025 to 2026)

| Model | Version | Documented row limit | Feature limit | CPU support (documented) | Licence (weights / code) | Source | Evidence (paraphrase; locator) |
|---|---|---|---|---|---|---|---|
| TabICL | v1 (ICML 2025, classification) | Pretrained on up to 60K samples; claims up to 500K on modest resources (abstract) | Up to 500 | Auto device selection includes CPU; CPU and disk offloading of activations. Memory example is 100K x 500 on GPU with 5 GB VRAM and 32 GB RAM. No CPU runtime | BSD-3-Clause (repo LICENSE) | arXiv 2502.05564; github.com/soda-inria/tabicl | Abstract; contributions list; repo LICENSE |
| TabICLv2 | v2 (arXiv 2602.11139, Feb 2026) | Validated 300 to 100,000 training samples; README says 500K via CPU and disk offload, with accuracy possibly degrading | Up to 2,000 validated; degrades well beyond 100 features | README recommends a GPU for larger datasets. Paper: 1M x 500 run takes 115 s on H100 with 250 GB CPU RAM; disk offload takes 450 s with under 24 GB CPU RAM. CPU-only time NOT STATED | BSD-3-Clause (repo) | arXiv 2602.11139; github.com/soda-inria/tabicl | README speed and scalability sections; paper appendix H |
| LimiX | LimiX-2 (Sep 2026); LimiX-2M and LimiX-16M (v1 family) | NOT STATED. LimiX v1 benchmark excluded datasets with over 50K training samples (benchmark filter, not a model limit) | NOT STATED (benchmark filter excluded over 10K features) | CLI accepts `--device cpu`. CPU timing NOT STATED | LimiX-2 weights: StableAI LimiX Non-Commercial License (no commercial rights). Code: StableAI licence derived from Apache-2.0 | github.com/limix-ldm/LimiX; HF stableai-org/LimiX-2 LICENSE | README news and CLI table; LICENSE notice |
| TabDPT | v1.3 (Sep 2026); v1.2 = TabDPT-Turbo (Jun 2026) | NOT STATED. Inference retrieves top-K training rows as context (paper K = 2,048) | NOT STATED | NOT STATED. Paper runs all training and inference on A100 GPUs | Code Apache-2.0 (LICENSE). Weights apache-2.0 (HF card front matter) | github.com/layer6ai-labs/TabDPT-inference; HF Layer6/TabDPT; arXiv 2410.18164 | Paper sec. 3.3 (retrieval); README; LICENSE |
| Mitra | v1 (AWS AutoGluon) | Pretrained on up to 640 rows. AutoGluon default `max_rows` = 10,000 (enforcement semantics UNVERIFIED) | Pretrained on up to 16 features. AutoGluon default `max_features` = 500 | AutoGluon says GPU is strongly recommended because in-context inference is 12 to 63 times slower on CPU | Apache-2.0 | auto.gluon.ai mitra_model source; HF autogluon/mitra-classifier; arXiv 2510.21204 | Source docstring and defaults; HF front matter |
| Mitra | v2 (arXiv 2609.04540) | In-context support set capped at 32,768 rows at prediction | NOT STATED | NOT STATED | Apache-2.0 (weights, code and evaluation results) | arXiv 2609.04540 | Release statement; support-cap statements |
| TabFM (Google Research) | v1.0.0 (2026) | Default `max_num_rows` = 100 context rows; larger tables are sampled, not consumed whole | Default `max_num_features` = 500 | JAX (CPU) and PyTorch (CPU or GPU) install paths. CPU runtime NOT STATED | Code Apache-2.0. Default weights `tabfm-non-commercial-v1.0`: non-commercial, non-production | github.com/google-research/tabfm; Google Research blog | README licence notice and FAQ |

Notes:
- TabFM's 10-class limit appeared only in a search summary. UNVERIFIED.
- Not checked in this pass: Orion-MSP, TabSTAR, EXAONE Tabular, Xiaomi-TabLDM, Causilo. The LimiX-2 report names some of these.

---

## Q4. Can any of these fit about 300k rows on CPU in reasonable time?

**No, not demonstrated.**

1. **TabPFN family (all versions).** The package caps CPU runs at 1,000 or 5,000 samples. The override exists, but the README warns it can still be slow. No CPU timing above the cap is published. Weights from v2.5 onward are non-commercial. Fails on evidence and on licence for production.
2. **TabICLv2 (BSD-3).** Documented scale is 100K training rows and 2,000 features, with 500K reachable through CPU and disk offload. The only CPU-specific figure in the paper is a relative speedup over TabPFN-2.5 (11.8 times at 10K rows). The 1M x 500 timings are on a GPU. No absolute CPU time at 300k. UNVERIFIED. This is the most plausible CPU candidate on licence.
3. **TabDPT (Apache-2.0).** Each query uses a retrieved context of a fixed size (paper: 2,048 rows), so the per-query model cost should not grow with the full training table. Retrieval over 300k rows is an extra cost. No CPU measurement is published. UNVERIFIED. Plausible candidate.
4. **Mitra v1.** AutoGluon itself says CPU in-context inference is 12 to 63 times slower, and its default row limit is 10,000. Not suitable at 300k without a benchmark.
5. **TabFM and LimiX-2.** CPU backends exist, but the default weights are non-commercial. TabFM's default context is 100 rows. No CPU timing is published.

**Next step (not done here, since packages must not be installed):** time TabICLv2 and TabDPT on a 300k-row subsample on the target CPU, and record fit and predict wall-clock time and peak RAM. Until that is done, any claim that 300k rows is feasible on CPU is UNVERIFIED. If the work is commercial or production, the TabPFN v2.5-and-later and LimiX-2 and TabFM default weights are excluded regardless of speed.

---

## UNVERIFIED items

- Any CPU runtime for TabPFN above its sample cap, for any version.
- Absolute CPU runtime for TabICLv2 or TabICL at about 300k rows.
- Whether TabDPT's inference code runs on CPU, and its speed there.
- Mitra's CPU limits beyond the AutoGluon defaults, and how AutoGluon enforces `max_rows`.
- LimiX-2 row and feature limits (none found in README or the LimiX-2 report).
- TabFM: the 10-class limit (search summary only), and the Hugging Face weights card (fetch returned an authentication error).
- The TabPFN-2.6 licence text (not fetched).
- Full author lists for several papers. Author names in the bib are partial where the full list was not retrieved.
- Mitra v1 arXiv submission date and venue (from search metadata).
- TabICL v1 submission date (from search metadata).
- Conflicts: TabPFN-2.5 licence version (v1.0 in report, v1.1 in docs); TabPFN-2.5 and TabPFN-3 code licence (Apache-2.0 in README, Prior Labs License in docs); a search snippet reported a non-Apache licence tag for the Layer6 organisation page, while the raw model card says apache-2.0; an older search claimed all LimiX models are Apache-2.0, which the current README contradicts for LimiX-2.
- The meridian MCP servers (code, debug, extractor, fs, outputs) reported that they need OAuth authorization. They were not used here.
