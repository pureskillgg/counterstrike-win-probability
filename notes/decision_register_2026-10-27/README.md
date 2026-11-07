# Decision and test register (27 Oct 2026)

Built from the records, not from memory. Three files:

- `pins_register.md`: all 79 pinned decisions in Meridian (project fa6f6e09), in time order (Central time). Each row gives the kind (pre-registration, outcome, correction, decision, data record), who decided, and the verdict where there is one.
- `unpinned_results.md`: 124 JSON result files under `data/combined` and `data/diagnostics` that have no pin in Meridian. Some are experiments and some are data checks. This list is not yet sorted into experiments and checks.
- `papers_status.md`: the paper reading list, and what was tested from each paper.

## Who decided

- "owner: ..." means the pin records your approval, with your words where the pin quotes them.
- "assistant proposal (no owner approval pinned)" means I proposed it and the pin records no approval from you. Most of the tests (T0 to T7, C1, G1, W2, B1, the learning curve) are in this group. Your general instruction to test ideas is in our conversation, not in the pins.
- Pins that are gates, not approvals: 56e6e642 (candidate B locked read) needs the audit reviewed and your go before the read. It has not been given.

## Owner approvals recorded in pins (16 pins, some still open)

- a7b482f6 (2024 kept separate), 5a1dec05 (round-state window), 6923f58c (roster rule for 2024): "ok sure".
- 81e18a80: approved the 2024 exclusion rule.
- afa62544: approved the 2024 data rules.
- c432a055: approved the awpy navigation-mesh download (internal research use only).
- da8ee067: chose to run T3 as pinned, locally.
- 5f8dea45 and b2f4a9d7: the 22 to 25 October extension and its cut-off.
- 56e6e642: chose candidate B. The read itself is still gated.
- 9ec38dc9: confirmed the 7 maps.
- ab37b667: score lead is not a feature.
- 80ba0fc6: download approvals (14 Oct).
- 2569aac7: suggested 80/5/15 for the split. Open.
- 5138cc29: asked for the open decisions to be handled rather than put to you.
- 2702982d: merge-check decision requested. Resolved by 5138cc29.

Notes on the pins above: 2702982d is a request, resolved by 5138cc29. 56e6e642 is a gate, not an approval: it records the choice of candidate B, and the read still needs the audit reviewed and your go. 2569aac7 is open.

## Papers

See `papers_status.md`. Of the 27 numbered papers, 4 were tested directly (1, 2, 17 partly, 28), 2 are related to runs that have no pin (6, 7), and 21 were not tested.

## Results without a pin

124 result files exist with no pin. Some of them ran on the test set. Two examples: `exp_full_comparison.json` and `exp_model_families.json` both report on the all-maps test of 3,059 matches (exp_full_comparison verified; exp_model_families: see exp_model_families_check.md). Each needs a decision: pin it, retire it, or label it as descriptive. This is part of the test-set audit.

## Open items waiting on you

1. The candidate B locked read: needs your go after the audit review. Holdout B (653 matches, all maps) is built and its features are verified; it has not been scored. No read has run.
2. Whether 2024 goes into any pre-registered test. Recommended: not for the final read.
3. The split ratio for new matches: 72/8/20 stays unless you change it (open since 23 Oct).
4. The unconfirmed citations: Huang et al. 2026 (not found online), Chahrour 2023 (repo exists, no paper found). The source confirmations are in Meridian (note 45c53f08).
5. Which of the 124 result files to pin, retire, or label as descriptive.
