DRAFT, NOT SENT. For the user to review and send.

Subject: CS2 win-probability: final candidate and status

Dr Freeman,

Following the owner ruling of 3 November, the final candidate is the 83-input XGBoost model (59 base inputs, geometry 16, view 4, blindness 4). It was chosen on validation: log loss 0.450252 on 1,168 validation matches (seed mean), and GDT 0.0254 (95% interval 0.0218 to 0.0292) on the 297-match validation population. Training is on log loss; GDT is used for selection and reporting.

Prior reads, disclosed:
- The 83-input model was scored once on the Oct 26 to 31 holdout on 3 November (seed-mean log loss 0.452295, 3,315 matches). That was sent before a written go from you; the user had approved the scoring in chat. It is reported as a one-time event and was not used to choose anything.
- The 57-input model's test log loss (0.456411) was reported in the 20 October package, and its selection used the test split. Disclosed.
- Holdout B (653 matches) is contaminated and is excluded from all clean claims.

Since the last update:
- A 77-input variant (dropping the armour sums and the view features) is 0.00031 worse in log loss than the 83 on validation, with an interval excluding zero; GDT difference is not significant.
- A defuse-margin feature added no benefit on validation and was dropped.
- Positions every second: the bucket matches are built on all seconds; laptop-sourced matches are [status to be confirmed].
- Freeze-second loadout side defect, fixed in the loadout export (v4): 283 rounds in 23 matches had a duplicated player id in the roster and one alive player missing. The side was taken from the player's position at freeze, and every fixed round now has 5 CT and 5 T alive. Validated on kill sides: 99.65% of kills are between opposite sides, against 86.2% under the old roster. It does not affect the 83 model, which was trained on the sprint matrix. One round (2d963086, round 18) is listed but not patched. A check on per-second steps outside these 23 matches is still open.
- 2,848 train and validation matches dated 21 to 25 October are in the export and the loadout build; this follows the confirmed decision by the user.

Pins for each decision are in Meridian. Code, docs and the final model file are on the wrap-up-nov-2026 branch of the repository (commit 3936180).

Happy to walk through any of this before Friday.

Adam
