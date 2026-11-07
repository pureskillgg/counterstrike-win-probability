# Scratch

Working scripts from the analysis sessions, kept because they show how numbers in the reports were produced. They are not polished and some are one-off checks.

- `session_2026-11-06/`: scripts from the 5 to 6 November sessions. That covers pod management for the sequence runs, the GDT scoring, the regularisation test, the segment check, the weights backup and handover, the PII and split audits, the data-day manifests, and the deck builder.

Nothing here holds data or credentials. Scripts that touch S3 or RunPod read credentials from local files outside the repo (for example `~/.aws/runpod_s3_credentials`), which are not in git.

Many scripts have hard-coded local paths (`C:/Users/...`, `G:/pureskill/...`). Change those to your own paths before running anything.
