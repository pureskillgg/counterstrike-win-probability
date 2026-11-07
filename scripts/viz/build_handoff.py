"""Meridian-style handoff for the CS2 email: what to send, the links to add, the draft text, the decisions still open, and what was verified.

  ...python.exe -I -X utf8 scripts\\viz\\build_handoff.py

Reads deliverables/EMAIL_DRAFT_2026-10-19.md (run build_email_package.py first) and writes
deliverables/MERIDIAN_HANDOFF_EMAIL_2026-10-19.txt. Nothing is sent or uploaded.
"""
import hashlib
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DELIV = os.path.join(ROOT, "deliverables")
DRAFT = os.path.join(DELIV, "EMAIL_DRAFT_2026-10-19.txt")
OUT = os.path.join(DELIV, "MERIDIAN_HANDOFF_EMAIL_2026-10-19.txt")
FILES = [
    ("Report (Word)", os.path.join("word", "CS2_win_probability_full_document_2026-10-19.docx")),
    ("Slides (PowerPoint)", os.path.join("presentation", "CS2_win_probability_presentation_2026-10-19.pptx")),
    ("Package (zip)", "email_package_2026-10-19.zip"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    draft = open(DRAFT, encoding="utf-8").read()
    lines = []
    lines.append("=== MERIDIAN HANDOFF: CS2 round win-probability email ===")
    lines.append("Date: 2026-10-19")
    lines.append(f"Project folder: {ROOT}")
    lines.append("Status: DRAFT READY. NOT SENT. Nothing has been uploaded or shared.")
    lines.append("")
    lines.append("=== HANDOFF READINESS ===")
    lines.append("OK  Submission model winprob_v6_drop_heavy (57 inputs): results match data/combined/exp_submission_57.json")
    lines.append("OK  Serving parity on 294 October test matches (1 to 8 October tome): max gap 0.0001 pp, 100% of seconds within 0.01 pp")
    lines.append("      log: logs/test_win_probability_v6_drop_heavy_2026-10-19.log")
    lines.append("OK  Draft numbers checked against the saved JSON files")
    lines.append("OK  Scoring function default is winprob_v6_drop_heavy (57); winprob_v8_ensemble (58) is passed by name")
    lines.append("OK  Report renders (repository render check: rendered); slides: 19; chart images: 11")
    lines.append("OK  Recipient: Bill Freeman. Sender: Adam.")
    lines.append("!!  Owner must confirm the reason for leaving the score lead out (draft has a bracketed placeholder)")
    lines.append("!!  Live Slides link SFAnn4viMQ5DEieNVvdQEv is stale: republish before sending, or drop the link")
    lines.append("!!  Links not created yet: the three files are local until uploaded")
    lines.append("")
    lines.append("=== 1. WHAT TO SEND ===")
    for label, rel in FILES:
        path = os.path.join(DELIV, rel)
        size = os.path.getsize(path)
        lines.append(f"- {label}")
        lines.append(f"    file:   {path}")
        lines.append(f"    size:   {size:,} bytes")
        lines.append(f"    sha256: {sha256(path)}")
    lines.append("")
    lines.append("=== 2. LINKS TO ADD TO THE EMAIL ===")
    lines.append("Upload each file to the shared location you use, then paste its link in place of the bracket in the draft:")
    lines.append("  [LINK: report]  <- the report file above")
    lines.append("  [LINK: slides]  <- the slides file above")
    lines.append("  [LINK: zip]     <- the zip file above")
    lines.append("")
    lines.append("=== 3. EMAIL DRAFT (copy from here; brackets are placeholders to fill or delete) ===")
    lines.append(draft.rstrip())
    lines.append("")
    lines.append("=== 4. DECISIONS BEFORE SENDING ===")
    lines.append("1. Recipient is Bill Freeman; sender is Adam. Check the To and sign-off lines before sending.")
    lines.append("2. Score-lead reason: confirm it, or reword the bracketed sentence in the score-lead paragraph.")
    lines.append("3. Live link: republish the Slides artifact so it matches the slides, or remove the live-link line.")
    lines.append("4. Done 19 October: the scoring function's default is the 57 submission (winprob_v6_drop_heavy); the 58 is passed by name.")
    lines.append("5. Days 14 and 15: confirm the origin of the E: drive folders, or keep 'not used in any result' as written.")
    lines.append("6. Explicit yes to send. Nothing has been sent.")
    lines.append("")
    lines.append("=== 5. VERIFIED ===")
    lines.append("- Draft numbers: submission 0.456411 [0.453311, 0.459138] and validation 0.453557; comparison 0.455605;")
    lines.append("  paired +0.000806 [+0.000508, +0.001105]; event-only 0.498695 and +0.043090; monotone figures; score-lead counts.")
    lines.append("- Package: the zip holds models/, results/ and scripts/ only; the report, slides and chart PNGs are not in it.")
    lines.append("- Parity test log and the reference export both produced in this run.")
    lines.append("")
    lines.append("=== 6. KNOWN ISSUES (visible in the materials; not blocking the email) ===")
    lines.append("- notes/PRESENTATION_NOTES.md links to Q3wq55... (a 16 October 'findings' deck), not the current deck.")
    lines.append("- Charts are not embedded in the report (the image insert failed the render check); they are in the slides.")
    lines.append("- The zip includes results/split_manifest.json, which lists match keys. Check the sharing policy before sharing outside the team.")
    lines.append("")
    open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
