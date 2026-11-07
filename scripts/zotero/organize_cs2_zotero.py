"""Organize the Zotero collection "CS2 CSDS Round Win Probability" into subfolders and add the sources found by the research rounds.

  python -I -X utf8 scripts\\zotero\\organize_cs2_zotero.py            # dry run: prints the plan, writes nothing
  python -I -X utf8 scripts\\zotero\\organize_cs2_zotero.py --apply    # backs up, creates the subcollections, files items, adds the new sources
Needs Zotero 10 running and a write key the user granted in Zotero ($ZOTERO_LOCAL_KEY_FILE, see zotero_local.py). Safe to rerun: existing
subcollections are reused and sources already in the library (matched by arXiv id, DOI, URL or title) are never added twice.

Existing items are MOVED: removed from the parent collection and put in every subcollection their title matches (an item may sit in several);
their membership in any other collection (Masters, ...) is kept. The membership before the change is saved to notes/zotero/ first.
New sources come from the Haiku research rounds (workflow outputs): verdict confirmed or partly_confirmed only; every item carries tags saying
which round found it, how it was verified and how much of it was read, because most are agent summaries that a person has not read.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from zotero_local import Zotero

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
PARENT = "CS2 CSDS Round Win Probability"
SUBS = [
    "1 CS2 and esports win probability",
    "2 Sports win probability and critiques",
    "3 Calibration, scoring and evaluation",
    "4 Boosted trees, deep learning and tabular benchmarks",
    "5 Tabular foundation models and licences",
    "6 Interpretable and custom architectures",
    "7 CS2 CSDS data, spec and tooling",
    "8 Inspecting rounds and uncertainty",
    "9 Serving, testing and reliability",
]
# title rules; an item joins EVERY subcollection whose pattern matches its title (case-insensitive)
RULES = [
    (SUBS[4], r"tabpfn|tabicl|tabdpt|mitra|limix|tabfm|prior labs license|foundation model|in-context"),
    (SUBS[6], r"csds|csgo-dsdk|counter-strike 2 data|game state widget|datasheet|revision stats|getting the data|known flaws|data primer|enums|how the data fits|pop_overtime|\bdsdk\b|demoparser|awpy"),
    (SUBS[5], r"interpret|\bebm\b|explainable boosting|pygam|splinetransformer|monotonic|additive model|honor of kings|\bnam\b|neural additive|set transformer|deepsets|graph neural|entity"),
    (SUBS[3], r"xgboost|lightgbm|catboost|tree-based|boosted trees|closer look|neural nets outperform|tabarena|gradient boosting|talent|numpy 2\.0|tabm|realmlp|grinsztajn|deep learning on tabular|tabular data"),
    (SUBS[2], r"calibrat|boldness|scoring rule|obtaining|good probabilities|rethinking evaluation|brier|reliability diagram|proper scor|\bece\b|conformal|venn|forecast|probabilistic"),
    (SUBS[1], r"\bnfl\b|\bnba\b|soccer|basketball|nflwar|iwinrnfl|football|random forests to estimate|flaws of win probability|calibration-leverage|bayesian approach to in-game|win probability added|\bwpa\b|tennis|cricket|baseball|inpredictable|win probability"),
    (SUBS[0], r"counter-strike|cs:go|csgo|\bcs2\b|valorant|league of legends|e-?sports|\bmoba\b|leetify|riot games|\bestA\b|dota|win prediction|round (winner|outcome)|pandascore|starcraft|rainbow six|overwatch|apex"),
    (SUBS[7], r"checklist|slice finder|errudite|what-if tool|error analysis|behavio(u)?ral test|counterfactual|bootstrap|clustered|cluster|cross-validation|grouped|visuali|replay|reliability|uncertainty|quantile dot|groupkfold|thinning|model selection|approximate the use-case|prediction sets|selection bias"),
    (SUBS[8], r"treelite|onnx|lleaves|ml test score|latency|serving|inference|robust|fuzz|schema validation|hidden technical debt|production"),
]
TOPIC_TO_SUB = {  # fallback from the research topic when no title rule matches (the topic keys the three workflows used)
    "modern-models": SUBS[3], "missing-features": SUBS[0], "other-esports": SUBS[1], "wp-critiques": SUBS[2], "inspect-rounds": SUBS[7], "uncertainty": SUBS[2],
    "tabular-benchmarks": SUBS[3],
    "additive-interpretable": SUBS[5], "realtime-esports-nets": SUBS[0], "entity-set-models": SUBS[5], "hybrid-and-distill": SUBS[3], "cpu-serving": SUBS[8],
    "tabular-fm-core": SUBS[4], "latency-serving": SUBS[4], "licenses-weights": SUBS[4], "sequence-fm": SUBS[4], "zotero-library": None,
}
ROUNDS = [("round2", "wjl9rba4b", "wf_ca5e007b-4a2"), ("round3", "w1kvmo8kw", "wf_c455bed1-6e9"), ("round4", "w57clange", "wf_8e7723e4-4c1")]
TASKS = os.path.join(ROOT, "notes", "research", "raw")  # copies of the three workflow outputs (wjl9rba4b = round 2, w1kvmo8kw = round 3, w57clange = round 4)


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:120]


def ids_of(url, doi=""):
    """Identifiers used to recognise the same work: arXiv id, DOI, normalised URL."""
    out = set()
    u = (url or "").strip().lower()
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", u) or re.search(r"ar5iv\.(?:arxiv\.)?org/(?:html|abs)/(\d{4}\.\d{4,5})", u)
    if m:
        out.add("arxiv:" + m.group(1))
    m = re.search(r"(10\.\d{4,9}/[^\s?#]+)", u + " " + (doi or "").lower())
    if m:
        out.add("doi:" + m.group(1).rstrip("/.").lower())
    if u:
        out.add("url:" + re.sub(r"^https?://(www\.)?", "", u).split("#")[0].rstrip("/"))
    return out


def subs_for(title, topics=()):
    low = (title or "").lower()
    hit = [sub for sub, pat in RULES if re.search(pat, low)]
    if not hit:
        hit = [TOPIC_TO_SUB[t] for t in topics if TOPIC_TO_SUB.get(t)][:1]
    if SUBS[6] in hit and SUBS[0] in hit:  # a CSDS data/spec page is about the data, not an esports win-probability study
        hit.remove(SUBS[0])
    seen, out = set(), []
    for h in hit:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def load_sources():
    rows = []
    for tag, task, wf in ROUNDS:
        raw = open(os.path.join(TASKS, task + ".output"), encoding="utf-8", errors="replace").read()
        data, _ = json.JSONDecoder().raw_decode(raw[raw.find("{"):])
        for r in data["result"]["rows"]:
            c = r.get("check") or {}
            if c.get("verdict") in ("confirmed", "partly_confirmed"):
                rows.append({**r, "round": tag, "workflow": wf})
    return rows


def item_json(r):
    s, c = r["source"], r.get("check") or {}
    typ, url = s.get("type", "document"), s.get("url", "")
    arx = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url.lower())
    creators = []
    if s.get("authors_or_org"):
        creators = [{"creatorType": "author", "name": s["authors_or_org"][:200]}]
    year = s.get("year")
    note = f"Relevance: {s.get('relevance_to_us', '')}"
    if s.get("numbers"):
        note += f"\nReported numbers: {s['numbers']}"
    if s.get("quality_flags"):
        note += f"\nQuality flags: {s['quality_flags']}"
    note += f"\nFound by the Haiku research swarm {r['workflow']} ({r['round']}); re-checked by a second agent: {c.get('verdict')}. " + (f"Correction: {c['corrections']}" if c.get("corrections") else "")
    if arx:
        d = {"itemType": "preprint", "title": s["title"], "creators": creators, "date": str(year or ""), "url": url, "abstractNote": note, "repository": "arXiv", "archiveID": "arXiv:" + arx.group(1)}
    elif typ == "thesis":
        d = {"itemType": "thesis", "title": s["title"], "creators": creators, "date": str(year or ""), "url": url, "abstractNote": note}
    elif typ in ("paper", "preprint"):
        d = {"itemType": "journalArticle", "title": s["title"], "creators": creators, "date": str(year or ""), "url": url, "abstractNote": note}
    else:
        d = {"itemType": "webpage", "title": s["title"], "creators": creators, "date": str(year or ""), "url": url, "abstractNote": note, "websiteTitle": s.get("authors_or_org", "")[:120]}
    d["tags"] = [{"tag": t} for t in (f"swarm-{r['round']}", f"verify:{c.get('verdict')}", f"read:{s.get('read_depth', 'unknown')}", *[f"topic:{t}" for t in r.get("topics", [])])]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    z = Zotero()
    parent = z.find_collection(PARENT)
    if not parent:
        sys.exit(f"collection {PARENT!r} not found")
    mine = z.items(collection=parent)
    library = z.all_items(include_trash=True)  # trashed items count as known: a source the user deleted is not added again
    print(f"{len(mine)} items in {PARENT!r}; {len(library)} top-level items in the whole library (trash included)")

    # 1. where each existing item goes
    plan, unmatched = {}, []
    for it in mine:
        subs = subs_for(it["data"].get("title", ""))
        (plan.__setitem__(it["key"], subs) if subs else unmatched.append(it))
    print("existing items: matched", len(plan), "unmatched", len(unmatched))
    counts = {}
    for subs in plan.values():
        for sname in subs:
            counts[sname] = counts.get(sname, 0) + 1
    for sname in SUBS:
        print(f"  {sname}: {counts.get(sname, 0)} existing")
    for it in unmatched:
        print("  UNMATCHED (stays in the parent):", it["data"].get("title", "")[:90])

    # 2. new sources, de-duplicated against the whole library and against each other
    known = set()
    for it in library:
        d = it["data"]
        known |= ids_of(d.get("url", ""), d.get("DOI", ""))
        known.add("title:" + norm_title(d.get("title", "")))
    new_items, skipped_dup, skipped_nosub = [], 0, 0
    for r in load_sources():
        s = r["source"]
        keys = ids_of(s.get("url", "")) | {"title:" + norm_title(s.get("title", ""))}
        if any(k in known for k in keys):
            skipped_dup += 1
            continue
        subs = subs_for(s.get("title", ""), r.get("topics", []))
        if not subs:
            skipped_nosub += 1
            print("  no folder for:", s.get("title", "")[:90], r.get("topics"))
            continue
        known |= keys
        new_items.append((item_json(r), subs))
    ncount = {}
    for _, subs in new_items:
        for sname in subs:
            ncount[sname] = ncount.get(sname, 0) + 1
    print(f"new sources to add: {len(new_items)} (already in the library or repeated across the rounds: {skipped_dup}; no folder: {skipped_nosub})")
    for sname in SUBS:
        print(f"  {sname}: +{ncount.get(sname, 0)} new")
    if not a.apply:
        print("dry run: nothing written")
        return

    # 3. backup, then write
    os.makedirs(os.path.join(ROOT, "notes", "zotero"), exist_ok=True)
    if mine:  # a rerun after the items were moved has nothing in the parent to back up
        bk = os.path.join(ROOT, "notes", "zotero", f"backup_before_reorg_{time.strftime('%Y-%m-%d_%H%M%S')}.json")
        with open(bk, "x", encoding="utf-8") as fh:  # "x": never overwrite an earlier backup, it may be the only record of the original membership
            json.dump([{"key": i["key"], "version": i["version"], "title": i["data"].get("title"), "collections": i["data"].get("collections", [])} for i in mine], fh, indent=1)
        print("backup ->", bk)
    sub_key = {name: z.ensure_collection(name, parent) for name in SUBS}
    moved = 0
    for it in mine:
        subs = plan.get(it["key"])
        if not subs:
            continue
        cols = [c for c in it["data"].get("collections", []) if c != parent] + [sub_key[s] for s in subs]
        z.set_collections(it, list(dict.fromkeys(cols)))
        moved += 1
        if moved % 20 == 0:
            print("  moved", moved, flush=True)
    print("moved", moved, "existing items")
    added = 0
    for (d, subs) in new_items:
        keys = z.create_items([d], collection_keys=[sub_key[s] for s in subs])
        added += len(keys)
        if added % 25 == 0:
            print("  added", added, flush=True)
    print("added", added, "new items")
    cols = z.collections()
    for name in SUBS:
        print(f"  {name}: {cols[sub_key[name]]['num_items']} items")


if __name__ == "__main__":
    main()
