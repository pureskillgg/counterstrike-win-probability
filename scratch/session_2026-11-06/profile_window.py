"""Count-only profile of identifier-like columns in one raw player channel file (training-era window).
Prints counts, pattern classes and frequency shapes only. No values are printed.
"""
import glob, json, re, subprocess, sys
from collections import Counter
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
BASE = r"G:/pureskill/csds-cs2/evac_2026-10-13/tomes/2024-02-g/tome/csds"
MATCH_RE = re.compile(r"^csds/\d{4}/\d{2}/\d{2}/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/csds$")
STEAM64_RE = re.compile(r"^7656119\d{10}$")
def profile(label, vals):
    nonempty = [v for v in vals if v not in (None, "")]
    c = Counter(str(v) for v in nonempty)
    pats = {
        "steam64_pattern": sum(bool(STEAM64_RE.match(str(v))) for v in nonempty),
        "hex_hash_32_64": sum(bool(re.fullmatch(r"[0-9a-f]{32,64}", str(v))) for v in nonempty),
        "redact_anon_unknown": sum(bool(re.search(r"redact|anon|unknown|placeholder", str(v), re.I)) for v in nonempty),
        "digits_only": sum(str(v).isdigit() for v in nonempty),
        "all_zero": sum(bool(re.fullmatch(r"0+", str(v))) for v in nonempty),
    }
    freq = sorted(c.values(), reverse=True)
    print(f"{label}: rows={len(vals)} nonempty={len(nonempty)} distinct={len(c)} patterns={pats}")
    print(f"    top-10 value frequencies (counts only): {freq[:10]}; values seen once: {sum(1 for v in c.values() if v == 1)}")
for ch, cols in [("basic_player_personal", ["match_key", "steam_id", "name", "clan_tag", "player_id", "is_bot"]),
                 ("basic_player_name", ["match_key", "player_id", "name_new", "name_old"])]:
    files = sorted(glob.glob(f"{BASE}/{ch}.*/dataframe_*"))
    print(f"== {ch}: files={len(files)}")
    t = pq.read_table(files[0], columns=cols)
    print(f"   rows={t.num_rows}")
    for c in cols:
        if c == "match_key":
            mk = t.column(c).to_pylist()
            print(f"   match keys: distinct={len(set(mk))} format_ok_share={round(sum(bool(MATCH_RE.match(str(k))) for k in mk)/len(mk),4)}")
            continue
        profile(f"   {c}", t.column(c).to_pylist())
    # overlap with export key list (counts only)
    raw = subprocess.run(["aws","--profile","runpod","--endpoint-url","https://s3api-eu-ro-1.runpod.io","--region","eu-ro-1","s3","cp","s3://z7yvc40ykx/export_clean_v5/match_keys.json","-"], capture_output=True, check=True).stdout
    exp = set(json.loads(raw))
    rk = set(t.column("match_key").to_pylist())
    print(f"   raw distinct match keys in export_clean_v5 key list: {len(rk & exp)} of {len(rk)}")
