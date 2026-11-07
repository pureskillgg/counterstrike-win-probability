import gzip, json, sys
PER_TICK = {"player_vector", "player_status", "tick", "grenade_vector", "world_item_vector"}
raw = open(sys.argv[1], "rb").read()
if raw[:2] == b"\x1f\x8b":
    raw = gzip.decompress(raw)
chans = [c["channel"] for c in json.loads(raw)["channels"]]
keep = [c for c in chans if c != "header" and c not in PER_TICK]
print(f"declared={len(chans)} per_match_non_tick={len(keep)}", file=sys.stderr)
print("\n".join(keep))
