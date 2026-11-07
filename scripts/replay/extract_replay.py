"""Build a replay JSON (schema v1, see build_replay_page.py) for ONE real match from the raw per-tick channels and the win-probability predictions.

  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\replay\\extract_replay.py --match-key csds/2026/09/03/<id>/csds [--pred <predictions.parquet>] [--out notes\\replay\\replay_<id8>.json]

Reads (nothing is written under E:):
  E:\\csds_final\\csds\\<yyyy>\\<mm>\\<dd>\\<id>\\{player_vector, player_status, player_death, bomb_action, bomb_defuse}   (64 ticks per second)
  data/combined/{rounds,matches}.parquet, the predictions parquet (default: the held-out v2 predictions pulled from the pod), notes/replay/map_meta.json
Per used round: frames at 8 Hz from the live tick (freeze end) to the effective end; the 10 players are slotted by side (CT 0-4, T 5-9, ordered by
player id, so the same person can change slot at the half); names are redacted in the source, so labels are CT1..T5. The curve is the model's P(CT wins)
at every second it scored (the match must come from the held-out test split, otherwise the page would show training matches). Position yaw is the raw
phi_ang. A dead player keeps the last position with hp 0. Bomb events: plant (planter), defuse, explode.
Rounds that are not 5v5 at the live tick, have no predictions, or are not 'ok' in rounds.parquet are skipped and listed in match.skipped.
"""
import argparse
import json
import os
import sys

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
E_CSDS = r"E:\csds_final\csds"
DEFAULT_PRED = r"G:\pureskill\csds-cs2\data\combined\export_winprob_v3_test_predictions.parquet"
HZ = 8
TICKS_PER_FRAME = 64 // HZ
# observed (code -> name) pairs of notes/schema_dictionary.md plus the standard CS2 definition indexes for guns that were not fired in that sample
NAMES = {1: "deagle", 2: "elite", 3: "fiveseven", 4: "glock", 7: "ak47", 8: "aug", 9: "awp", 10: "famas", 11: "g3sg1", 13: "galilar", 14: "m249", 16: "m4a1",
         17: "mac10", 19: "p90", 23: "mp5sd", 24: "ump45", 25: "xm1014", 26: "bizon", 27: "mag7", 28: "negev", 29: "sawedoff", 30: "tec9", 32: "hkp2000",
         33: "mp7", 34: "mp9", 35: "nova", 36: "p250", 38: "scar20", 39: "sg556", 40: "ssg08", 60: "m4a1_silencer", 61: "usp_silencer", 63: "cz75a", 64: "revolver"}


def raw_dir(mk):
    p = mk.split("/")  # csds/2026/09/03/<id>/csds
    return os.path.join(E_CSDS, p[1], p[2], p[3], p[4])


def read(d, ch, cols):
    return pl.from_arrow(pq.read_table(os.path.join(d, ch), columns=cols))


def build(mk, pred_path, out_path):
    import re
    mlabel = (re.search(r"export_(winprob_v\d+)", os.path.basename(pred_path)) or [None, "winprob"])[1]
    d = raw_dir(mk)
    meta_all = json.load(open(os.path.join(ROOT, "notes", "replay", "map_meta.json"), encoding="utf-8"))
    mrow = pl.read_parquet(os.path.join(ROOT, "data", "combined", "matches.parquet")).filter(pl.col("match_key") == mk)
    if mrow.height != 1:
        sys.exit(f"match {mk} not found in matches.parquet")
    mrow = mrow.row(0, named=True)
    map_name = mrow["map_name"]
    if map_name not in meta_all:
        sys.exit(f"no radar metadata for {map_name}")
    mm = meta_all[map_name]
    pred = pl.read_parquet(pred_path, columns=["match_key", "round", "tick", "second", "p", "label_ct_win"]).filter(pl.col("match_key") == mk)
    if pred.height == 0:
        sys.exit("this match has no held-out predictions (pick a match from the test split)")
    rounds = pl.read_parquet(os.path.join(ROOT, "data", "combined", "rounds.parquet")).filter(pl.col("match_key") == mk).sort("round")

    # theta_ang is the horizontal view angle (degrees, 0 = +x, counter-clockwise); phi_ang is the polar angle (about 90 when level), NOT the heading
    vec = read(d, "player_vector", ["tick", "round", "player_id", "x_pos", "y_pos", "theta_ang", "team_code"]).unique(["round", "tick", "player_id"], keep="first")
    sta = read(d, "player_status", ["tick", "round", "player_id", "health", "armor", "money", "inv_primary", "inv_secondary", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade",
                                    "inv_molotov", "inv_incgrenade", "inv_decoy", "has_c4", "has_defuser", "has_helmet"]).unique(["round", "tick", "player_id"], keep="first")
    hurt = read(d, "player_hurt", ["round", "tick", "player_id", "attacker_id", "weapon_name", "effective_health_removed"])
    gstate = read(d, "grenade_state", ["round", "tick", "event_type", "entity_id", "x_pos", "y_pos"])
    mstate = read(d, "molotov_state", ["round", "tick", "event_type", "entity_id", "x_pos", "y_pos"])
    blind = read(d, "player_blind", ["round", "tick", "player_id", "blind_duration"])
    gvec = read(d, "grenade_vector", ["round", "tick", "entity_id", "grenade_type", "x_pos", "y_pos"])
    dth = read(d, "player_death", ["round", "tick", "player_id", "attacker_id", "weapon_name", "is_headshot"])
    bact = read(d, "bomb_action", ["round", "tick", "event_type", "player_id"])
    cur = vec.join(sta, on=["round", "tick", "player_id"], how="left")

    weapons = []

    def widx(code):
        if code is None or code <= 0:
            return -1
        name = NAMES.get(int(code), f"item{int(code)}")
        if name not in weapons:
            weapons.append(name)
        return weapons.index(name)

    out_rounds, skipped = [], []
    for r in rounds.iter_rows(named=True):
        rn = r["round"]
        pr = pred.filter(pl.col("round") == rn).sort("tick")
        if r["round_status"] != "ok" or not r["use"] or pr.height == 0:
            skipped.append({"round": rn, "why": "not used or no predictions"})
            continue
        t0 = int(pr["tick"][0])
        t_end = int(r["end_eff"])
        if t_end <= t0:
            skipped.append({"round": rn, "why": "empty live window"})
            continue
        nfr = (t_end - t0 + TICKS_PER_FRAME - 1) // TICKS_PER_FRAME + 1  # the last frame is at or after the effective end, so the decisive kill is shown
        ticks = [t0 + TICKS_PER_FRAME * k for k in range(nfr)]
        cr = cur.filter((pl.col("round") == rn) & (pl.col("tick") >= t0 - 64) & (pl.col("tick") <= t_end + 8))
        roster = cr.filter(pl.col("tick") <= t0 + 128).group_by("player_id").agg(pl.col("team_code").mode().first().alias("side")).sort("player_id")
        ct_ids = [x for x, s in roster.select("player_id", "side").iter_rows() if s == 3]
        t_ids = [x for x, s in roster.select("player_id", "side").iter_rows() if s == 2]
        if len(ct_ids) != 5 or len(t_ids) != 5:
            skipped.append({"round": rn, "why": f"not 5v5 at the live tick ({len(ct_ids)} CT, {len(t_ids)} T)"})
            continue
        ids = ct_ids + t_ids
        slot_of = {pid: i for i, pid in enumerate(ids)}
        deaths = {}
        for tk, vic in dth.filter((pl.col("round") == rn)).select("tick", "player_id").iter_rows():
            if vic in slot_of:
                deaths.setdefault(vic, tk)
        grid = pl.DataFrame({"tick": ticks}).cast({"tick": pl.Int64})
        players_frames = []
        for pid in ids:
            sub = cr.filter(pl.col("player_id") == pid).sort("tick").drop("round", "player_id", "team_code")
            j = grid.join_asof(sub, on="tick", strategy="backward")
            dt = deaths.get(pid)
            tl = j["tick"].to_list()
            alive = [0 if (dt is not None and tk >= dt) else 1 for tk in tl]
            x = j["x_pos"].fill_null(strategy="forward").fill_null(strategy="backward").to_list()
            y = j["y_pos"].fill_null(strategy="forward").fill_null(strategy="backward").to_list()
            yaw = j["theta_ang"].fill_null(strategy="forward").fill_null(strategy="backward").to_list()
            hp = j["health"].fill_null(0).to_list()
            gear = [0] * len(tl)
            for col, bit in (("inv_flashbang", 1), ("inv_smokegrenade", 2), ("inv_hegrenade", 4), ("inv_molotov", 8), ("inv_incgrenade", 8), ("inv_decoy", 16), ("has_c4", 32), ("has_defuser", 64), ("has_helmet", 128)):
                gear = [g | (bit if (v or 0) else 0) for g, v in zip(gear, j[col].to_list())]
            gear = [g if a else 0 for g, a in zip(gear, alive)]
            bev = [(int(tk), float(du)) for tk, du in blind.filter((pl.col("round") == rn) & (pl.col("player_id") == pid)).select("tick", "blind_duration").iter_rows()]
            brem, btot = [], []
            for tk, a in zip(tl, alive):
                best = (0.0, 0.0)
                for te, du in bev:
                    if te <= tk:
                        rem = (te + du * 64 - tk) / 64
                        if rem > best[0]:
                            best = (rem, du)
                brem.append(round(best[0], 2) if a else 0)
                btot.append(round(best[1], 2) if a and best[0] > 0 else 0)
            ar = j["armor"].fill_null(0).to_list()
            mo = j["money"].fill_null(strategy="forward").fill_null(0).to_list()
            pri = j["inv_primary"].to_list()
            sec = j["inv_secondary"].to_list()
            hp = [h if a else 0 for h, a in zip(hp, alive)]
            ar = [v if a else 0 for v, a in zip(ar, alive)]
            wp = [widx(p if (p or 0) > 0 else s) if a else -1 for p, s, a in zip(pri, sec, alive)]
            players_frames.append({"x": [round(float(v), 1) for v in x], "y": [round(float(v), 1) for v in y], "yaw": [round(float(v), 1) for v in yaw],
                                   "hp": [int(v) for v in hp], "armor": [int(v) for v in ar], "alive": alive, "weapon": wp, "money": [int(v) for v in mo], "gear": gear, "blind": brem, "blindd": btot})
        events = []
        for tk, vic, att, wn, hs in dth.filter(pl.col("round") == rn).sort("tick").select("tick", "player_id", "attacker_id", "weapon_name", "is_headshot").iter_rows():
            if tk < t0 or tk > ticks[-1] or vic not in slot_of:  # deaths after the round is decided (post-explosion, round-end) are not shown
                continue
            events.append({"t": round((tk - t0) / 64, 2), "kind": "kill", "killer": slot_of.get(att, slot_of[vic]), "victim": slot_of[vic], "weapon": wn or "world", "hs": bool(hs)})
        for tk, vic, att, wn, dmg in hurt.filter(pl.col("round") == rn).sort("tick").select("tick", "player_id", "attacker_id", "weapon_name", "effective_health_removed").iter_rows():
            if tk < t0 or tk > ticks[-1] or vic not in slot_of or att not in slot_of or att == vic or not dmg:
                continue
            events.append({"t": round((tk - t0) / 64, 2), "kind": "hurt", "attacker": slot_of[att], "victim": slot_of[vic], "dmg": int(dmg), "weapon": wn or "?"})
        utility = []
        rel = lambda tk: round((tk - t0) / 64, 2)
        gs = gstate.filter(pl.col("round") == rn)
        for ent, tk, x, y in gs.filter(pl.col("event_type") == "smokegrenade_detonate").select("entity_id", "tick", "x_pos", "y_pos").iter_rows():
            ex = gs.filter((pl.col("entity_id") == ent) & (pl.col("event_type") == "smokegrenade_expired"))
            t1 = int(ex["tick"][0]) if ex.height else tk + 64 * 18
            if t1 >= t0 and tk <= ticks[-1]:
                utility.append({"kind": "smoke", "x": round(x, 1), "y": round(y, 1), "t0": rel(tk), "t1": rel(t1)})
        for tk, et, x, y in gs.filter(pl.col("event_type").is_in(["hegrenade_detonate", "flashbang_detonate", "decoy_detonate"])).select("tick", "event_type", "x_pos", "y_pos").iter_rows():
            if t0 <= tk <= ticks[-1]:
                utility.append({"kind": "burst", "type": et.split("_")[0].replace("hegrenade", "he").replace("flashbang", "flash"), "x": round(x, 1), "y": round(y, 1), "t0": rel(tk)})
        ms = mstate.filter(pl.col("round") == rn)
        for ent, tk, x, y in ms.filter(pl.col("event_type") == "inferno_startburn").select("entity_id", "tick", "x_pos", "y_pos").iter_rows():
            ex = ms.filter((pl.col("entity_id") == ent) & (pl.col("event_type") == "inferno_expire"))
            t1 = int(ex["tick"][0]) if ex.height else tk + 64 * 7
            if t1 >= t0 and tk <= ticks[-1]:
                utility.append({"kind": "fire", "x": round(x, 1), "y": round(y, 1), "t0": rel(tk), "t1": rel(t1)})
        gv = gvec.filter((pl.col("round") == rn) & (pl.col("tick") >= t0 - 64) & (pl.col("tick") <= ticks[-1])).sort("entity_id", "tick")
        for (ent,), g1 in gv.group_by("entity_id", maintain_order=True):
            g1 = g1.filter((pl.col("tick") - pl.col("tick").min()) % 4 == 0)
            if g1.height < 2:
                continue
            typ = {"Flashbang": "flash", "Smoke Grenade": "smoke", "HE Grenade": "he", "Molotov": "fire", "Incendiary Grenade": "fire", "Decoy Grenade": "decoy"}.get(g1["grenade_type"][0], "other")
            utility.append({"kind": "nade", "type": typ, "path": [[rel(tk), round(x, 1), round(y, 1)] for tk, x, y in g1.select("tick", "x_pos", "y_pos").iter_rows()]})
        pt = r["plant_tick"]
        if pt is not None and pt >= t0:
            planter = bact.filter((pl.col("round") == rn) & (pl.col("event_type") == "bomb_begin_plant") & (pl.col("tick") <= pt)).sort("tick")
            pid = int(planter["player_id"][-1]) if planter.height else None
            if pid in slot_of:
                events.append({"t": round((pt - t0) / 64, 2), "kind": "plant", "player": slot_of[pid]})
        dft = r["defuse_tick"]
        if dft is not None and dft >= t0:
            bd = bact.filter(pl.col("round") == rn).head(0)  # placeholder to keep the type; the defuser comes from bomb_defuse below
            del bd
            dd = read(d, "bomb_defuse", ["round", "tick", "event_type", "player_id"]).filter((pl.col("round") == rn) & (pl.col("tick") <= dft)).sort("tick")
            pid = int(dd["player_id"][-1]) if dd.height else None
            if pid in slot_of:
                events.append({"t": round((dft - t0) / 64, 2), "kind": "defuse", "player": slot_of[pid]})
        ext = r["explode_tick"]
        if ext is not None and ext >= t0:
            events.append({"t": round((ext - t0) / 64, 2), "kind": "explode"})
        events.sort(key=lambda e: e["t"])
        winner = "CT" if int(pr["label_ct_win"][0]) == 1 else "T"
        out_rounds.append({
            "n": int(rn), "winner": winner, "duration": round((ticks[-1] - t0) / 64, 2),
            "players": [{"slot": i, "label": ("CT%d" % (i + 1)) if i < 5 else ("T%d" % (i - 4)), "side": "CT" if i < 5 else "T"} for i in range(10)],
            "frames": {"dt": 1.0 / HZ, "n": nfr, "p": players_frames},
            "events": events, "utility": utility,
            "prob": {"t": [float(v) for v in pr["second"].to_list()], "p": [round(float(v), 4) for v in pr["p"].to_list()]},
        })
        print(f"round {rn}: {nfr} frames, {len(events)} events, winner {winner}", flush=True)
    if not out_rounds:
        sys.exit("no usable round")
    ct_wins = sum(1 for x in out_rounds if x["winner"] == "CT")
    p_all = pred["p"].to_numpy().astype("float64")
    y_all = pred["label_ct_win"].to_numpy().astype("float64")
    import numpy as np
    ll = float(-np.mean(y_all * np.log(np.clip(p_all, 1e-6, 1)) + (1 - y_all) * np.log(np.clip(1 - p_all, 1e-6, 1))))
    doc = {
        "version": 1,
        "match": {"id": mk.split("/")[4][:8], "map": map_name, "ct_score": int(mrow["ct_starters_score_final"]), "t_score": int(mrow["t_starters_score_final"]),
                  "n_rounds": len(out_rounds), "model": mlabel + " (held-out match)", "scores_are": "by starting side", "model_logloss_this_match": round(ll, 4),
                  "skipped": skipped, "ct_round_wins_shown": ct_wins},
        "map_meta": {"map": map_name, "pos_x": mm["pos_x"], "pos_y": mm["pos_y"], "scale": mm["scale"], "size": mm["size"]},
        "weapons": weapons,
        "model_notes": {"model": mlabel + ": boosted trees on one-second snapshots (held-out match).",
                        "used": ["players alive per side", "round clock", "bomb planted and time since the plant", "map",
                                 "per-side totals: health, armor, helmets, defuse kits, rifle / AK / M4 / AWP / SMG counts, grenades held, equipment value"],
                        "not_used": ["player positions and rotations", "which way anyone is facing", "smokes, fire and grenades in flight", "who is blinded or for how long", "who damaged whom", "distance to the bomb"]},
        "rounds": out_rounds,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, separators=(",", ":"), allow_nan=False)
    print(f"-> {out_path}  {os.path.getsize(out_path) / 1e6:.2f} MB  rounds {len(out_rounds)}  skipped {len(skipped)}  match log loss {ll:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-key", required=True)
    ap.add_argument("--pred", default=DEFAULT_PRED)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "notes", "replay", "replay_" + a.match_key.split("/")[4][:8] + ".json")
    build(a.match_key, a.pred, out)
