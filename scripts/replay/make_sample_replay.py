#!/usr/bin/env python3
"""Write a synthetic replay JSON (schema v1) for viewer development.

Nothing in the output comes from a real match. Ten players walk between
hand-picked waypoints on a made-up de_inferno-like layout, kills and a bomb
plant follow a fixed schedule, and the win-probability curve is a made-up
random walk that drifts toward the round winner.

The generator checks its own event logic (no dead player kills or plants,
weapons match the killer's loadout, bomb timer is 40 s) and fails with an
AssertionError if the schedule is ever edited into an impossible state.

Usage:
    python -I -X utf8 make_sample_replay.py [--out PATH] [--seed N]

Default output: notes/replay/replay_sample.json (under the project root).
Standard library only.
"""

import argparse
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = ROOT / "notes" / "replay" / "replay_sample.json"

DT = 0.125            # frames are 8 Hz
BOMB_SECONDS = 40.0   # plant -> explosion
WALK_SPEED = 230.0    # game units per second

MAP_META = {"map": "de_inferno", "pos_x": -2087.0, "pos_y": 3870.0, "scale": 4.9, "size": 1024}
WEAPONS = ["ak47", "m4a1", "awp", "glock", "usp_silencer", "deagle"]
PRICE = {"ak47": 2700, "m4a1": 3100, "awp": 4750, "glock": 0, "usp_silencer": 0, "deagle": 700}

# Fixed loadout per slot. Kill weapons in the schedule must match the killer's loadout.
LOADOUT = {0: "m4a1", 1: "awp", 2: "usp_silencer", 3: "ak47", 4: "m4a1",
           5: "ak47", 6: "deagle", 7: "ak47", 8: "m4a1", 9: "glock"}
LABELS = ["CT1", "CT2", "CT3", "CT4", "CT5", "T1", "T2", "T3", "T4", "T5"]

# Waypoints in game units. Synthetic positions, not map-accurate.
CT_SPAWN = (1500.0, 2300.0)
T_SPAWN = (-1200.0, 300.0)
MID = (300.0, 1200.0)
SITE_POINT = {"A": (1600.0, 1100.0), "B": (-300.0, 2000.0)}
CT_VIA = {"A": (1000.0, 1800.0), "B": (400.0, 2000.0)}
CT_HOLD = {
    "A": [(1300.0, 900.0), (1750.0, 1350.0), (1450.0, 1000.0), (1900.0, 1000.0), (1600.0, 700.0)],
    "B": [(-500.0, 1700.0), (-100.0, 1900.0), (-700.0, 1900.0), (50.0, 1650.0), (-300.0, 1500.0)],
}
T_VIA = {
    "A": [(-300.0, 800.0), MID, (1000.0, 1000.0)],
    "B": [(-600.0, 900.0), (-400.0, 1500.0)],
}

# Schedule. Kill tuple: (t, killer slot, victim slot, weapon, headshot).
# Plant tuple: (t, planter slot). End: ("defuse", t, slot) or ("explode", t).
ROUND_SPECS = [
    {
        "n": 1,
        "site": "A",
        "kills": [
            (12.4, 3, 7, "ak47", True),
            (25.8, 0, 5, "m4a1", False),
            (38.2, 9, 1, "glock", False),
            (55.0, 4, 8, "m4a1", True),
            (70.3, 2, 9, "usp_silencer", False),
        ],
        "plant": (41.0, 6),
        "end": ("defuse", 76.5, 2),
    },
    {
        "n": 2,
        "site": "B",
        "kills": [
            (9.6, 5, 0, "ak47", False),
            (20.1, 6, 3, "deagle", True),
            (33.7, 2, 8, "usp_silencer", False),
            (45.2, 9, 4, "glock", False),
            (66.4, 1, 6, "awp", True),
        ],
        "plant": (52.0, 7),
        "end": ("explode", 92.0),
    },
]


class Route:
    """Polyline walked at constant speed, then a small idle wobble."""

    def __init__(self, pts, start_delay, speed, phase):
        self.pts = pts
        self.cum = [0.0]
        for a, b in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + math.dist(a, b))
        self.total = self.cum[-1]
        self.start_delay = start_delay
        self.speed = speed
        self.phase = phase

    def at(self, t):
        """Return (x, y, heading_deg) at time t."""
        d = max(0.0, t - self.start_delay) * self.speed
        if d >= self.total:
            a, b = self.pts[-2], self.pts[-1]
            heading = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            heading += 35.0 * math.sin(0.4 * t + self.phase)  # look around while idle
            x, y = b
        else:
            i = 1
            while self.cum[i] < d:
                i += 1
            a, b = self.pts[i - 1], self.pts[i]
            seg = self.cum[i] - self.cum[i - 1]
            f = (d - self.cum[i - 1]) / seg if seg > 0 else 0.0
            x = a[0] + (b[0] - a[0]) * f
            y = a[1] + (b[1] - a[1]) * f
            heading = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        x += 8.0 * math.sin(0.9 * t + self.phase)
        y += 8.0 * math.cos(0.7 * t + self.phase)
        return x, y, heading


def build_routes(spec, rng):
    site = spec["site"]
    routes = []
    for s in range(10):
        jitter = lambda p: (p[0] + rng.uniform(-60, 60), p[1] + rng.uniform(-60, 60))
        if s < 5:
            pts = [jitter(CT_SPAWN), jitter(CT_VIA[site]), jitter(CT_HOLD[site][s])]
        else:
            pts = [jitter(T_SPAWN)] + [jitter(p) for p in T_VIA[site]] + [jitter(SITE_POINT[site])]
        routes.append(Route(pts, start_delay=rng.uniform(0.0, 2.0), speed=WALK_SPEED + rng.uniform(-20, 20),
                            phase=rng.uniform(0, 2 * math.pi)))
    return routes


def check_schedule(spec):
    """Assert the schedule is physically possible. Returns (death_time_by_slot, events)."""
    n = spec["n"]
    death_t = {}
    alive = set(range(10))
    events = []
    for t, killer, victim, weapon, hs in sorted(spec["kills"]):
        assert killer in alive, "round %d: killer %d is dead at %.1f" % (n, killer, t)
        assert victim in alive, "round %d: victim %d already dead at %.1f" % (n, victim, t)
        assert (killer < 5) != (victim < 5), "round %d: teamkill at %.1f" % (n, t)
        assert weapon == LOADOUT[killer], "round %d: %s used %s, loadout is %s" % (
            n, LABELS[killer], weapon, LOADOUT[killer])
        alive.discard(victim)
        death_t[victim] = t
        events.append({"t": t, "kind": "kill", "killer": killer, "victim": victim,
                       "weapon": weapon, "hs": bool(hs)})

    plant_t, planter = spec["plant"]
    assert planter in alive and planter >= 5, "round %d: planter %d cannot plant" % (n, planter)
    assert death_t.get(planter, math.inf) > plant_t, "round %d: planter dies before planting" % n
    events.append({"t": plant_t, "kind": "plant", "player": planter})

    end = spec["end"]
    if end[0] == "defuse":
        _, t_end, defuser = end
        assert defuser in alive and defuser < 5, "round %d: defuser must be a live CT" % n
        assert plant_t < t_end <= plant_t + BOMB_SECONDS, "round %d: defuse outside bomb timer" % n
        events.append({"t": t_end, "kind": "defuse", "player": defuser})
        winner = "CT"
    else:
        t_end = end[1]
        assert abs(t_end - (plant_t + BOMB_SECONDS)) < 1e-9, "round %d: explode must be plant + 40 s" % n
        events.append({"t": t_end, "kind": "explode"})
        winner = "T"
    events.sort(key=lambda e: e["t"])
    return death_t, events, t_end, winner


def make_prob(duration, events, winner, rng):
    """Made-up P(CT wins) at 1 Hz with a logistic link and a random walk."""
    impulses = []
    for ev in events:
        if ev["kind"] == "kill":
            impulses.append((ev["t"], 0.35 if ev["killer"] < 5 else -0.35))
        elif ev["kind"] == "plant":
            impulses.append((ev["t"], -0.8))
    impulses.sort()

    times = [float(s) for s in range(int(math.floor(duration)) + 1)]
    if times[-1] < duration:
        times.append(duration)

    z_base = 0.0
    noise = 0.0
    k = 0
    probs = []
    for tt in times:
        while k < len(impulses) and impulses[k][0] <= tt:
            z_base += impulses[k][1]
            k += 1
        z_base *= 0.97
        noise = 0.8 * noise + rng.gauss(0.0, 0.12)
        drift = (0.05 if winner == "CT" else -0.05) * tt / max(duration, 1.0)
        z = z_base + noise + drift
        if tt >= duration:
            z = 4.2 if winner == "CT" else -4.2  # decisive outcome
        probs.append(round(1.0 / (1.0 + math.exp(-z)), 3))
    return {"t": times, "p": probs}


def simulate_round(spec, rng):
    death_t, events, duration, winner = check_schedule(spec)
    n_frames = int(round(duration / DT)) + 1
    routes = build_routes(spec, rng)

    # Two short hit windows so some living players show reduced HP.
    hit = {}
    for s in rng.sample(range(10), 2):
        t0 = rng.uniform(12.0, 30.0)
        hit[s] = (t0, t0 + 4.0)

    # Loadout buys: money drops at t=5 by the weapon price.
    money0 = [rng.randrange(5000, 7000, 50) for _ in range(10)]

    player_frames = []
    for s in range(10):
        route = routes[s]
        xs, ys, yaws, hps, arms, alives, wpns, monies = [], [], [], [], [], [], [], []
        td = death_t.get(s)
        weapon_idx = WEAPONS.index(LOADOUT[s])
        armor = 0 if s in (8, 9) else 100
        for i in range(n_frames):
            t = i * DT
            te = min(t, td) if td is not None else t
            x, y, heading = route.at(te)
            yaw = int(round((heading + 180.0) % 360.0 - 180.0))
            alive = td is None or t < td
            if not alive:
                hp = 0
            else:
                hp = 100
                if td is not None and t >= td - 2.0:
                    hp = 35
                if s in hit and hit[s][0] <= t < hit[s][1]:
                    hp = min(hp, 60)
            xs.append(round(x, 1))
            ys.append(round(y, 1))
            yaws.append(yaw)
            hps.append(hp)
            arms.append(armor)
            alives.append(1 if alive else 0)
            wpns.append(weapon_idx)
            monies.append(money0[s] if t < 5.0 else max(0, money0[s] - PRICE[LOADOUT[s]] - 400))
        player_frames.append({"x": xs, "y": ys, "yaw": yaws, "hp": hps, "armor": arms,
                              "alive": alives, "weapon": wpns, "money": monies})

    return {
        "n": spec["n"],
        "winner": winner,
        "duration": duration,
        "players": [{"slot": s, "label": LABELS[s], "side": "CT" if s < 5 else "T"} for s in range(10)],
        "frames": {"dt": DT, "n": n_frames, "p": player_frames},
        "events": events,
        "prob": make_prob(duration, events, winner, rng),
    }


def build(seed):
    rng = random.Random(seed)
    rounds = [simulate_round(spec, rng) for spec in ROUND_SPECS]
    ct = sum(1 for r in rounds if r["winner"] == "CT")
    t = len(rounds) - ct
    return {
        "version": 1,
        "match": {"id": "synth001", "map": "de_inferno", "ct_score": ct, "t_score": t,
                  "n_rounds": len(rounds), "model": "winprob_v2 (synthetic sample)"},
        "map_meta": MAP_META,
        "weapons": WEAPONS,
        "rounds": rounds,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write a synthetic replay JSON (schema v1).")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output path (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=7, help="random seed (default: %(default)s)")
    args = ap.parse_args(argv)

    doc = build(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    n_frames = [r["frames"]["n"] for r in doc["rounds"]]
    print("wrote %s (%.1f KB): %d rounds, frames per round %s, score CT %d - T %d" % (
        out, out.stat().st_size / 1024.0, len(doc["rounds"]), n_frames,
        doc["match"]["ct_score"], doc["match"]["t_score"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
