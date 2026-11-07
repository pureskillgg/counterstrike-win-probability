"""verify D5_10, step 5: trace one 2026 MR24 overtime match under the bake-off and D5 team-A schedules.

Shows per round: winner, freeze-end t/ct, team-A side under each schedule, team-A freeze score and
cumulative team-A wins before the round. Checks that the 2026 zero-diff result at rounds 25-30 is not an artifact.
Read-only; uses verify_D5_10_analysis.py functions.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_trace26.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_D5_10_analysis as A  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    R26, M26 = A.load("2026")
    cands = [m for m in M26 if M26[m]["max_rounds"] == 24 and max(R26[m]) >= 30
             and all(R26[m].get(r, {}).get("winner") is not None for r in range(1, 31))]
    print(f"candidate 2026 MR24 matches with winners for rounds 1-30: {len(cands)}")
    m = cands[0]
    print("match tail", m[-12:], "header t/ct", M26[m]["t_starters_score_final"], M26[m]["ct_starters_score_final"])
    for name, sw in [("BO", A.SCHEDS["BO_MR24"]), ("D5", A.SCHEDS["D5_12_24_27"])]:
        cum = 0
        print(f"-- schedule {name} swaps {sw[:6]}")
        for r in range(1, 31):
            x = R26[m][r]
            aT = A.a_on_t(r, sw)
            a_fe = x["fe_t"] if aT else x["fe_ct"]
            w = x["winner"]
            a_won = 1 if ((aT and w == A.T) or ((not aT) and w == A.CT)) else 0
            if r >= 2:
                print(f"   r={r:>2} w={w} fe_t={x['fe_t']} fe_ct={x['fe_ct']} A_on_T={aT} a_fe={a_fe} cum_before={cum} diff={a_fe - cum}")
            cum += a_won
        print(f"   final A wins (rounds 1-30) = {cum}")


if __name__ == "__main__":
    main()
