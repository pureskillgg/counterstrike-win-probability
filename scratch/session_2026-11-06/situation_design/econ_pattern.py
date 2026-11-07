import csv, collections
PATH = "C:/Users/13144/Documents/pureskill/counterstrike-win-probability/docs/eval_reports/positions_validation_gdt/by_situation.csv"
rows = list(csv.DictReader(open(PATH, encoding="utf-8")))
N = sum(int(r["datapoints"]) for r in rows)
M = "p_full_trainctr"
def key(r): return (r["econ_ct"], r["econ_t"])
tot = collections.Counter(); weak = collections.Counter(); totdp = collections.Counter(); weakdp = collections.Counter()
for r in rows:
    k = key(r); n = int(r["datapoints"])
    tot[k] += 1; totdp[k] += n
    if float(r["gain_" + M]) < 0:
        weak[k] += 1; weakdp[k] += n
print("econ (ct,t): all cells / weak cells / all datapoints / weak datapoints")
for k in sorted(tot, key=lambda k: -totdp[k]):
    print(k, tot[k], weak[k], totdp[k], weakdp[k], f"weak share of cells {weak[k]/tot[k]:.1%}")
lowlow = [r for r in rows if r["econ_ct"] == "low" and r["econ_t"] == "low"]
print("low/low cells", len(lowlow), "datapoints", sum(int(r['datapoints']) for r in lowlow))
early = [r for r in rows if r["planted"] == "0" and r["tb"] == "lot"]
print("planted 0, tb lot cells", len(early), "datapoints", sum(int(r['datapoints']) for r in early))
ll_low = sum(float(r["ll_" + M]) * int(r["datapoints"]) for r in lowlow); ll_all = sum(float(r["ll_" + M]) * int(r["datapoints"]) for r in rows)
print(f"model loss share in low/low cells: {ll_low/ll_all:.2%}")
# signed level errors among weak cells with n>=51
sub = [r for r in rows if float(r["gain_" + M]) < 0 and int(r["datapoints"]) >= 51]
over = sum(1 for r in sub if float(r["mean_" + M]) > float(r["ct_win_share"]))
print("weak cells n>=51:", len(sub), "model mean above observed CT share:", over, "below:", len(sub)-over)
