# Builds PATCHED COPIES of the two space scripts in this scratch folder (proposal only; the originals are not touched, nothing is run).
import pathlib
S = pathlib.Path(__file__).resolve().parent

def rep(s, old, new, count=1):
    n = s.count(old)
    assert n == count, (old[:70], n)
    return s.replace(old, new)

b = (S / "pos_bucket_current.py").read_text(encoding="utf-8")
b = rep(b, 'B = "pureskillgg-csgo-production-sagemaker-0"',
        'B = "pureskillgg-csgo-production-sagemaker-0"\nSTEP = int(os.environ.get("POSB_STEP", "1"))  # 1 = every export second (k = 0,1,2,...); 2 = the old even-step build')
b = rep(b, 'ks = np.arange(0, n, 2, dtype=np.int64)', 'ks = np.arange(0, n, STEP, dtype=np.int64)')
b = rep(b, 'MPF.feats_for_day(pos, sta, G, pxy_b, sites, 2, ev)', 'MPF.feats_for_day(pos, sta, G, pxy_b, sites, STEP, ev)')
b = rep(b, 'OUTD = os.environ.get("POSB_OUT", FIN + "/pos_parts")', 'OUTD = os.environ.get("POSB_OUT", FIN + "/pos_parts_every1")')
b = rep(b, "The wanted seconds are the export's seconds with an even index inside the round (k = 0, 2, 4, ...), at tick = freeze + 64 k.",
        "The wanted seconds are every export second inside the round (k = 0, 1, 2, ...; STEP=1), at tick = freeze + 64 k.")
(S / "pos_bucket_step.py").write_text(b, encoding="utf-8")

a = (S / "pos_assemble_current.py").read_text(encoding="utf-8")
a = rep(a, "cnt = (n + 1) // 2", "cnt = n  # every export second: one pos row per export step, so pos_offsets == offsets")
a = rep(a, "tick = frz[rid] + 64 * (2 * j)", "tick = frz[rid] + 64 * j")
a = rep(a, 'glob.glob(FIN + "/pos_parts/bucket_c*.parquet")', 'glob.glob(FIN + "/pos_parts_every1/bucket_c*.parquet")')
a = rep(a, 'glob.glob(HOME + "/laptop_pos/every2_sites_v1/*.parquet")', 'glob.glob(HOME + "/laptop_pos/every1_sites_v1/*.parquet")')
a = rep(a, 'V2 = FIN + "/export_clean_v2"', 'V2 = FIN + "/export_clean_v2"  # input export (read only)\nOUT = os.environ.get("POSA_OUT", FIN + "/export_clean_v4")  # new export folder; the v2 folder is never written')
a = rep(a, 'E, N = int(pos_off[-1]), len(rn)', 'E, N = int(pos_off[-1]), len(rn)\nassert E == int(off[-1]), "every-second layout must have one row per export step"')
a = rep(a, 'np.save(V2 + "/pos_', 'np.save(OUT + "/pos_', count=5)
a = rep(a, 'open(V2 + "/pos_README.json", "w")', 'open(OUT + "/pos_README.json", "w")')
a = rep(a, '"step_rule": "even steps k=0,2,..<n per round; tick = freeze + 64k; pos_offsets indexes these steps"',
        '"step_rule": "every export step k=0..n-1 per round; tick = freeze + 64k; pos_offsets == offsets.npy"')
a = rep(a, '"steps_even": E', '"steps": E')
a = rep(a, "Assemble the position features onto export_clean_v2's every-other-second steps, in mixed dtypes.",
        "Assemble the position features onto export_clean_v2's every-second steps (every export step), in mixed dtypes.")
(S / "pos_assemble_step.py").write_text(a, encoding="utf-8")
print("patched copies written:", S / "pos_bucket_step.py", S / "pos_assemble_step.py")
