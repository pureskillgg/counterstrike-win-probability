import sys, re
t=open(sys.argv[1], encoding="utf-8", errors="replace").read()
for pat in ["contaminat", "GDT grad", "intern", "ext cache", "cache_snapshot_2026_ext", "Clean for", "3,463", "3463", "oct23_31", "ext_baseline", "s4_xgb"]:
    idx=[m.start() for m in re.finditer(re.escape(pat), t)]
    print("==", pat, len(idx), idx[:8])
for m in list(re.finditer("contaminat", t))[:3]:
    i=m.start(); print("--- ctx contaminat", t[max(0,i-500):i+400].replace("\n"," | ")); print()
for m in list(re.finditer("3,463|3463", t))[:2]:
    i=m.start(); print("--- ctx 3463", t[max(0,i-300):i+300].replace("\n"," | ")); print()
