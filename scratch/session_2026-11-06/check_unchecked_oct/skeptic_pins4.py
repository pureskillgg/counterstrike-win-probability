import sys, re
t=open(sys.argv[1], encoding="utf-8", errors="replace").read()
for m in re.finditer(r"oct23_31|2,549|2,550|2,848|expanded cache|21 Oct|Oct 21", t):
    i=m.start()
    print("--", m.group(0), "@", i, ":", t[max(0,i-160):i+200].replace("\n"," | "))
    print()
