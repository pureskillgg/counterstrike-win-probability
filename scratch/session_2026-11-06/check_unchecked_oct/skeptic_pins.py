import re, sys
t=open(sys.argv[1], encoding="utf-8", errors="replace").read()
print("chars", len(t))
for pid in ["425fca91","e53ed710","4edc74c3","dd78df27","5f8dea45","2702982d","ea90870a","815ecd00"]:
    i=t.find(pid)
    print("==", pid, "found" if i>=0 else "NOT FOUND", i)
    if i>=0:
        print(t[max(0,i-200):i+700].replace("\n"," | "))
        print()
