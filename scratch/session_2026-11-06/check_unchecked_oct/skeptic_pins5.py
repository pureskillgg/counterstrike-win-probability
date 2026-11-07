import sys
t=open(sys.argv[1], encoding="utf-8", errors="replace").read()
i=232572
j=t.rfind('"id": "', 0, i)
print(t[j:j+300].replace("\n"," | "))
print("...")
print(t[i-900:i+200].replace("\n"," | "))
