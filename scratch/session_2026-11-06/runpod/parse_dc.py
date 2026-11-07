import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))["data"]["dataCenters"]
targets = {"NVIDIA GeForce RTX 4090":"RTX4090","NVIDIA RTX A5000":"A5000","NVIDIA RTX A6000":"A6000","NVIDIA L40S":"L40S","NVIDIA A100 80GB PCIe":"A100PCIe","NVIDIA A100-SXM4-80GB":"A100SXM80","NVIDIA H100 80GB HBM3":"H100SXM","NVIDIA H100 PCIe":"H100PCIe"}
rows = {}
allids = set()
for dc in d:
    av = {g["gpuTypeId"]: g["stockStatus"] for g in dc.get("gpuAvailability", [])}
    for g in av: allids.add(g)
    for tid, short in targets.items():
        if tid in av:
            rows.setdefault(dc["id"], {})[short] = av[tid]
print("datacenters returned:", len(d))
print("all gpuTypeIds seen:", sorted(allids))
print("--- per-DC stockStatus for target types (only DCs listing the type) ---")
for dc in sorted(rows):
    print(dc, rows[dc])
print("--- count of DCs per target listing type (any status) ---")
for tid, short in targets.items():
    n = sum(1 for dc in d if any(g["gpuTypeId"]==tid for g in dc.get("gpuAvailability",[])))
    st = {}
    for dc in d:
        for g in dc.get("gpuAvailability",[]):
            if g["gpuTypeId"]==tid:
                st[str(g["stockStatus"])] = st.get(str(g["stockStatus"]),0)+1
    print(short, "DCs listing:", n, "status counts:", st)
