import json, os, sys
S = sys.argv[1]
def load(n):
    return json.load(open(os.path.join(S, n), encoding="utf-8"))
gpu_types = {g["id"]: g for g in load("gql_q1.json")["data"]["gpuTypes"]}
alias = load("gql_alias.json")["data"]
dcs = load("gql_a5.json")["data"]["dataCenters"]
vols = load("rest_networkvolumes.json")
pods = load("rest_pods.json")

targets = [
    ("RTX 4090", "NVIDIA GeForce RTX 4090"),
    ("RTX A5000", "NVIDIA RTX A5000"),
    ("RTX A6000", "NVIDIA RTX A6000"),
    ("L40S", "NVIDIA L40S"),
    ("A100 80GB PCIe", "NVIDIA A100 80GB PCIe"),
    ("A100 80GB SXM", "NVIDIA A100-SXM4-80GB"),
    ("H100 80GB SXM", "NVIDIA H100 80GB HBM3"),
    ("H100 PCIe", "NVIDIA H100 PCIe"),
]

def lowest(i, gc, sec):
    rows = alias[f"t{i}_g{gc}_s{sec}"]
    lp = rows[0]["lowestPrice"]
    return {"uninterruptable_price": lp["uninterruptablePrice"],
            "stock_status": lp["stockStatus"],
            "minimum_bid_price": lp["minimumBidPrice"]}

def dc_stock(tid):
    low, null, listed = [], [], []
    for dc in dcs:
        for g in dc.get("gpuAvailability", []):
            if g["gpuTypeId"] == tid:
                listed.append(dc["id"])
                (low if g["stockStatus"] == "Low" else null).append(dc["id"])
    return sorted(low), sorted(null), sorted(listed)

gpus = []
stock_map = {}
for i, (short, tid) in enumerate(targets, start=1):
    p = gpu_types[tid]
    low, null, listed = dc_stock(tid)
    stock_map[short] = {"low": low, "null": null}
    gpus.append({
        "gpu": short,
        "graphql_gpu_type_id": tid,
        "memory_gb": p["memoryInGb"],
        "secure_cloud_offered": p["secureCloud"],
        "community_cloud_offered": p["communityCloud"],
        "list_price_per_gpu_hr": {"secure": p["securePrice"], "community": p["communityPrice"]},
        "lowest_price_1gpu": {"secure": lowest(i, 1, "true"), "community": lowest(i, 1, "false")},
        "lowest_price_5gpu": {"secure": lowest(i, 5, "true"), "community": lowest(i, 5, "false")},
        "five_gpu_cost_per_hr_at_list": {"secure": round(5 * p["securePrice"], 2), "community": round(5 * p["communityPrice"], 2)},
        "datacenters_listing_type": listed,
        "datacenters_stock_low": low,
        "datacenters_stock_null": null,
    })

def vol_row(v):
    short_dc = v["dataCenterId"]
    rel = []
    for short, tid in targets:
        if short_dc in stock_map.get(short, {}).get("low", []):
            rel.append(f"{short}: Low")
    return {"id": v["id"], "name": v["name"], "size_gb": v["size"], "datacenter": short_dc,
            "target_types_with_low_stock_here": rel}

volumes = [vol_row(v) for v in vols]

recommended = [
    {"pod": 1, "gpu": "A100 80GB SXM", "graphql_gpu_type_id": "NVIDIA A100-SXM4-80GB", "datacenter": "EU-RO-1", "cloud": "secure", "price_per_gpu_hr": 1.79},
    {"pod": 2, "gpu": "A100 80GB SXM", "graphql_gpu_type_id": "NVIDIA A100-SXM4-80GB", "datacenter": "EUR-IS-1", "cloud": "secure", "price_per_gpu_hr": 1.79},
    {"pod": 3, "gpu": "RTX 4090", "graphql_gpu_type_id": "NVIDIA GeForce RTX 4090", "datacenter": "EU-RO-1", "cloud": "secure", "price_per_gpu_hr": 0.89},
    {"pod": 4, "gpu": "RTX 4090", "graphql_gpu_type_id": "NVIDIA GeForce RTX 4090", "datacenter": "EUR-IS-1", "cloud": "secure", "price_per_gpu_hr": 0.89},
    {"pod": 5, "gpu": "RTX 4090", "graphql_gpu_type_id": "NVIDIA GeForce RTX 4090", "datacenter": "EUR-IS-2", "cloud": "secure", "price_per_gpu_hr": 0.89},
]
for r in recommended:
    low = stock_map[r["gpu"]]["low"]
    assert r["datacenter"] in low, ("recommendation not backed by Low stock", r)
    r["stock_status_in_datacenter"] = "Low"
    r["stock_note"] = "API stockStatus Low; no GPU count returned"
total_secure = round(sum(r["price_per_gpu_hr"] for r in recommended), 2)
total_comm = round(2 * 1.39 + 3 * 0.34, 2)

out = {
    "title": "fix332 RunPod capacity and cost for 5+ GPUs (read-only)",
    "as_of": "2026-11-05",
    "read_only": True,
    "pods_created_or_changed": 0,
    "volumes_created_or_changed": 0,
    "pods_running_now": len(pods),
    "network_volumes": volumes,
    "gpu_capacity_and_price": gpus,
    "recommended_five_pods": recommended,
    "recommended_five_pods_total_per_hr": {"secure": total_secure, "community_alternative_same_mix": total_comm,
                                           "note": "community alternative uses A100 SXM 1.39 and RTX 4090 0.34 list prices; per-datacenter community stock is not separable in the API"},
    "alternatives": {
        "fastest_80gb_option": {"gpu": "H100 80GB SXM", "low_stock_datacenters": stock_map["H100 80GB SXM"]["low"],
                                "five_gpu_per_hr": {"secure": round(5 * 3.99, 2), "community": round(5 * 2.69, 2)},
                                "volume_note": "US-CA-2 has H100 SXM Low and holds network volume cs2-copy-caS2 (60 GB)"},
        "cheapest_option": {"gpu": "RTX 4090", "low_stock_datacenters": stock_map["RTX 4090"]["low"],
                            "five_gpu_per_hr": {"secure": round(5 * 0.89, 2), "community": round(5 * 0.34, 2)}},
    },
    "datacenters_with_room_for_5_of_one_type": [],
    "datacenters_with_room_for_5_of_one_type_note": "none confirmed: the API returns no GPU counts, only stockStatus Low or null, and lowestPrice with gpuCount 5 is null for every type",
    "cpu_ssh_notes": {
        "source": "user-reported, not verifiable via read-only API (pods list is empty)",
        "community_cpu_pod": "no SSH address",
        "secure_cpu_pod_other_datacenter": "no SSH address",
        "EU-RO-1": "SSH worked",
        "implication": "start the GPU pods in EU-RO-1 first and confirm SSH before scaling to five"
    },
    "api_notes": [
        "lowestPrice(input: {gpuCount: 5}) returned null for every GPU type in secure and community filters",
        "lowestPrice(gpuCount 1) returned values for RTX 4090, L40S, A100 PCIe, A100 SXM 80GB, H100 SXM (secure) and A100 SXM 80GB (community); other values null",
        "L40S lowestPrice secure 1-GPU shows Low, but no datacenter lists L40S as Low; the two sources disagree",
        "dataCenters.gpuAvailability does not separate secure from community",
        "communityCloud is not a valid GpuLowestPriceInput field; DataCenter has no listing, gpuTypes or availableGpuTypes field",
        "stockStatus values seen: Low and null only; null is treated as not confirmed available, not as proven out of stock"
    ],
    "evidence": [
        "REST GET /v1/networkvolumes HTTP 200: 4 volumes (US-IL-1 x3, US-CA-2 x1)",
        "REST GET /v1/pods HTTP 200: empty list",
        "GraphQL gpuTypes list prices HTTP 200: %d GPU types" % len(gpu_types),
        "GraphQL gpuTypes lowestPrice aliases HTTP 200 for 8 types at gpuCount 1 and 5, secure and community",
        "GraphQL dataCenters gpuAvailability HTTP 200: %d datacenters" % len(dcs)
    ],
}
dest = os.path.join(S, "runpod_capacity.json")
with open(dest, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("wrote", dest, os.path.getsize(dest), "bytes")
print("total secure per hr:", total_secure, "community alt:", total_comm)
for g in gpus:
    print(g["gpu"], "secure", g["list_price_per_gpu_hr"]["secure"], "comm", g["list_price_per_gpu_hr"]["community"], "low:", g["datacenters_stock_low"])
for v in volumes: print(v)
