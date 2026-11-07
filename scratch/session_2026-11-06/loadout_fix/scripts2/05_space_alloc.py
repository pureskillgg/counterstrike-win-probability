import numpy as np, os
OUT = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
NS = 72567700
a = np.lib.format.open_memmap(OUT + "loadout_step_i16.npy", mode="w+", dtype=np.int16, shape=(NS, 2, 19))
del a
b = np.lib.format.open_memmap(OUT + "loadout_equip_i32.npy", mode="w+", dtype=np.int32, shape=(NS, 2))
del b
os.makedirs("/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout_stage", exist_ok=True)
print("alloc ok", os.path.getsize(OUT + "loadout_step_i16.npy"), os.path.getsize(OUT + "loadout_equip_i32.npy"))
