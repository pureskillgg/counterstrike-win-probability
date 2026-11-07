import json
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

F = "/home/sagemaker-user/adam/builder/final"
X = F + "/export_clean_v2"
E = F + "/export_clean"

rm2 = np.load(X + "/round_match.npy", mmap_mode="r")
off2 = np.load(X + "/offsets.npy", mmap_mode="r")
steps2 = np.load(X + "/steps_alive.npy", mmap_mode="r")
print("EXPORT_V2 rounds(round_match len)=%d offsets_last=%d steps_alive_len=%d" % (len(rm2), int(off2[-1]), len(steps2)))
flag1 = np.load(E + "/loadout_flag.npy", mmap_mode="r")
print("EXPORT_V1 loadout_flag==1 count (dup-player rounds)=%d of %d" % (int(np.sum(flag1 == 1)), len(flag1)))
cr = pq.read_table(F + "/clean_rounds.parquet", columns=["use"])
print("CLEAN_ROUNDS rows=%d use_true=%d" % (cr.num_rows, int(pc.sum(pc.cast(cr["use"], "int64")).as_py())))
cg = pq.ParquetFile(F + "/clean_grid.parquet").metadata
print("CLEAN_GRID rows=%d" % cg.num_rows)
cm = pq.ParquetFile(F + "/clean_matches.parquet").metadata
print("CLEAN_MATCHES rows=%d" % cm.num_rows)
