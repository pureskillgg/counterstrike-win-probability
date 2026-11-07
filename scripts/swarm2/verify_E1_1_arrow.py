r"""E1-1 verifier, part 4: side-effect check. The C: cs2gpu solve switches pyarrow/libarrow to *_cuda builds.

Reads metadata and one column of an existing project parquet file (read-only, from G:) with the C: interpreter,
and round-trips a small table through a scratch file in the Windows temp scratchpad (not the repo).

Run:
  "C:\Users\<user>\.pixi\workspace-envs\cs2gpu-15282324412032665829\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_E1_1_arrow.py
"""
import os
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

SRC = r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet"

print("python", sys.executable, flush=True)
print("pyarrow", pa.__version__, "build cuda info:", pa.cpp_build_info if hasattr(pa, "cpp_build_info") else "n/a", flush=True)
md = pq.read_metadata(SRC)
print(f"{SRC}: rows={md.num_rows} row_groups={md.num_row_groups} cols={md.num_columns}", flush=True)
first_col = md.schema.names[0]
t = pq.read_table(SRC, columns=[first_col])
print(f"read column {first_col!r}: rows={t.num_rows} dtype={t.schema.field(0).type}", flush=True)
tmp = os.path.join(tempfile.gettempdir(), "verify_E1_1_arrow_roundtrip.parquet")
tbl = pa.table({"x": pa.array(range(1000)), "y": pa.array([float(i) / 3 for i in range(1000)])})
pq.write_table(tbl, tmp)
back = pq.read_table(tmp)
print(f"round trip rows={back.num_rows} equal={back.equals(tbl)}", flush=True)
os.remove(tmp)
