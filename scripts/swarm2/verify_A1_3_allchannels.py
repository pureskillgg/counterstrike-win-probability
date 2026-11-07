# Verifier for A1-3 scope: for EVERY channel folder in the 2026 tome, count distinct per-page schemas
# (names+types, and names-only) using footer reads. Confirms whether the drift is confined to the three claimed channels.
import os
import pyarrow.parquet as pq

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
drift_names = []
for ch in sorted(os.listdir(T)):
    folder = os.path.join(T, ch)
    if ch == "header" or not os.path.isdir(folder):
        continue
    pages = sorted(f for f in os.listdir(folder) if f.startswith("dataframe_"))
    sig_full = []
    sig_names = []
    for p in pages:
        s = pq.ParquetFile(os.path.join(folder, p)).schema_arrow
        sig_full.append(tuple((n, str(t)) for n, t in zip(s.names, s.types)))
        sig_names.append(tuple(sorted(s.names)))
    nf = len(set(sig_full))
    nn = len(set(sig_names))
    flag = "DRIFT(types)" if nf > 1 else ("order-only" if nn < nf else "")
    if nf > 1:
        drift_names.append(ch)
    print(f"{ch:60s} pages={len(pages):2d} distinct_full_schemas={nf} distinct_name_sets={nn} {flag}")
print("channels with type-or-column drift:", drift_names)
print("done")
