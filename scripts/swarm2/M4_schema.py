"""M4: schema and map-relevance inventory of the tomes (metadata only, no row reads).

Lists, for each tome channel, the columns that could carry map geometry, bombsite,
callout or map identity. Reads parquet footers only.
"""
import glob
import os
import re
import sys

import pyarrow.parquet as pq

ROOT = r"G:\pureskill\csds-cs2\tomes"
TOMES = {
    "2026": os.path.join(ROOT, "2026-10-01_to_08", "tome", "csds"),
    "2024": os.path.join(ROOT, "2024-02-g", "tome", "csds"),
}
PAT = re.compile(r"(x_pos|y_pos|z_pos|pos|site|place|map|callout|bomb|plant|area|nav|zone|location)", re.I)


def main():
    for era, base in TOMES.items():
        print(f"===== {era}: {base}")
        for ch in sorted(os.listdir(base)):
            folder = os.path.join(base, ch)
            if not os.path.isdir(folder):
                continue
            files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
            if not files:
                print(f"  {ch}: no dataframe files")
                continue
            schema = pq.read_schema(files[0])
            names = schema.names
            hits = [n for n in names if PAT.search(n)]
            n_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
            print(f"  {ch}: files={len(files)} rows={n_rows} cols={len(names)} map_relevant={hits}")
        print()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
