# Builds the v4 S3 channel job from the v3 body and the shared rule. Four edits, each asserted to match exactly once.
import json
V3S = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/scripts/"
V4S = "/home/sagemaker-user/adam/builder/final/export_clean_v4/scripts/"
rule = open(V3S + "loadout_rule.py", encoding="utf-8").read()
body = open(V3S + "s3_body.py", encoding="utf-8").read()


def rep(text, old, new):
    n = text.count(old)
    assert n == 1, (n, old[:90])
    return text.replace(old, new)


body = rep(body, 'import boto3\n', 'import boto3\nimport pyarrow.parquet as pq\n')
body = rep(body, 'OUTD = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"',
           'OUTD = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"')
body = rep(body,
           '# chunks c063 onward hold matches dated 2026-10-15 and later (their windows reach Oct 21+): not read\n'
           'CHUNKS = ["c%03d" % i for i in range(0, 63)]',
           '# v4: c063-c067 are read as well (they hold the protected keys dated 2026-10-21..25). c068 and up hold no folders.\n'
           'CHUNKS = ["c%03d" % i for i in range(0, 68)]')
body = rep(body, 'PARITY_OUT = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout_stage/parity_s3_rows.parquet"',
           'PARITY_OUT = "/home/sagemaker-user/adam/builder/final/export_clean_v4/scripts/parity_s3_rows_v4_unused.parquet"')
old_rp = '''def read_pages(pkeys, columns):
    bs = list(POOL.map(get_bytes, pkeys))
    frames = []
    for b in bs:
        f = pl.read_parquet(io.BytesIO(b))
        missing = [c for c in columns if c not in f.columns]
        assert not missing, ("missing columns", missing)
        frames.append(f.select(columns))
    return frames
'''
new_rp = '''def read_pages(pkeys, columns):
    # v4: the export-key filter is applied inside the parquet read, so rows of other matches are never materialized
    bs = list(POOL.map(get_bytes, pkeys))
    frames = []
    for b in bs:
        names = pq.read_schema(io.BytesIO(b)).names
        missing = [c for c in columns if c not in names]
        assert not missing, ("missing columns", missing)
        t = pq.read_table(io.BytesIO(b), columns=columns, filters=[("match_key", "in", wanted_list)])
        f = pl.from_arrow(t).with_columns(pl.col("match_key").cast(pl.String))
        frames.append(f.select(columns))
    return frames
'''
body = rep(body, old_rp, new_rp)
TARGET = json.load(open(V4S + "s3_target_v4.json"))
body = rep(body, "TARGET = __TARGET__", "TARGET = " + repr(TARGET))
body = rep(body, "P3 = __P3__", "P3 = []")
body = rep(body, "P4 = __P4__", "P4 = []")
job = rule + "\n" + body
open(V4S + "s3_job_v4.py", "w", encoding="utf-8").write(job)
print("s3_job_v4.py written; target keys", len(TARGET), "; chunks c000-c067; bytes", len(job))
