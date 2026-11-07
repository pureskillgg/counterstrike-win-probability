import glob, os, pyarrow.parquet as pq, pyarrow.compute as pc
for label, root in [("2026", r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"), ("2024", r"G:\pureskill\csds-cs2\tomes\2024-02-all\tome\csds")]:
    d = [x for x in os.listdir(root) if x.startswith("basic_round_state.")][0]
    tb = pq.read_table(sorted(glob.glob(os.path.join(root, d, "dataframe_*")))[0], columns=["t_score", "ct_score", "t_score_raw", "ct_score_raw", "event_type"])
    n = tb.num_rows
    diff_t = pc.sum(pc.not_equal(pc.fill_null(tb["t_score"], -999), pc.fill_null(tb["t_score_raw"], -999))).as_py()
    diff_c = pc.sum(pc.not_equal(pc.fill_null(tb["ct_score"], -999), pc.fill_null(tb["ct_score_raw"], -999))).as_py()
    print(f"{label}: round_state rows={n}, t_score differs from raw on {diff_t} rows, ct_score differs on {diff_c} rows")

