import json, collections, re
import polars as pl

B = '/home/sagemaker-user/adam/builder/'
FX = B + 'fixes/fix332/'
plan = json.load(open(FX + 'pos_plan.json'))

c = (pl.read_csv(FX + 'candidates.csv', infer_schema_length=0)
     .with_columns(pl.col('round').cast(pl.Int64)))
fx = (pl.read_csv(B + 'fixes/freeze_fix/override_rounds.csv', infer_schema_length=0)
      .with_columns(pl.col('round').cast(pl.Int64))
      .filter(pl.col('status') == 'fixed').select('match_key', 'round'))
f_in = c.join(fx, on=['match_key', 'round'], how='inner')
rem = c.join(fx, on=['match_key', 'round'], how='anti')
cand_check = {
    'rows': c.height,
    'unique_match_round': c.select('match_key', 'round').unique().height,
    'fixed_283_inside': f_in.height,
    'remaining_332_outside': rem.height,
    'kind_fixed': dict(collections.Counter(f_in['kind'].to_list())),
    'kind_remaining': dict(collections.Counter(rem['kind'].to_list())),
    'alive_ct_t_fixed_pairs': dict(collections.Counter('%s/%s' % (a, b) for a, b in zip(f_in['alive_ct'].to_list(), f_in['alive_t'].to_list()))),
    'note': 'alive_ct and alive_t are roster-based counts before the fix (4/4 in all 283 fixed rows). Do not use them as the 5v5 gate; the gate is on the override map.',
}

plan['candidates_csv'] = ('fixes/fix332/candidates.csv exists (written by another process, read here as data): 615 rows, the full affected set. '
                          'It contains the 283 fixed rounds and the 332 remaining rounds, so the rebuild scope is 615 rows with no double count. '
                          'Phase 1 takes the 283 fixed rows; Phase 2 takes the other 332 after decisions D2 to D4.')
plan['candidates_check'] = cand_check

# count-only checks on the download list; no keys printed
dl = open(FX + 'dl_list.tsv').read().splitlines()
n_21_25 = sum(1 for ln in dl if re.search(r'/2026/10/(2[1-5])/', ln))
n_26_31 = sum(1 for ln in dl if re.search(r'/2026/10/(2[6-9]|3[01])/', ln))
s3_lines = sum(1 for _ in open(FX + 's3_folders_2026_10.txt'))
plan['other_process_files_in_fix332'] = {
    'note': 'Written by another process after 22:40 (not by this plan). Read here only for the counts below; contents not used. Not executed by this plan.',
    'download_plan': {
        'script': 'fixes/fix332/download.sh',
        'what': 'downloads player_info and player_status for laptop-era matches that are not in the S3 tome chunks, into fixes/fix332/raw',
        'dl_list_lines': len(dl),
        'dl_list_lines_dated_2026_10_21_to_25_count_only': n_21_25,
        'dl_list_lines_dated_2026_10_26_to_31_count_only': n_26_31,
    },
    's3_folder_listing': {'file': 'fixes/fix332/s3_folders_2026_10.txt', 'lines': s3_lines, 'note': 'listing of October 2026 S3 folders; the holdout rule (no Oct 26-31 keys read) and the protected-date rule (B1, Oct 21-25) need owner confirmation before any further use'},
    'others_not_used': ['candidates_detail.csv', 'missing_matches.csv', 'recover_list.csv', 'round_flags.parquet', 'summary.json', 'scan.log', 'fix332_scan.py', 'finalize_rule_test.py', 'prep_downloads.py', 'sample_oct.txt', 'probe_*.py', 'rule_test.*', 'runpod_capacity.json', 'side_rule_rounds.csv'],
    'flag': 'Owner or parent should confirm whether the 2026-10 listing and downloads are allowed. This plan does not use them.',
}
plan['blockers_summary'] += ' Separate flag: fixes/fix332/download.sh and dl_list.tsv prepare downloads that include 2026-10-21..25 folders (count %d), and s3_folders_2026_10.txt lists October 2026 folders; owner to confirm before use.' % n_21_25

json.dump(plan, open(FX + 'pos_plan.json', 'w'), indent=1, default=str)
print('UPDATED', cand_check['rows'], cand_check['fixed_283_inside'], cand_check['remaining_332_outside'], cand_check['kind_remaining'])
print('dl counts 21-25', n_21_25, '26-31', n_26_31, 's3 lines', s3_lines)
