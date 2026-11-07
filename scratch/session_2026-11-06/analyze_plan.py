import json, collections, re, os
import numpy as np
import polars as pl

B = '/home/sagemaker-user/adam/builder/'
V2 = B + 'final/export_clean_v2/'
V4 = B + 'final/export_clean_v4/'
FX = B + 'fixes/'
CH = B + 'final/pos_all_seconds/'
OUTJ = B + 'fixes/fix332/analysis.json'
res = {}

keys = json.load(open(V2 + 'match_keys.json'))
kid = {k: i for i, k in enumerate(keys)}
rm = np.load(V2 + 'round_match.npy').astype(np.int64)
rn = np.load(V2 + 'round_num.npy').astype(np.int64)
off = np.load(V2 + 'offsets.npy').astype(np.int64)
ridx = {(int(rm[i]), int(rn[i])): i for i in range(len(rm))}

ov = pl.read_csv(FX + 'freeze_fix/override_rounds.csv', infer_schema_length=0).with_columns(pl.col('round').cast(pl.Int64))
fixed = ov.filter(pl.col('status') == 'fixed').select('match_key', 'round')
notfixed = ov.filter(pl.col('status') != 'fixed').select('match_key', 'round')
sc = pl.read_parquet(FX + 'freeze_scope/scope_rounds.parquet')
aff = sc.filter(pl.col('affected'))
res['scope_rounds'] = sc.height
res['scope_affected_rounds'] = aff.height
res['scope_affected_matches'] = aff['match_key'].n_unique()
res['fixed_rounds'] = fixed.height
res['fixed_in_scope'] = fixed.join(sc, on=['match_key', 'round'], how='inner').height
res['fixed_in_scope_affected'] = fixed.join(aff, on=['match_key', 'round'], how='inner').height
rem = aff.join(fixed, on=['match_key', 'round'], how='anti')
res['affected_minus_fixed'] = rem.height
res['rem_unro_only'] = rem.filter((pl.col('dup_alive') == 0) & (pl.col('unro_alive') > 0)).height
res['rem_dup_and_unro'] = rem.filter((pl.col('dup_alive') > 0) & (pl.col('unro_alive') > 0)).height
res['rem_dup_only'] = rem.filter((pl.col('dup_alive') > 0) & (pl.col('unro_alive') == 0)).height
res['rem_other'] = rem.filter((pl.col('dup_alive') == 0) & (pl.col('unro_alive') == 0)).height
res['notfixed_row'] = notfixed.to_dicts()
res['rem_alive_n_hist'] = {str(k): v for k, v in collections.Counter(rem['alive_n'].to_list()).items()}
res['rem_matches'] = rem['match_key'].n_unique()
res['fixed_matches'] = fixed['match_key'].n_unique()
aff_m = sorted(aff['match_key'].unique().to_list())
fix_m = sorted(fixed['match_key'].unique().to_list())
res['affected_matches'] = len(aff_m)
res['fixed_matches_subset_of_affected'] = set(fix_m) <= set(aff_m)
res['affected_matches_not_fixed_count'] = len(set(aff_m) - set(fix_m))
res['affected_matches_not_in_v2_index'] = [m for m in aff_m if m not in kid][:5]

src = np.load(V4 + 'pos_v4/match_pos_source.npy')
res['affected_pos_source_counts'] = {str(k): v for k, v in collections.Counter(int(src[kid[m]]) for m in aff_m if m in kid).items()}
res['fixed_pos_source_counts'] = {str(k): v for k, v in collections.Counter(int(src[kid[m]]) for m in fix_m if m in kid).items()}
res['laptop_sourced_affected_matches'] = [m for m in aff_m if m in kid and int(src[kid[m]]) != 1]

dates = {}
for m in aff_m:
    d = re.match(r'csds/(\d{4})/(\d{2})/(\d{2})/', m)
    dates[m] = d.group(1) + '-' + d.group(2) + '-' + d.group(3)
res['affected_dates_min_max'] = [min(dates.values()), max(dates.values())]
res['affected_matches_dated_2026_10_21_to_31'] = sorted(m for m, d in dates.items() if '2026-10-21' <= d <= '2026-10-31')
res['affected_matches_dated_2026_10_26_to_31'] = sorted(m for m, d in dates.items() if '2026-10-26' <= d <= '2026-10-31')

ls = pl.read_csv(V4 + 'match_loadout_status.csv', infer_schema_length=0)
lsa = ls.filter(pl.col('match_key').is_in(aff_m))
res['loadout_status_batch_counts_affected'] = {k: v for k, v in collections.Counter(lsa['v4_batch'].to_list()).items()}

aff_set = set(aff_m)
chunk_of = {}
for i in range(0, 72):
    c = 'c%03d' % i
    p = B + 'data/%s/matches.parquet' % c
    if not os.path.exists(p):
        continue
    mk = pl.read_parquet(p, columns=['match_key'])['match_key'].to_list()
    for m in mk:
        if m in aff_set:
            chunk_of.setdefault(m, []).append(c)
res['affected_matches_found_in_chunk_lists'] = len(chunk_of)
res['affected_match_chunk_multi'] = {m: v for m, v in chunk_of.items() if len(v) > 1}
chunks_aff = sorted(set(c for v in chunk_of.values() for c in v))
res['affected_chunks'] = chunks_aff
res['affected_chunks_count'] = len(chunks_aff)
res['affected_matches_per_chunk'] = {c: sum(1 for v in chunk_of.values() if c in v) for c in chunks_aff}
res['affected_matches_missing_chunk'] = [m for m in aff_m if m not in chunk_of][:10]

rows_by_chunk = {}
tot_rows = 0
aff_rounds_set = set(zip(aff['match_key'].to_list(), aff['round'].to_list()))
fix_set = set(zip(fixed['match_key'].to_list(), fixed['round'].to_list()))
rem_set = aff_rounds_set - fix_set
for c in chunks_aff:
    d = pl.read_parquet(CH + 'bucket_%s.parquet' % c, columns=['match_key', 'round', 'second'])
    tot_rows += d.height
    ar = d.filter(pl.col('match_key').is_in(aff_m))
    pairs = set(zip(ar['match_key'].to_list(), ar['round'].to_list()))
    rows_by_chunk[c] = {
        'rows_total': d.height,
        'rows_affected_matches': ar.height,
        'rounds_affected_with_rows': len(pairs & aff_rounds_set),
        'rounds_fixed_with_rows': len(pairs & fix_set),
        'rounds_rem_with_rows': len(pairs & rem_set),
    }
res['rows_by_affected_chunk'] = rows_by_chunk
res['rows_total_in_affected_chunks'] = tot_rows
res['rounds_affected_with_rows_total'] = sum(v['rounds_affected_with_rows'] for v in rows_by_chunk.values())
res['rounds_fixed_with_rows_total'] = sum(v['rounds_fixed_with_rows'] for v in rows_by_chunk.values())
res['rounds_rem_with_rows_total'] = sum(v['rounds_rem_with_rows'] for v in rows_by_chunk.values())

steps_fix = steps_rem = 0
for m, r in fixed.iter_rows():
    i = ridx.get((kid.get(m, -1), int(r)))
    if i is not None:
        steps_fix += int(off[i + 1] - off[i])
for m, r in rem.select('match_key', 'round').iter_rows():
    i = ridx.get((kid.get(m, -1), int(r)))
    if i is not None:
        steps_rem += int(off[i + 1] - off[i])
res['steps_fixed_rounds'] = steps_fix
res['steps_rem_rounds'] = steps_rem

chunk_secs = {}
for path in [B + 'final/pos_all_seconds/build.log', B + 'fixes/pos_every_second/keep_run.log']:
    for ln in open(path).read().splitlines():
        mm = re.match(r'(c\d{3}) matches (\d+) skipped \d+ rows \d+ .* secs (\d+)', ln)
        if mm:
            chunk_secs[mm.group(1)] = (int(mm.group(2)), int(mm.group(3)))
sm = sum(v[0] for v in chunk_secs.values())
ss = sum(v[1] for v in chunk_secs.values())
res['timed_chunks'] = len(chunk_secs)
res['timed_matches_sum'] = sm
res['timed_secs_sum'] = ss
res['secs_per_match'] = round(ss / max(sm, 1), 4)
res['est_sec_affected_matches_only'] = round(ss / max(sm, 1) * len(aff_m), 1)
res['est_sec_full_affected_chunks_from_log'] = sum(chunk_secs[c][1] for c in chunks_aff if c in chunk_secs)
res['matches_in_full_affected_chunks_from_log'] = sum(chunk_secs[c][0] for c in chunks_aff if c in chunk_secs)

json.dump(res, open(OUTJ, 'w'), indent=1, default=str)
print(json.dumps(res, indent=1, default=str)[:14000])
