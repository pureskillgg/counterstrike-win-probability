import json, collections, os, re, hashlib
import polars as pl

B = '/home/sagemaker-user/adam/builder/'
FX = B + 'fixes/fix332/'
A = json.load(open(FX + 'analysis.json'))

sc = pl.read_parquet(B + 'fixes/freeze_scope/scope_rounds.parquet')
aff = sc.filter(pl.col('affected'))
ov = pl.read_csv(B + 'fixes/freeze_fix/override_rounds.csv', infer_schema_length=0).with_columns(pl.col('round').cast(pl.Int64))
fx = ov.filter(pl.col('status') == 'fixed').select('match_key', 'round')
fixed_m = set(fx['match_key'].to_list())
aff_m = set(aff['match_key'].unique().to_list())
rem_df = aff.join(fx, on=['match_key', 'round'], how='anti')
rem_m = set(rem_df['match_key'].to_list())

chk = (pl.read_csv(FX + 'side_rule_332_check.csv', infer_schema_length=0)
       .with_columns(pl.col('round').cast(pl.Int64), pl.col('alive_n_scope').cast(pl.Int64),
                     pl.col('ct_place').cast(pl.Int64), pl.col('t_place').cast(pl.Int64)))


def cat(r):
    pr = json.loads(r['places_raw'])
    if r['alive_n_scope'] != 10:
        return 'D_not_10_alive'
    if r['ct_place'] == 5 and r['t_place'] == 5:
        if set(pr.keys()) <= {'CTSpawn', 'TSpawn'}:
            return 'A_place_5v5_spawn_labels'
        return 'A2_place_5v5_other_labels'
    if 'CTSideUpper' in pr:
        return 'B_ctsideupper_label'
    return 'C_player_off_spawn_label'


rows = chk.to_dicts()
cats = collections.defaultdict(list)
for r in rows:
    cats[cat(r)].append(r)

cat_summary = {}
for k, v in sorted(cats.items()):
    ms = sorted(set(x['match_key'] for x in v))
    cat_summary[k] = {
        'rounds': len(v),
        'matches': len(ms),
        'match_keys': ms if len(ms) <= 3 else None,
        'alive_n_hist': dict(collections.Counter(str(x['alive_n_scope']) for x in v)),
        'places_seen': sorted(set(p for x in v for p in json.loads(x['places_raw']).keys())),
    }
    if k in ('D_not_10_alive', 'C_player_off_spawn_label'):
        cat_summary[k]['rounds_list'] = [[x['match_key'], x['round']] for x in v]

B_rounds = sorted(x['round'] for x in cats['B_ctsideupper_label'])
B_match = sorted(set(x['match_key'] for x in cats['B_ctsideupper_label']))
C_rows = cats['C_player_off_spawn_label']
D_rows = cats['D_not_10_alive']
D9 = [x for x in D_rows if x['alive_n_scope'] == 9]
D6 = [x for x in D_rows if x['alive_n_scope'] == 6]
D9_match_counts = collections.Counter(x['match_key'] for x in D9)

# per-chunk for the 5 affected chunks
CH5 = ['c029', 'c043', 'c044', 'c045', 'c046']
chunk_of = {}
for c in CH5:
    mk = pl.read_parquet(B + 'data/%s/matches.parquet' % c, columns=['match_key'])['match_key'].to_list()
    for m in mk:
        if m in aff_m:
            chunk_of[m] = c

chunk_secs = {}
for path in [B + 'final/pos_all_seconds/build.log', B + 'fixes/pos_every_second/keep_run.log']:
    for ln in open(path).read().splitlines():
        mm = re.match(r'(c\d{3}) matches (\d+) skipped \d+ rows \d+ .* secs (\d+)', ln)
        if mm:
            chunk_secs[mm.group(1)] = (int(mm.group(2)), int(mm.group(3)))

per_chunk = {}
for c in CH5:
    ms = [m for m in chunk_of if chunk_of[m] == c]
    per_chunk[c] = {
        'matches_total': len(ms),
        'matches_fixed_only': len([m for m in ms if m in fixed_m and m not in rem_m]),
        'matches_remaining_only': len([m for m in ms if m in rem_m and m not in fixed_m]),
        'matches_mixed_fixed_and_remaining': len([m for m in ms if m in rem_m and m in fixed_m]),
        'rounds_fixed_283': A['rows_by_affected_chunk'][c]['rounds_fixed_with_rows'],
        'rounds_remaining_332': A['rows_by_affected_chunk'][c]['rounds_rem_with_rows'],
        'rows_in_chunk_file': A['rows_by_affected_chunk'][c]['rows_total'],
        'rows_in_affected_matches': A['rows_by_affected_chunk'][c]['rows_affected_matches'],
        'logged_matches_full_rerun': chunk_secs.get(c, [0, 0])[0],
        'logged_secs_full_rerun': chunk_secs.get(c, [0, 0])[1],
    }

fixed_by_chunk = collections.Counter(chunk_of[m] for m in fixed_m if m in chunk_of)
phase1_chunks = sorted(fixed_by_chunk.keys())
phase1_secs_full = sum(chunk_secs[c][1] for c in phase1_chunks if c in chunk_secs)
spm = A['secs_per_match']
phase1_match_only = round(spm * len(fixed_m), 1)
phase2_chunks = sorted(set(chunk_of[m] for m in rem_m if m in chunk_of))

need = ['player_vector', 'player_status', 'player_info', 'player_blind']
raw_ok = 0
scr_ok = 0
raw_miss = []
for m in sorted(aff_m):
    f = m.rsplit('/', 1)[0]
    if all(os.path.exists(B + 'raw_cache_bucket/' + f + '/' + n) for n in need):
        raw_ok += 1
    else:
        raw_miss.append(m)
    if all(os.path.exists('/mnt/sagemaker-nvme/posb_all_seconds/' + f + '/' + n) for n in need):
        scr_ok += 1

ls = pl.read_csv(B + 'final/export_clean_v4/match_loadout_status.csv', infer_schema_length=0)
lsa = ls.filter(pl.col('match_key').is_in(sorted(aff_m)))
test_m = sorted(set(lsa.filter(pl.col('v4_batch') == 'v4_s3_test')['match_key'].to_list()))
test_m_cats = {m: ('fixed_and_remaining' if (m in fixed_m and m in rem_m) else ('fixed_only' if m in fixed_m else 'remaining_only')) for m in test_m}
test_rounds_in_rem = int(rem_df.filter(pl.col('match_key').is_in(test_m)).height)
protected_n = int(lsa.filter(pl.col('v4_batch') == 'v4_s3_protected').height)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


hashes = {
    'feat_used_by_chunks': sha(B + 'fixes/pos_every_second/feat/make_pos_features.py'),
    'posport_current_make_pos_features': sha(B + 'posport/scripts/data/make_pos_features.py'),
    'override_rounds_csv': sha(B + 'fixes/freeze_fix/override_rounds.csv'),
    'scope_rounds_parquet': sha(B + 'fixes/freeze_scope/scope_rounds.parquet'),
}

plan = {
    'task': 'fix332 plan: rebuild pos_v4 positions for the 283 freeze-fixed rounds and the 332 unfixed unrostered/duplicate rounds',
    'built_on': '2026-11-05',
    'status': 'PLAN ONLY. Nothing built. Writes limited to fixes/fix332/. pos_v4, export_clean_v2, export_clean_v3_loadout and export_clean_v4 untouched. Holdout B and 2026-10-26..31 keys not read (date check on match_key strings only).',
    'candidates_csv': 'fixes/fix332/candidates.csv does not exist, so no rounds were added from it. The 332 list is in fixes/fix332/side_rule_332_check.csv (affected rounds from freeze_scope/scope_rounds.parquet minus the 283 fixed).',
    'answers': {
        'which_feature_columns_depend_on_side': 'All 24 pos columns. feats_for_day splits players into CT (side 3) and T (side 2) at feat lines 89-93, and every column is built from those sets: site_hi/lo, nearest_site_mean and spread (sites), blind_n and blind_sum (blindness), sees_enemy and exposed (view), in_contact, ct_t_centroid_dist and min_enemy_dist (geometry), c_min_bomb_dist, c_n_bomb_500 and defuse_slack (CT), t_min_bomb_dist (T). Not side-dependent inputs: bomb position bx and by, seconds_since_plant, planted flag, map sites, the tick grid. The step set depends on side only through the side 2 or 3 filter at keep script line 125.',
        'chunk_files_holding_rounds': {'phase1_283_rounds': phase1_chunks, 'phase2_332_rounds': phase2_chunks, 'all_615_rounds': CH5, 'count_all': len(CH5)},
        'per_round_patch_feasible': 'Yes at round level, not as an arithmetic patch. Bucket chunk rows are keyed by (match_key, round, second) with 0 duplicates (pos_v4 README counts), so the rows of target rounds can be dropped and recomputed. Chunk rows have no side or player column, so a corrected value cannot be derived from them; rows must be recomputed from raw. Chunk parquet files are rewritten as new v5 files, not edited in place. Non-target rounds are copied and checked by value.',
        'runtime': {
            'keep_log_secs_per_match': spm,
            'keep_log_basis': 'final/pos_all_seconds/build.log (c000) and fixes/pos_every_second/keep_run.log (c001 to c046): 47 chunks, 32,065 matches, 4,933 s, both parities, 16 threads.',
            'phase1_283_compute_match_level_sec': phase1_match_only,
            'phase1_283_full_chunk_rerun_sec_from_log': phase1_secs_full,
            'assemble_pos_v4_sec_from_run_log': 57,
            'note': 'Cost scales with matches processed, so a match-level run filtered to target rounds is far cheaper than a full-chunk rerun. Setup and reads add time. A run over about 100 s goes under nohup with a log, polled in short calls.',
        },
    },
    'scope': {
        'fixed_283': {'rounds': 283, 'matches': len(fixed_m), 'steps': A['steps_fixed_rounds'], 'chunks': phase1_chunks,
                      'chunk_fixed_match_counts': dict(fixed_by_chunk),
                      'side_source': 'fixes/freeze_fix/override_rounds.csv column override: a full map of player_id 0..9 to side 2 or 3 per round; 5 CT and 5 T in all 283; gate passed 283 of 283 in fixes/freeze_fix/rule_test.log'},
        'remaining_332': {'rounds': A['affected_minus_fixed'], 'matches': len(rem_m), 'steps': A['steps_rem_rounds'], 'chunks': phase2_chunks,
                          'breakdown_by_flags': {'unro_only': A['rem_unro_only'], 'dup_and_unro': A['rem_dup_and_unro'], 'dup_only': A['rem_dup_only']},
                          'categories': cat_summary,
                          'side_rule_status': 'Not decided. Owner side rule pending (fixes/readiness/inventory.json blocker B2). Place-based check: fixes/fix332/side_rule_332_check.csv and side_rule_332_summary.json.'},
    },
    'per_chunk': per_chunk,
    'matches_in_both_fixed_and_remaining': sorted(fixed_m & rem_m),
    'test_batch_matches_among_45': test_m_cats,
    'test_batch_remaining_rounds': test_rounds_in_rem,
    'protected_batch_matches_among_45': protected_n,
    'dates_check': {'min': A['affected_dates_min_max'][0], 'max': A['affected_dates_min_max'][1], 'matches_dated_2026_10_21_to_31': len(A['affected_matches_dated_2026_10_21_to_31'])},
    'sources_among_45': A['affected_pos_source_counts'],
    'inputs_present': {'raw_cache_bucket_has_4_channels_for_matches': raw_ok, 'raw_cache_missing': raw_miss[:5],
                       'scratch_posb_all_seconds_has_4_channels_for_matches': scr_ok, 'of_matches': len(aff_m)},
    'hashes_sha256': hashes,
    'blockers': [
        {'id': 'K1', 'blocks': 'the 332 remaining rounds, not the 283',
         'item': 'Side rule for unrostered and duplicate rounds not decided by owner (readiness inventory B2; CHANGES_freeze_cexp freeze_tick_open).',
         'detail': 'Only the 283 fixed rounds have an agreed side map. The place rule gives 5v5 for %d rounds (category A, spawn labels only). It is not yet approved for the rest.' % cat_summary.get('A_place_5v5_spawn_labels', {}).get('rounds', 0)},
        {'id': 'K2', 'blocks': '%d rounds of the 332' % len(B_rounds),
         'item': 'Place label CTSideUpper is not in the CT rule (only CTSpawn is).',
         'detail': 'All %d rounds are in %s, rounds %s. Each shows 5 players at CTSideUpper and 5 at TSpawn, with 1 duplicate and 1 unrostered alive player. CTSideUpper as the CT spawn label is an inference to check against the map before use; a CTSpawn-only rule would leave 5 CT players unassigned.' % (len(B_rounds), B_match[0], ','.join(str(x) for x in B_rounds))},
        {'id': 'K3', 'blocks': '%d round of the 332' % len(C_rows),
         'item': 'Round %s, round %d: at freeze, 5 CTSpawn, 4 TSpawn and 1 alive player on TopofMid (places_raw %s).' % (C_rows[0]['match_key'], C_rows[0]['round'], C_rows[0]['places_raw']),
         'detail': 'Unrostered alive count 1, duplicate 0. The place rule cannot assign that player to a side. Owner decides exclude or manual check.'},
        {'id': 'K4', 'blocks': '%d rounds of the 332 (exclude or separate rule)' % len(D_rows),
         'item': 'Not 10 alive at freeze: %d rounds with 9 alive (split 5v4 or 4v5 by place) and %d round with 6 alive (5v1 by place).' % (len(D9), len(D6)),
         'detail': 'Nine-alive rounds by match: %s. The 6-alive round is %s round %d. A 5 CT and 5 T gate cannot hold. Includes csds/2026/06/28/m_e0992218ce75516acac42094/csds round 18 (the not_fixed round).' % (
             '; '.join('%s = %d' % (k, v) for k, v in D9_match_counts.most_common()), D6[0]['match_key'] if D6 else '', D6[0]['round'] if D6 else -1)},
        {'id': 'K5', 'blocks': 'nothing if followed; hazard',
         'item': 'Wrong feature module. posport/scripts/data/make_pos_features.py bomb lookup is a causal per-tick asof join (posport lines 96-98 and 172-175). The chunk build used fixes/pos_every_second/feat/make_pos_features.py, which takes the last plant per round (feat lines 96 and 170). The two copies have different sha256.',
         'detail': 'Using the posport copy would change bomb columns for rebuilt rounds only, giving mixed definitions in one chunk. Keep script header lines 4-5 say the feat copy is pinned.'},
        {'id': 'K6', 'blocks': 'nothing; data-quality caveat that needs a decision',
         'item': 'N2: odd-second cone columns (c_sees_enemy, t_sees_enemy, c_exposed, t_exposed) use the float32 cone test; the float64 fix covered even steps only.',
         'detail': 'Rebuilt rounds keep float32 odd steps unless the owner chooses float64. Plan: measure float32 against float64 on the odd steps of the 283 rounds during the parity run and report the count.'},
        {'id': 'K7', 'blocks': 'the v5 write, not the rebuild',
         'item': 'assemble_pos_v4.py cannot write v5 as is.',
         'detail': 'It asserts OUT contains export_clean_v4/pos_v4 (line 17) and that pos_all_seconds is not in OUT (line 18), and it globs bucket_c*.parquet from pos_all_seconds (lines 14 and 71). A v5 copy with CH and OUT changed is needed. The 39-cell float64 cone patch (fixes/cexp_f64/out/pos_v4_changes.csv) must be reapplied; it touches none of the 615 rounds (0 of 39).'},
        {'id': 'K8', 'blocks': 'nothing if configured; procedure',
         'item': 'Raw source. keep script fetch() (lines 70-75) reads SCR = /mnt/sagemaker-nvme/posb_all_seconds (line 37) and downloads to it on a miss.',
         'detail': 'The raw cache /home/sagemaker-user/adam/builder/raw_cache_bucket has all four channels for all %d matches (inputs_present). Run with POSB_SCR set to that folder so nothing is written to the cache.' % len(aff_m)},
    ],
    'owner_decisions_needed': [
        'D1: Approve the 283 rebuild now. Its side labels are the override map already used by loadout_round in export_clean_v4, so Phase 1 needs no new rule.',
        'D2: Side rule for the 332 (K1). Confirm the place rule for category A (spawn labels only). Decide the CTSideUpper label (K2) after a map check.',
        'D3: K3 round and K4 rounds: exclude, or handle separately.',
        'D4: float32 or float64 for odd cone steps of rebuilt rounds (K6).',
        'D5: Joint pos plus loadout use on the protected batch stays blocked until B1 is cleared. None of the 45 matches are in v4_s3_protected, so this plan does not change B1.',
    ],
    'steps': [
        'Phase 1 (283 rounds, 23 matches, chunks %s): rerun the every-second pipeline on those 23 matches, with POSB_FEAT pointing to fixes/pos_every_second/feat and POSB_SCR to the raw cache. Keep all rounds of those matches, take the 283 target rounds from the new run, and check every other round against the existing chunk rows by value (parity, expect 0 mismatches).' % ', '.join(phase1_chunks),
        'Phase 1 side: replace the roster side with the override map for the 283 rounds (status fixed in override_rounds.csv). Keep m_e0992218 round 18 (not_fixed) on the roster rule.',
        'Write rewritten chunk files to a new folder (proposed: final/pos_all_seconds_v5/, 47 chunk files: the Phase 1 chunks rewritten, the rest copied). Never edit pos_all_seconds in place.',
        'Assemble v5 with a copy of assemble_pos_v4.py whose CH and OUT point to v5 paths and whose two assert lines are changed for v5. Reapply the 39-cell float64 cone patch. Write final/export_clean_v5/pos_v5/ (proposed), not pos_v4.',
        'Validation: (a) changed steps only in target rounds (at most 20,125 for the 283); (b) 5v5 gate holds for all 283 by override; (c) sentinel counts per source unchanged outside target rounds; (d) per-column change counts reported.',
        'Phase 2 (332) only after D2 to D4. Same procedure on the 23 Phase 2 matches with the decided rule. Chunk c029 holds one remaining round.',
    ],
    'output_plan': {
        'pos_v5_dir': '/home/sagemaker-user/adam/builder/final/export_clean_v5/pos_v5/ (proposed, not created)',
        'chunk_dir': '/home/sagemaker-user/adam/builder/final/pos_all_seconds_v5/ (proposed, not created)',
        'files': ['pos_int8.npy', 'pos_uint16.npy', 'pos_uint8.npy', 'pos_float16.npy', 'pos_offsets.npy', 'match_pos_source.npy', 'every_second_match_keys.json', 'README.json', 'CHANGES_side_v5.json'],
        'layout_unchanged': 'Same layout as pos_v4: 72,567,700 steps, same column order, same sentinels.',
        'not_changed': 'loadout arrays in export_clean_v4, pos_v4, and all earlier exports.',
    },
    'evidence': [
        'fixes/pos_every_second/pos_bucket_every_second_keep.py: lines 29-31 (feat copy import), 109-110 (roster keep last), 123-125 (roster join, coalesce with player_vector team_code, side 2 or 3 filter), 70-75 (fetch), 136 (run_chunk; matches from data/cNNN/matches.parquet).',
        'fixes/pos_every_second/feat/make_pos_features.py: lines 74-105 (feats_for_day; side split 89-93; slot cap 90), 106-150 (feature bodies), 96 and 170 (bomb lookup, last plant per round).',
        'posport/scripts/data/make_pos_features.py: lines 96-98 and 172-175 (causal per-tick bomb lookup).',
        'final/export_clean_v4/pos_v4/README.json: known_issues (freeze roster fix not applied to positions); layout (24 columns).',
        'final/export_clean_v4/CHANGES_freeze_cexp.json: freeze_tick_open (unro_only 302; duplicate-only 30; pos features not fixed).',
        'final/export_clean_v4/CHANGES_freeze_side.json: rounds_fixed 283; rounds_not_fixed_listed 1 (m_e0992218 round 18).',
        'fixes/freeze_fix/override_rounds.csv: 284 rows, 283 fixed, 1 not_fixed; override column covers player_id 0..9 per round.',
        'fixes/freeze_scope/scope_summary.json: affected_rounds 615, affected_matches 45, rounds_alive_not_10_and_affected 16.',
        'fixes/fix332/analysis.json: chunk, row, step, source, date and batch counts used above.',
        'fixes/fix332/side_rule_332_check.csv and side_rule_332_summary.json: place-rule gate on the 332 rounds.',
        'fixes/freeze_fix/rule_test.log: place rule gives 5v5 in 283 of 283 fixed rounds; player_vector and spawn rules give 0 for the unrostered player.',
        'final/pos_all_seconds/build.log (c000, 97 s) and fixes/pos_every_second/keep_run.log (c001 to c046, total 4,839 s).',
        'final/pos_all_seconds/bucket_c000.parquet: 646,695 rows, 28 columns, no side or player column.',
        'assemble_v4/assemble_pos_v4.py: lines 14, 16-18, 71, 245-254; assemble run log total 57 s.',
    ],
    'blockers_summary': 'Phase 1 (283 rounds) has no unresolved technical blocker and can run on owner approval D1. Phase 2 (332 rounds) is blocked by K1 to K4: the owner side rule, the CTSideUpper label, one player at TopofMid, and 16 rounds that are not 10 alive.',
}

out = FX + 'pos_plan.json'
json.dump(plan, open(out, 'w'), indent=1, default=str)
print('WROTE', out)
print('phase1 chunks', phase1_chunks, 'fixed matches by chunk', dict(fixed_by_chunk))
print('phase2 chunks', phase2_chunks)
print('cats', {k: (v['rounds'], v['matches']) for k, v in cat_summary.items()})
print('B rounds', B_rounds, B_match)
print('D9 by match', dict(D9_match_counts))
print('raw ok', raw_ok, 'scr ok', scr_ok, 'missing', raw_miss[:3])
print('test', test_m_cats, 'test rounds in rem', test_rounds_in_rem, 'protected', protected_n)
print('phase1 secs full', phase1_secs_full, 'match-only', phase1_match_only)
print('per_chunk', json.dumps(per_chunk))
print('hashes', hashes)
