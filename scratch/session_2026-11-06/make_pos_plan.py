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
rem_m = set(aff.join(fx, on=['match_key', 'round'], how='anti')['match_key'].to_list())

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
    cat_summary[k] = {
        'rounds': len(v),
        'matches': len(set(x['match_key'] for x in v)),
        'match_keys': sorted(set(x['match_key'] for x in v)) if len(set(x['match_key'] for x in v)) <= 5 else None,
        'alive_n_hist': dict(collections.Counter(str(x['alive_n_scope']) for x in v)),
        'places_seen': sorted(set(p for x in v for p in json.loads(x['places_raw']).keys())),
    }
    if k == 'D_not_10_alive' or k == 'C_player_off_spawn_label':
        cat_summary[k]['rounds_list'] = [[x['match_key'], x['round']] for x in v]

# per-chunk for the 5 affected chunks
CH5 = ['c029', 'c043', 'c044', 'c045', 'c046']
chunk_of = {}
for c in CH5:
    mk = pl.read_parquet(B + 'data/%s/matches.parquet' % c, columns=['match_key'])['match_key'].to_list()
    for m in mk:
        if m in aff_m:
            chunk_of[m] = c

# seconds from logs
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
        'matches_fixed_283': len([m for m in ms if m in fixed_m]),
        'matches_remaining_332_only': len([m for m in ms if m in rem_m and m not in fixed_m]),
        'matches_mixed_fixed_and_remaining': len([m for m in ms if m in rem_m and m in fixed_m]),
        'rounds_fixed_283': A['rows_by_affected_chunk'][c]['rounds_fixed_with_rows'],
        'rounds_remaining_332': A['rows_by_affected_chunk'][c]['rounds_rem_with_rows'],
        'rows_in_chunk_file': A['rows_by_affected_chunk'][c]['rows_total'],
        'rows_in_affected_matches': A['rows_by_affected_chunk'][c]['rows_affected_matches'],
        'logged_secs_full_chunk_rerun': chunk_secs.get(c, [0, 0])[1],
        'logged_matches_full_chunk_rerun': chunk_secs.get(c, [0, 0])[0],
    }

fixed_matches_by_chunk = collections.Counter(chunk_of[m] for m in fixed_m if m in chunk_of)
phase1_chunks = sorted(fixed_matches_by_chunk.keys())
phase1_secs_full = sum(chunk_secs[c][1] for c in phase1_chunks if c in chunk_secs)
spm = A['secs_per_match']
phase1_match_only = round(spm * len(fixed_m), 1)
phase2_chunks = sorted(set(chunk_of[m] for m in rem_m if m in chunk_of))

# raw and scratch presence for needed channels
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

# test-split and protected batch membership among the 45
ls = pl.read_csv(B + 'final/export_clean_v4/match_loadout_status.csv', infer_schema_length=0)
lsa = ls.filter(pl.col('match_key').is_in(sorted(aff_m)))
test_m = sorted(set(lsa.filter(pl.col('v4_batch') == 'v4_s3_test')['match_key'].to_list()))
test_m_cats = {m: ('fixed' if m in fixed_m else 'remaining_only') for m in test_m}

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
    'status': 'PLAN ONLY. Nothing built. Writes limited to fixes/fix332/. pos_v4, export_clean_v2, export_clean_v3_loadout and export_clean_v4 untouched. Holdout B and 2026-10-26..31 keys were not read (date check on match_key strings only).',
    'candidates_csv': 'fixes/fix332/candidates.csv does not exist, so no rounds were added from it. The 332 list is in fixes/fix332/side_rule_332_check.csv (from freeze_scope/scope_rounds.parquet, affected minus the 283 fixed).',
    'answers': {
        'which_feature_columns_depend_on_side': 'All 24 pos columns. CT and T sets come from side 3 and side 2 in feats_for_day (make_pos_features.py:88-93) and every column is computed from one of those sets: sites (site_hi/lo, nearest_site_mean, spread), blindness (blind_n, blind_sum), view (sees_enemy, exposed), contact, centroid and min_enemy_dist, and the bomb columns (c_min_bomb_dist, c_n_bomb_500, defuse_slack for CT; t_min_bomb_dist for T). Not side-dependent inputs: bomb position bx/by and seconds_since_plant, map sites, the tick grid. The step set also depends on side only through the side 2/3 filter at pos_bucket_every_second_keep.py:125.',
        'chunk_files_holding_rounds': {'phase1_283_rounds': phase1_chunks, 'all_332_plus_283_rounds': CH5, 'count_all': len(CH5)},
        'per_round_patch_feasible': 'Yes at round level, not as an arithmetic patch. Bucket chunk rows are keyed by (match_key, round, second) with no duplicates (pos_v4 README counts), so rows of the target rounds can be dropped and recomputed. The stored chunk rows have no side or player column, so a patched value cannot be derived from them; the rows must be recomputed from raw. Chunk parquet files are rewritten as new v5 files, not edited in place. Non-target rounds are copied unchanged and checked by value.',
        'runtime': {
            'keep_log_secs_per_match': spm,
            'keep_log_basis': 'fixes/pos_every_second/keep_run.log and final/pos_all_seconds/build.log: 47 chunks, 32,065 matches, 4,933 s total (both parities, 16 threads).',
            'phase1_283_compute_matches_only_sec': phase1_match_only,
            'phase1_283_full_chunk_rerun_sec_from_log': phase1_secs_full,
            'assemble_pos_v4_sec': 57,
            'note': 'Pipeline cost is per match, so a match-level run with a target-round filter is much cheaper than the full-chunk rerun. Setup, S3 or cache reads and the parity check add time. If a run exceeds about 100 s, use nohup with a log and poll in short calls.',
        },
    },
    'scope': {
        'fixed_283': {'rounds': 283, 'matches': len(fixed_m), 'steps': A['steps_fixed_rounds'], 'chunks': phase1_chunks,
                      'side_source': 'fixes/freeze_fix/override_rounds.csv column override (complete map of player_id 0..9 to side 2 or 3 per round, 5 CT and 5 T in all 283; gate check passed 283/283 in rule_test.log)'},
        'remaining_332': {'rounds': A['affected_minus_fixed'], 'matches': len(rem_m), 'steps': A['steps_rem_rounds'], 'chunks': phase2_chunks,
                          'breakdown': {'unro_only': A['rem_unro_only'], 'dup_and_unro': A['rem_dup_and_unro'], 'dup_only': A['rem_dup_only']},
                          'categories': cat_summary,
                          'side_rule_status': 'Not decided. Owner side rule pending (readiness/inventory.json B2). Place-based check in fixes/fix332/side_rule_332_check.csv.'},
    },
    'per_chunk': per_chunk,
    'matches_in_both_fixed_and_remaining': sorted(fixed_m & rem_m),
    'test_batch_matches_among_45': test_m_cats,
    'protected_batch_matches_among_45': 0,
    'dates_check': {'min': A['affected_dates_min_max'][0], 'max': A['affected_dates_min_max'][1], 'in_2026_10_21_to_31': 0},
    'sources_among_45': A['affected_pos_source_counts'],
    'inputs_present': {'raw_cache_bucket_all_4_channels_for_45': raw_ok, 'raw_cache_missing': raw_miss[:5],
                       'scratch_posb_all_seconds_all_4_channels_for_45': scr_ok},
    'hashes': hashes,
    'blockers': [
        {'id': 'K1', 'severity': 'blocks the 332, not the 283',
         'item': 'Side rule for unrostered and duplicate rounds not decided by owner (readiness inventory B2, CHANGES_freeze_cexp freeze_tick_open).',
         'detail': 'Only the 283 fixed rounds have an agreed side map. The place rule would give 5v5 for the 296 rounds in category A only (see categories).'},
        {'id': 'K2', 'severity': 'blocks 19 rounds of the 332',
         'item': 'Place label CTSideUpper is not in the rule (CTSpawn to CT only).',
         'detail': 'All 19 rounds are match csds/2026/07/03/e16fd82b-5eba-4907-a435-505d5a695e0d/csds, rounds 2 to 23 minus some. Five CT players sit on CTSideUpper and five T on TSpawn. A rule that only maps CTSpawn would leave 5 CT players unassigned. Needs a map-label decision and a check on that map.'},
        {'id': 'K3', 'severity': 'blocks 1 round of the 332',
         'item': 'Round csds/2026/06/26/4b4cc794-67d7-4b6c-b68a-3d671f68d877/csds round 15 has 5 CTSpawn, 4 TSpawn and 1 player on TopofMid at freeze (the unrostered one).',
         'detail': 'The place rule cannot assign that player to a side.'},
        {'id': 'K4', 'severity': 'blocks 16 rounds of the 332 (exclude or separate rule)',
         'item': 'Non-10-alive freeze rounds: 15 with 9 alive (5v4 by place), 1 with 6 alive (5v1 by place) (includes 2d963086 round 18, the not_fixed round).',
         'detail': 'A 5 CT and 5 T gate cannot hold. Owner decides exclude or a separate rule. The 9-alive rounds are mostly in match csds/2026/06/27/659cf41b-53a8-43ef-a1a1-5739eef27bc4/csds (12 rounds).'},
        {'id': 'K5', 'severity': 'hazard, not a blocker if followed',
         'item': 'Wrong feature module. posport/scripts/data/make_pos_features.py differs from the chunk build copy (fixes/pos_every_second/feat/make_pos_features.py) at bomb lookup: causal per-tick asof join (posport lines 96-98) versus last plant per round (feat lines 96 and 170).',
         'detail': 'Rebuilt rounds must use the feat copy, or bomb columns get mixed definitions within a chunk. Keep script header lines 4-5 say the same.'},
        {'id': 'K6', 'severity': 'data-quality caveat, decision needed',
         'item': 'N2: odd-second cone columns (c_sees_enemy, t_sees_enemy, c_exposed, t_exposed) use the float32 cone test. The float64 fix covered even steps only.',
         'detail': 'Rebuilt rounds keep float32 odd steps unless the owner chooses float64. Plan: measure float32 versus float64 on the odd steps of the 283 rounds during the parity run and report the count.'},
        {'id': 'K7', 'severity': 'procedure',
         'item': 'assemble_pos_v4.py cannot write v5 as is.',
         'detail': 'It asserts OUT contains export_clean_v4/pos_v4 (line 17) and forbids pos_all_seconds (line 18), and it globs bucket_c*.parquet from pos_all_seconds (line 14 and line 71). A v5 copy is needed with CH and OUT changed, plus the same 39-cell float64 cone patch. The cone CSV does not touch any of the 615 rounds (0 of 39), but the patch is applied by cell, so it must be reapplied.'},
        {'id': 'K8', 'severity': 'procedure',
         'item': 'Raw source. keep script fetch() (pos_bucket_every_second_keep.py:70-75) reads from SCR = /mnt/sagemaker-nvme/posb_all_seconds and downloads to it on a miss.',
         'detail': 'The raw cache /home/sagemaker-user/adam/builder/raw_cache_bucket has all four channels for all 45 matches (see inputs_present). Run with POSB_SCR set to that path so no write goes to the cache.'},
    ],
    'owner_decisions_needed': [
        'D1: Approve the 283 rebuild now. Side labels are the override map already used by loadout_round in export_clean_v4.',
        'D2: Side rule for the 332 (K1). Confirm the place rule for category A, with CTSpawn and TSpawn only, and decide the CTSideUpper label (K2).',
        'D3: K3 round and K4 rounds: exclude or handle separately.',
        'D4: float32 versus float64 for odd cone steps of rebuilt rounds (K6).',
        'D5: Joint pos plus loadout use stays blocked until B1 is resolved for the protected batch. None of the 45 matches are in v4_s3_protected, so the 283 rebuild does not change B1.',
    ],
    'steps': [
        'Phase 1 (283 rounds, 23 matches, chunks %s): rerun the every-second pipeline on those 23 matches with POSB_SCR set to the raw cache and POSB_FEAT set to fixes/pos_every_second/feat. Keep all rounds of the matches, take the 283 target rounds from the new run, and check that the other rounds match the existing chunk rows by value (parity, expect 0 mismatches).' % (', '.join(phase1_chunks)),
        'Phase 1 side: replace the roster side with the override map (override_rounds.csv, status fixed) for the 283 rounds. Keep the one not_fixed round (2d963086 round 18) on the roster rule.',
        'Write new chunk files to a new folder (proposed: final/pos_all_seconds_v5/, 47 chunk files, the 4 Phase 1 chunks rewritten, the rest copied). Never edit pos_all_seconds in place.',
        'Assemble v5 with a copy of assemble_pos_v4.py whose CH and OUT are the v5 paths and whose assert lines are changed for v5. Re-apply the 39-cell float64 cone patch. Write final/export_clean_v5/pos_v5/ (proposed), not pos_v4.',
        'Validation: (a) changed steps only in target rounds (expect at most 20,125 steps for the 283 rounds); (b) 5v5 gate holds for all 283 by override; (c) sentinel counts per source unchanged outside the target rounds; (d) per-column change counts reported.',
        'Phase 2 (332) only after D2 to D4. Use the same procedure on the 23 matches of Phase 2 with the decided rule. Chunk c029 holds one remaining round.',
    ],
    'output_plan': {
        'pos_v5_dir': '/home/sagemaker-user/adam/builder/final/export_clean_v5/pos_v5/ (proposed, not created)',
        'chunk_dir': '/home/sagemaker-user/adam/builder/final/pos_all_seconds_v5/ (proposed, not created)',
        'files': ['pos_int8.npy', 'pos_uint16.npy', 'pos_uint8.npy', 'pos_float16.npy', 'pos_offsets.npy', 'match_pos_source.npy', 'every_second_match_keys.json', 'README.json', 'CHANGES_side_v5.json'],
        'layout_unchanged': 'Same layout as pos_v4 (72,567,700 steps, the same column order and sentinels).',
        'not_changed': 'loadout arrays (export_clean_v4), pos_v4 and all earlier exports.',
    },
    'evidence': [
        'fixes/pos_every_second/pos_bucket_every_second_keep.py: lines 29-31 (feat import), 109-110 (roster keep last), 122-125 (roster join, coalesce with player_vector team_code, filter side 2 or 3), 136-150 (run_chunk; matches from data/cNNN/matches.parquet).',
        'fixes/pos_every_second/feat/make_pos_features.py: lines 74-105 (feats_for_day side split at 92-93, slot cap at 90), 106-150 (feature bodies), 170-175 (bomb lookup in posport copy differs).',
        'posport/scripts/data/make_pos_features.py: lines 96-98 and 170-175 (causal per-tick bomb lookup).',
        'final/export_clean_v4/pos_v4/README.json: known_issues (freeze roster fix not applied to positions), layout (24 columns).',
        'final/export_clean_v4/CHANGES_freeze_cexp.json: freeze_tick_open (unro_only 302; dup_rounds_other 30; pos features not fixed).',
        'final/export_clean_v4/CHANGES_freeze_side.json: rounds_fixed 283, rounds_not_fixed_listed 1 (2d963086 round 18).',
        'fixes/freeze_fix/override_rounds.csv: 284 rows, 283 fixed, 1 not_fixed; override column covers player_id 0..9 per round.',
        'fixes/freeze_scope/scope_summary.json: affected_rounds 615, affected_matches 45, rounds_alive_not_10_and_affected 16.',
        'fixes/fix332/analysis.json: counts used above (chunks, rows, steps, sources, dates, batches).',
        'fixes/fix332/side_rule_332_check.csv and side_rule_332_summary.json: place-rule gate on the 332 rounds (296 pass, 19 CTSideUpper, 1 TopofMid, 16 not 10-alive).',
        'fixes/freeze_fix/rule_test.log: place rule 5v5 in 283 of 283 fixed rounds; the other rules fail the 5v5 gate.',
        'final/pos_all_seconds/build.log and fixes/pos_every_second/keep_run.log: per-chunk seconds (c000 97 s; full run 4,839 s incl. cached c000).',
        'final/pos_all_seconds/bucket_c000.parquet: 646,695 rows x 28 columns; no side or player column.',
        'assemble_v4/assemble_pos_v4.py: lines 14, 16-18, 71, 245-254; assemble run log total 57 s.',
    ],
    'blockers_summary': 'Phase 1 (283 rounds) is executable on owner approval D1 with no unresolved technical blocker. Phase 2 (332 rounds) is blocked by K1 to K4 (owner side rule and 20 rounds that the place rule cannot assign or that are not 10-alive).',
}

out = FX + 'pos_plan.json'
json.dump(plan, open(out, 'w'), indent=1, default=str)
print('WROTE', out)
print(json.dumps({k: plan[k] for k in ['answers']}, indent=1, default=str)[:3000])
print('phase1 chunks', phase1_chunks, 'fixed matches by chunk', dict(fixed_matches_by_chunk))
print('phase2 chunks', phase2_chunks)
print('cat summary', {k: (v['rounds'], v['matches']) for k, v in cat_summary.items()})
print('raw ok', raw_ok, 'scr ok', scr_ok, 'test matches', test_m_cats)
print('phase1 secs full', phase1_secs_full, 'match-only', phase1_match_only)
print('hashes', hashes)
