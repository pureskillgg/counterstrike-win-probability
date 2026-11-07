#!/bin/bash
# Per-match raw files for laptop-era matches that are not in the S3 tome chunks.
# Downloads only player_info and player_status (the two tables the freeze-tick roster test reads).
set -u
B=s3://pureskillgg-csgo-production-sagemaker-0/data
W=/home/sagemaker-user/adam/builder/fixes/fix332
R=$W/raw
mkdir -p "$R"
: > "$W/dl_jobs.txt"
while IFS=$'\t' read -r folder name tag; do
  mkdir -p "$R/$name"
  for f in player_info player_status; do
    echo "$B/$folder/$f|$R/$name/$f" >> "$W/dl_jobs.txt"
  done
done < "$W/dl_list.tsv"
echo "jobs $(wc -l < "$W/dl_jobs.txt")"
xargs -P 24 -n 1 sh -c 'src="${0%%|*}"; dst="${0#*|}"; if [ ! -s "$dst" ]; then aws s3 cp --only-show-errors "$src" "$dst" || echo "FAILED $src"; fi' < "$W/dl_jobs.txt"
echo "download pass finished"
find "$R" -type f | wc -l
