#!/bin/bash
cd /c/Users/<user>/Documents/pureskill/counterstrike-win-probability
S=/c/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad
out=$S/hist_blob_list.txt
: > $out
while read p; do
  git log --all --format='%H' -- "$p" | while read c; do
    echo -e "$c\t$p" >> $out
  done
done < $S/hist_data_paths.txt
wc -l < $out
