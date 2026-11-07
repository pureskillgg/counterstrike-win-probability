#!/bin/bash
cmd.exe /c 'robocopy "G:\pureskill\csds-cs2" "C:\Users\<user>\Backups\csds-cs2_2026-11-06" /E /XJ /COPY:DAT /R:2 /W:5 /NP /LOG+:"C:\Users\<user>\Backups\csds-cs2_2026-11-06_robocopy.log"'
echo "MAIN_EXIT=$?"
cmd.exe /c 'robocopy "C:\Users\<user>\Documents\pureskill_internship\csds-cs2\data" "C:\Users\<user>\Backups\csds-cs2_2026-11-06_target_data" /E /XJ /COPY:DAT /R:2 /W:5 /NP /LOG+:"C:\Users\<user>\Backups\csds-cs2_2026-11-06_target_data_robocopy.log"'
echo "DATA_EXIT=$?"
cmd.exe /c 'robocopy "C:\Users\<user>\Documents\pureskill_internship\csds-cs2\tomes" "C:\Users\<user>\Backups\csds-cs2_2026-11-06_target_tomes" /E /XJ /COPY:DAT /R:2 /W:5 /NP /LOG+:"C:\Users\<user>\Backups\csds-cs2_2026-11-06_target_tomes_robocopy.log"'
echo "TOMES_EXIT=$?"
echo "ALL_DONE"
