"""Known-answer unit tests of wp_common (seconds): the log loss clip (ll_vec), the match-cluster bootstrap interval, the monotone-constraint pins and the
documented sizes of the legacy random split. They kill the faults  C1 (alive_t constraint sign), C2 (validation / test swapped), C3 (loose log-loss clip) and
C4 (interval at the quartiles) without any data."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not put the script folder on sys.path
from harness import T  # noqa: E402

from checks import unit  # noqa: E402
import wp_common as W  # noqa: E402

t = T("wp_common known answers")
for name, res in unit.run_suite(unit.SUITE_WP, W):
    t.res(name, res)
t.finish()
