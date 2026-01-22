# Worklog

## 2026-01-22
- Start logging here per user request (stop writing to ~/code/ziliang).
- Current best: 1776 cycles (submission_tests), achieved by replacing `cond + 1` with flow `vselect(cond, 2, 1)` before idx multiply_add.
- Remaining issues: speed thresholds in `tests/submission_tests.py` still fail (need <1790 for casual, <1579, <1548, <1487, <1363).
- Snapshot saved: `kernel_snapshots/build_kernel_2026-01-22_1348.py`.
