# Worklog

## 2026-01-22
- Start logging here per user request (stop writing to ~/code/ziliang).
- Current best: 1776 cycles (submission_tests), achieved by replacing `cond + 1` with flow `vselect(cond, 2, 1)` before idx multiply_add.
- Remaining issues: speed thresholds in `tests/submission_tests.py` still fail (need <1790 for casual, <1579, <1548, <1487, <1363).
- Snapshot saved: `kernel_snapshots/build_kernel_2026-01-22_1348.py`.

### TODO ideas
- Cache depth-3 nodes (forest[7..14]) and compute node values in depth-2 round to avoid load_offset for depth==3.
- If depth-3 caching helps, evaluate depth-4 caching (forest[15..30]).
- Use per-queue cond_vec temp to avoid recomputing `% 2` in depth==1 path.
- Tune scheduler to overlap load/valu more aggressively (keep one load per cycle and fill remaining valu slots from other queues).
- Re-tune group_size after any new temps to keep packing efficient.

### Attempted (regressed)
- Depth-3 caching with extra temps and smaller group_size (cycles 1986); reverted.
- Per-queue cond_vec and reduced group_size (cycles 1945); reverted.
- group_size=24/28 tuning (cycles 1829/1855); reverted.
- Depth0 flow vselect (forest1/2) instead of multiply_add (cycles 1868); reverted.
- 2026-01-22: Replaced wrap-round `idx *= 0` valu op with flow vselect using `zero_vec`; cycles improved to 1772.
- Snapshot saved: `kernel_snapshots/build_kernel_2026-01-22_1401.py`.
- 2026-01-22: Scheduler tweak (higher addr_budget when load_supply low) improved cycles to 1750.
- Snapshot saved: `kernel_snapshots/build_kernel_2026-01-22_1406.py`.
- 2026-01-22: Tried more aggressive addr_budget (4/3/2); no cycle improvement vs 1750, reverted.
- 2026-01-22: Tried need_addr threshold <=8; cycles regressed to 1754, reverted to <=6.
- 2026-01-22: Tried addr_budget threshold <=5 (instead of <=4); no cycle improvement vs 1750, reverted.
- 2026-01-22: Tried scheduling load before valu in scheduler; no improvement vs 1750, reverted.
- 2026-01-22: Depth1 path: kept cond recompute (due to cond_vec reuse) but replaced `+3` with flow vselect (cond?4:3) using new four_vec; cycles improved to 1733.
- 2026-01-22: Attempted to remove cond recompute entirely; broke correctness, reverted.
- 2026-01-22: Tried `group_size = min(chunk_count, 16)`; cycles regressed to 1813, reverted.
- Snapshot saved: `kernel_snapshots/build_kernel_2026-01-22_1441.py`.
- 2026-01-22: Dropped idx vload/vstore + idx_ptr consts, packed value vloads (2 per load instr), and set idx on depth0 via flow vselect; cycles improved to 1724.
- Snapshot saved: `kernel_snapshots/build_kernel_2026-01-22_1515.py`.
- 2026-01-22: Moved value vloads into per-queue ops and allowed packing multiple load ops per cycle; cycles improved to 1708.
- Snapshot saved: `kernel_snapshots/build_kernel_2026-01-22_1522.py`.
- 2026-01-22: Tried `group_size = min(chunk_count, 24)` with in-queue vloads; cycles regressed to 1757, reverted.
- 2026-01-22: Scheduler tweak `need_addr <= 8` improved cycles to 1701.
- 2026-01-22: Scheduler tweak `need_addr <= 10` improved cycles to 1695 (best so far).
- Snapshot saved: `kernel_snapshots/build_kernel_2026-01-22_1529.py`.
- 2026-01-22: Tried `need_addr <= 11/12`; cycles regressed to 1700/1697, reverted to 10.
- 2026-01-22: Tried addr_budget 4/3/1 with `need_addr <= 10`; no cycle change vs 1695, reverted.
- 2026-01-22: Tried `need_addr <= 9`; cycles regressed to 1699, reverted to 10.
- 2026-01-22: Retried depth0 flow vselect (forest1/2) + removed forest_diff_vec; cycles regressed to 1721, reverted.
- 2026-01-22: Scheduler load selection now prioritizes queues with smaller remaining load blocks (finishes load blocks to desync phases); cycles improved to 1596.
- Snapshot saved: `kernel_snapshots/build_kernel_2026-01-22_1611.py`.
