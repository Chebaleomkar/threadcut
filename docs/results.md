## Agent benchmark (totals)

| Mode | Tasks passed | Agent steps | Tokens in prompts | Tokens computed (prefill) | Reused from cache | Prefill time (s) | Decode time (s) | Wall time (s) | Peak context |
|---|---|---|---|---|---|---|---|---|---|
| No pruning | 2/6 | 290 | 3,046,827 | 37,126 | 99% | 70 | 2536 | 2637 | 25,952 |
| Pruning + prefix cache | 2/6 | 457 | 3,249,260 | 250,479 | 92% | 264 | 3602 | 3974 | 14,116 |
| Pruning + suffix reuse | 2/6 | 428 | 3,140,243 | 69,031 | 98% | 107 | 3756 | 3936 | 15,134 |

## Per task

| Task | No pruning | Pruning + prefix cache | Pruning + suffix reuse |
|---|---|---|---|
| csvpipe | fail (timeout), 150 steps, 15,661 computed, looped last 67 | fail (timeout), 95 steps, 44,422 computed | fail (timeout), 109 steps, 11,277 computed |
| inventory | pass, 13 steps, 3,425 computed | fail (timeout), 116 steps, 72,873 computed | pass, 27 steps, 6,499 computed |
| ledger | fail, 28 steps, 4,935 computed | pass, 34 steps, 19,470 computed | fail (timeout), 122 steps, 18,512 computed |
| life | fail, 12 steps, 2,722 computed | pass, 18 steps, 8,729 computed | pass, 18 steps, 4,009 computed |
| schedule | pass, 9 steps, 2,568 computed | fail (timeout), 89 steps, 59,396 computed | fail (timeout), 47 steps, 8,717 computed |
| textstats | fail (timeout), 78 steps, 7,815 computed | fail (timeout), 105 steps, 45,589 computed | fail (timeout), 105 steps, 20,017 computed |

## What the reused suffix remembers

| Context after a prune | Mean KL to unpruned model | Median KL | Top-1 agreement with unpruned |
|---|---|---|---|
| Recomputed from scratch (prefix cache) | 0.3380 | 0.1902 | 90.0% |
| Spliced cache (suffix reuse) | 0.2943 | 0.1710 | 90.9% |

Suffix reuse was closer to the unpruned model on 50/75 pruned steps (3,950 reply tokens scored, 6 runs, 0 repeated loop steps removed).

Paired difference KL(fresh) - KL(surgery): mean +0.0437, 95% bootstrap CI [+0.0025, +0.0915]; sign test 50/75, p = 0.00523.
