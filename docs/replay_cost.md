Same 36 recorded conversations (1,847 agent steps), replayed under each policy. Prefill time estimated at the measured T4 rate of 1,281 tokens/s (+158 ms per step).

| Policy | Tokens in prompts | Tokens computed (prefill) | vs. no pruning | Est. prefill time | Mean peak context |
|---|---|---|---|---|---|
| No pruning + prefix cache | 27,067,008 | 320,162 | 1.00x | 543 s | 14,336 |
| Pruning (k=1) + prefix cache | 13,769,130 | 965,414 | 3.02x | 1,046 s | 7,716 |
| Pruning (k=1) + suffix reuse | 13,769,130 | 319,415 | 1.00x | 542 s | 7,716 |
