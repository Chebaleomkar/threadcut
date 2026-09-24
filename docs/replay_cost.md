Same 18 recorded conversations (664 agent steps), replayed under each policy. Prefill time estimated at the measured T4 rate of 1,203 tokens/s (+170 ms per step).

| Policy | Tokens in prompts | Tokens computed (prefill) | vs. no pruning | Est. prefill time | Mean peak context |
|---|---|---|---|---|---|
| No pruning + prefix cache | 11,774,948 | 132,459 | 1.00x | 223 s | 11,021 |
| Pruning (k=1) + prefix cache | 5,447,634 | 361,331 | 2.73x | 414 s | 5,745 |
| Pruning (k=1) + suffix reuse | 5,447,634 | 132,459 | 1.00x | 223 s | 5,745 |
