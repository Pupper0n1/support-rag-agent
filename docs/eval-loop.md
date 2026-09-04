# Eval loop

How a retrieval or prompt change is checked before it lands.

1. **Baseline.** `make eval-retrieval NAME=before` (seconds, no model calls).
   For prompt or tool-description changes, `make eval-full NAME=before`.
2. **Change one thing.** A fusion parameter, the rerank depth, a tool
   docstring, a system prompt paragraph.
3. **Re-run** under a new name. Compare aggregates, then open the confusion
   table. Routing errors cluster by expected route; `escalate -> auto_reply`
   is the expensive direction (a wrong answer reaches a customer) and is
   weighted accordingly when deciding whether a change ships.
4. **Grade the new surface.** `make grade NAME=after` shows only pairs not
   yet graded, so the graded set grows with what the retriever actually
   returns instead of what someone guessed it would return.
5. **Trace regressions to a ticket.** Per-ticket rows carry retrieved ids
   and per-stage scores (`bm25_rank`, `knn_rank`, `fused_score`,
   `rerank_score` are all kept on the chunk). A ticket that lost its gold
   doc between runs tells you which stage dropped it.
6. **Pull real traffic in.** Conversation logs in S3 use the same record
   shape the harness reads; a day's prefix can be sampled, graded, and
   appended to the eval set.

## Things that looked like improvements and were not

- Passing `product_area` from the routing form as a hard ES filter raised
  recall on correctly-tagged tickets and zeroed it on mis-tagged ones. Kept
  as an opt-in flag (`--use-product-area`), off by default.
- Increasing `top_k` beyond 6 raised recall@k trivially and lowered routing
  accuracy: the model started citing tangential articles and choosing
  `draft_reply` on tickets that should have escalated.
- Score interpolation instead of RRF. See the fusion module docstring.
