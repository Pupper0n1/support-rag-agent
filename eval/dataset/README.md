# Evaluation dataset

`tickets.jsonl` - 150 support tickets, 30 templates x 5 fills, across five
product areas. Regenerate byte-for-byte with:

    python eval/dataset/generate_tickets.py

Each ticket carries:

| field | meaning |
|---|---|
| `relevant_docs` | KB `doc_id`s a human marked as answering the ticket (gold set for recall/MRR) |
| `expected_route` | `auto_reply` / `escalate` / `needs_info` - the route a human agent would take |

`human_grades.jsonl` - graded relevance for (ticket, doc) pairs the retriever
actually surfaced, on a 0-3 scale (0 off-topic, 1 mentions topic, 2 partially
answers, 3 answers). Collected with `python eval/grade.py <report.json>`; nDCG
uses these and falls back to binary gold labels for ungraded pairs.
