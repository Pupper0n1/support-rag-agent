"""Interactive relevance grading for surfaced (ticket, doc) pairs.

    python eval/grade.py eval/reports/baseline.json [--area billing]

Walks each ticket in a report, shows the ticket and each retrieved chunk,
and asks for a 0-3 grade. Pairs already graded are skipped so grading can
be resumed. Appends to eval/dataset/human_grades.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from eval.metrics import base_doc_id
from eval.run_eval import GRADES, Report, load_grades

KB = Path(__file__).parent.parent / "data" / "kb" / "knowledge_base.jsonl"

SCALE = "0 off-topic  1 mentions topic  2 partially answers  3 answers  (s skip, q quit)"


class KBArticle(TypedDict):
    doc_id: str
    title: str
    text: str


def load_kb() -> dict[str, KBArticle]:
    out: dict[str, KBArticle] = {}
    with KB.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                a: KBArticle = json.loads(line)
                out[a["doc_id"]] = a
    return out


def load_ticket_text() -> dict[str, str]:
    path = Path(__file__).parent / "dataset" / "tickets.jsonl"
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                t = json.loads(line)
                out[t["ticket_id"]] = f"{t['subject']}\n\n{t['body']}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--area")
    args = parser.parse_args()

    report: Report = json.loads(args.report.read_text())
    kb = load_kb()
    tickets = load_ticket_text()
    existing = load_grades()

    graded = 0
    with GRADES.open("a", encoding="utf-8") as fh:
        for row in report["rows"]:
            if args.area and row["product_area"] != args.area:
                continue
            done = existing.get(row["ticket_id"], {})
            pending = [d for d in row["retrieved"] if base_doc_id(d) not in done]
            if not pending:
                continue

            print("\n" + "=" * 78)
            print(f"[{row['ticket_id']}] {tickets[row['ticket_id']]}")
            print("-" * 78)
            for doc_id in pending:
                article = kb[base_doc_id(doc_id)]
                print(f"\n  {doc_id}  —  {article['title']}\n  {article['text'][:500]}\n")
                while True:
                    raw = input(f"  grade? {SCALE}\n  > ").strip().lower()
                    if raw == "q":
                        print(f"\n{graded} grades written")
                        return
                    if raw == "s":
                        break
                    if raw in {"0", "1", "2", "3"}:
                        fh.write(
                            json.dumps(
                                {
                                    "ticket_id": row["ticket_id"],
                                    "doc_id": base_doc_id(doc_id),
                                    "grade": int(raw),
                                }
                            )
                            + "\n"
                        )
                        fh.flush()
                        graded += 1
                        break
    print(f"\n{graded} grades written")


if __name__ == "__main__":
    main()
