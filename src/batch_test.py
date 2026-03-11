"""
Batch end-to-end test runner.

Usage:
    python batch_test.py                        # uses built-in QUERIES list
    python batch_test.py --queries queries.txt  # one query per line in a file
    python batch_test.py --out results.txt      # custom output file (default: batch_results.txt)
"""

import argparse
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from llm import LLMClient
from slop import get_query_context


QUERIES = [
    "If I control [[Chalice of the Void]] set to 0, can my opponent cast [[Ancestral Vision]] for free using suspend?",
    "I attack with [[Serra Angel]]. My opponent blocks with [[Llanowar Elves]]. What happens?",
    "Can I use [[Dark Ritual]] to cast [[Emrakul, the Aeons Torn]]?",
]


def run_batch(queries: list[str], llm_client: LLMClient) -> list[dict]:
    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Query: {query}")
        try:
            query_context = get_query_context(query)
            response = llm_client.generate(query_context)
        except Exception as exc:
            response = f"ERROR: {exc}"
        results.append({"query": query, "response": response})
        print(f"  Done.")
    return results


def write_results(results: list[dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"MTG Judge — Batch Test Results\n")
        f.write(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total queries: {len(results)}\n")
        f.write("=" * 72 + "\n\n")

        for i, entry in enumerate(results, 1):
            f.write(f"[{i}] QUERY\n")
            f.write(textwrap.fill(entry["query"], width=72, subsequent_indent="     ") + "\n\n")
            f.write("RESPONSE\n")
            for line in entry["response"].splitlines():
                f.write(f"  {line}\n")
            f.write("\n" + "-" * 72 + "\n\n")

    print(f"\nResults written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run MTG Judge on a batch of queries.")
    parser.add_argument("--queries", "-q", help="Path to a text file with one query per line.")
    parser.add_argument("--out", "-o", default="batch_results.txt", help="Output file path (default: batch_results.txt).")
    args = parser.parse_args()

    if args.queries:
        queries_path = Path(args.queries)
        if not queries_path.exists():
            print(f"Error: queries file not found: {queries_path}", file=sys.stderr)
            sys.exit(1)
        queries = [line.strip() for line in queries_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        queries = QUERIES

    if not queries:
        print("No queries to run.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model...")
    llm_client = LLMClient()
    llm_client._load()

    results = run_batch(queries, llm_client)
    write_results(results, Path(args.out))


if __name__ == "__main__":
    main()
