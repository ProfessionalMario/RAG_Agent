"""
RAG_Agent CLI.

Two modes:

- `pipeline <report.txt>` — feed an EDA report through the full RAG +
  critic pipeline and print the resulting per-column decisions.
- `query "..."` — ask a free-form scikit-learn question against the
  knowledge base.

When invoked with no arguments, drops into an interactive menu.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
import threading
import time
from typing import List

# Quiet down noisy libraries.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from core.logger import get_logger  # noqa: E402  (after env tweak)

logger = get_logger(__name__)

DEFAULT_REPORT = "data/reports/sample_report.txt"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)


# --- UX helpers --------------------------------------------------------------
def cli_notify(msg: str) -> None:
    print(f"\n> {msg}", flush=True)


class CLI_Spinner:
    def __init__(self, msg: str = "Processing") -> None:
        self.msg = msg
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._animate, daemon=True)

    def _animate(self) -> None:
        for c in itertools.cycle(["|", "/", "-", "\\"]):
            if self.stop_event.is_set():
                break
            sys.stdout.write(f"\r> {self.msg}... {c} ")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(self.msg) + 12) + "\r")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()


def display_results(results: List[dict]) -> None:
    border = "═" * 60
    print("\n╔" + border + "╗")
    print("║" + " FINAL PREPROCESSING STRATEGY ".center(60) + "║")
    print("╚" + border + "╝")
    for r in results:
        col = str(r.get("column", "N/A")).upper()
        print(f"\n◈ COLUMN: {col}")
        print(f"  ├─ DECISION: {r.get('decision', 'N/A')}")
        print(f"  └─ REASON:   {r.get('reason', 'N/A')}")
    print("\n" + border)


# --- Modes -------------------------------------------------------------------
def run_pipeline_cli(report_path: str = DEFAULT_REPORT) -> None:
    if not os.path.exists(report_path):
        print(f"❌ Report not found at {report_path}")
        return
    cli_notify(f"Analyzing: {report_path}")
    spinner = CLI_Spinner("Running RAG pipeline")
    try:
        from rag.pipeline import safe_run_pipeline
        spinner.start()
        results = safe_run_pipeline(report_path)
    finally:
        spinner.stop()
    display_results(results)
    cli_notify("Task complete.")


def run_query_cli() -> None:
    cli_notify("KNOWLEDGE MODE — ask scikit-learn / preprocessing questions.")
    print("   (Type 'exit' to return to the menu.)")
    from rag.pipeline import run_query
    while True:
        try:
            q = input("\n❓ Question: ").strip()
        except EOFError:
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        spinner = CLI_Spinner("Searching knowledge base")
        try:
            spinner.start()
            answer = run_query(q)
        finally:
            spinner.stop()
        print(f"\n📖 ANSWER:\n{answer}")


def run_query_once(question: str) -> None:
    from rag.pipeline import run_query
    print(run_query(question))


# --- Entry point -------------------------------------------------------------
def interactive_menu() -> None:
    print("\n" + "█" * 40)
    print("   RAG-ML PREPROCESSING ENGINE   ")
    print("█" * 40)
    while True:
        print("\n[1] Run pipeline (analyze EDA report)")
        print("[2] Live query (ask the knowledge base)")
        print("[3] Exit")
        try:
            choice = input("\nSelection: ").strip()
        except EOFError:
            return
        if choice == "1":
            run_pipeline_cli()
        elif choice == "2":
            run_query_cli()
        elif choice == "3":
            print("Goodbye!")
            return
        else:
            print("Invalid choice.")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag_agent",
        description="RAG-driven preprocessing decision agent (CLI).",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_pipeline = sub.add_parser("pipeline", help="Run the pipeline on a report.")
    p_pipeline.add_argument("report", nargs="?", default=DEFAULT_REPORT,
                            help=f"Report file path (default: {DEFAULT_REPORT})")

    p_query = sub.add_parser("query", help="Ask one question of the KB.")
    p_query.add_argument("question", nargs="+",
                         help="The question to ask (quote it).")

    args = parser.parse_args(argv)

    if args.cmd == "pipeline":
        run_pipeline_cli(args.report)
    elif args.cmd == "query":
        run_query_once(" ".join(args.question))
    else:
        interactive_menu()
    return 0


if __name__ == "__main__":
    try:
        # from debugflow import flow_engine
        # flow_engine.launch("main",Ghost=True,Real_Time=True)
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nSession ended.")
        sys.exit(0)
