import os
import time
import threading
import itertools
import sys
import logging
# from debugflow import flow_engine
# Suppress library noise BEFORE imports
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_TQDM"] = "1"

from rag.pipeline import safe_run_pipeline, run_query
from core.logger import get_logger

logger = get_logger(__name__)

# Standardize Logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING) 

# Hardcoded Default Path
DEFAULT_REPORT = "data/pdfs/reports.txt"

# -----------------------------
# UX UTILITIES
# -----------------------------

def cli_notify(msg: str):
    print(f"\n> {msg}", flush=True)

class CLI_Spinner:
    def __init__(self, msg="Processing"):
        self.msg = msg
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._animate)

    def _animate(self):
        for c in itertools.cycle(["|", "/", "-", "\\"]):
            if self.stop_event.is_set():
                break
            sys.stdout.write(f"\r> {self.msg}... {c} ")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(self.msg) + 10) + "\r")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()

# -----------------------------
# MODE: PIPELINE
# -----------------------------

def display_results(results):
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " FINAL PREPROCESSING STRATEGY ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    for r in results:
        col = r.get('column', 'N/A').upper()
        print(f"\n◈ COLUMN: {col}")
        print(f"  ├─ DECISION: {r.get('decision', 'N/A')}")
        print(f"  └─ REASON:   {r.get('reason', 'N/A')}")

    print("\n" + "═" * 60)

def run_pipeline_cli():
    """UX wrapper using hardcoded reports.txt path."""
    if not os.path.exists(DEFAULT_REPORT):
        print(f"❌ Error: Report not found at {DEFAULT_REPORT}")
        return

    cli_notify(f"Analyzing: {DEFAULT_REPORT}")
    spinner = CLI_Spinner("Running RAG Pipeline")
    
    try:
        spinner.start()
        # Using safe_run_pipeline as re-enabled in your pipeline.py
        results = safe_run_pipeline(DEFAULT_REPORT)
        # cli_notify(results)
        spinner.stop()

        display_results(results)
        cli_notify("Task Completed.")

    except Exception as e:
        spinner.stop()
        logger.exception(f"Pipeline failed: {e}")
        print(f"\n❌ Error: Pipeline execution failed.")

# -----------------------------
# MODE: LIVE QUERY
# -----------------------------

def run_query_cli():
    cli_notify("KNOWLEDGE MODE: Ask technical Scikit-Learn questions.")
    print("   (Type 'exit' to return to menu)")

    while True:
        q = input("\n❓ Question: ").strip()

        if q.lower() in ["exit", "quit"]:
            break

        if not q:
            continue

        spinner = CLI_Spinner("Searching Knowledge Base")
        try:
            spinner.start()
            # Using run_query as re-enabled in your pipeline.py
            ans = run_query(q)
            spinner.stop()
            
            print(f"\n📖 CONTEXTUAL ANSWER:\n{ans}")

        except Exception as e:
            spinner.stop()
            logger.error(f"Query failed: {e}")
            print("❌ Error: Could not retrieve answer.")

# -----------------------------
# MAIN ENTRY
# -----------------------------

def main():
    print("\n" + "█" * 40)
    print("   RAG-ML PREPROCESSING ENGINE   ")
    print("█" * 40)

    while True:
        print("\n[1] Run Pipeline (Analyze reports)")
        print("[2] Live Query (Ask Knowledge Base)")
        print("[3] Exit")

        choice = input("\nSelection: ").strip()

        if choice == "1":
            run_pipeline_cli()
        elif choice == "2":
            run_query_cli()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    try:
        # flow_engine.launch("main",Ghśost=True,Real_Time=True)
        main()
    except KeyboardInterrupt:
        print("\n\nSession ended.")
        sys.exit(0)