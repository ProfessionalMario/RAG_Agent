# cli/app.py

from rag.pipeline import safe_run_pipeline, run_query
from core.logger import get_logger
import logging
import os
import time
import threading
import itertools
import sys
logger = get_logger(__name__)

# 🔒 Suppress ONLY console noise (keep file logging intact)
# for handler in logger.handlers:
#     if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
#         handler.setLevel(logging.ERROR)

# 🔇 Disable transformers + HF noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🔇 Disable tqdm progress bars globally
os.environ["DISABLE_TQDM"] = "1"

# 🔇 Suppress library loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

def cli_notify(msg: str):
    """Minimal live update printed to terminal"""
    print(f"> {msg}", flush=True)

def spinner(msg="Processing"):
    stop_event = threading.Event()

    def run():
        for c in itertools.cycle(["|", "/", "-", "\\"]):
            if stop_event.is_set():
                break
            sys.stdout.write(f"\r> {msg}... {c}")
            sys.stdout.flush()
            time.sleep(0.1)

    t = threading.Thread(target=run)
    t.start()

    return stop_event

def display_results(results):
    """Formatted final results"""
    for r in results:
        print("\n" + "=" * 40)
        print(f"Column: {r.get('column', 'N/A')}")
        print("-" * 40)

        decision = r.get("decision", "No decision")
        for line in decision.split("\n"):
            print(line.strip())

    print("\n" + "=" * 40)


def run_pipeline_cli():
    """Run the EDA pipeline with clean UX"""
    path = input("Enter report path: ").strip()

    cli_notify(f"Starting pipeline for: {path}")
    start = time.time()
    try:
        cli_notify("Parsing report...")
        t1 = time.time()
        stop_event = spinner("Running pipeline")
        results = safe_run_pipeline(path)

        cli_notify("Pipeline finished. Preparing final summary...")
        display_results(results)
        stop_event.set()
        print("\r> Pipeline completed!     ")

        # cli_notify("Done.")

    except FileNotFoundError:
        logger.exception(f"Report not found: {path}")
        print(f"Error: Report file not found at {path}")

    except Exception as e:
        logger.exception(f"Pipeline failed: {str(e)}")
        print(f"Error: Pipeline execution failed")


def run_query_cli():
    """Run live query mode"""
    cli_notify("Query mode: ask anything (type 'exit' to quit)")

    while True:
        q = input("\nAsk: ").strip()

        if q.lower() == "exit":
            cli_notify("Exiting query mode.")
            break

        try:
            cli_notify("Thinking...")
            ans = run_query(q)
            print("\nAnswer:", ans)

        except Exception as e:
            logger.exception(f"Query failed: {str(e)}")
            print("Error: Could not process query.")


def main():
    print("\nSelect Mode:")
    print("1. Pipeline (EDA report)")
    print("2. Query (Ask anything)")

    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        run_pipeline_cli()
    elif choice == "2":
        run_query_cli()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()