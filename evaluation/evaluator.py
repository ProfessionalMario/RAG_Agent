from core.tracking import init_mlflow, start_run, log_metrics, end_run

def evaluate(results):
    init_mlflow()
    start_run("eda_rag_eval")

    correct = 0
    total = len(results)

    for r in results:
        if r["predicted"] == r["expected"]:
            correct += 1

    accuracy = correct / total

    log_metrics({
        "accuracy": accuracy,
        "total_samples": total
    })

    end_run()

    return accuracy