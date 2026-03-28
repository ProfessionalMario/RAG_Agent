import json
from rag.pipeline import run_pipeline
from core.logger import get_logger

logger = get_logger(__name__)


def evaluate_case(case, result):
    decision = result.get("decision", "").lower()
    expected = case["expected"].lower()

    return expected in decision


def run_evaluation():
    with open("tests/evaluation/test_cases.json", "r") as f:
        cases = json.load(f)

    correct = 0
    total = len(cases)

    for case in cases:
        try:
            # simulate single-column pipeline input
            fake_report = f"""
            Rows: 1000

            Missing by Column
            {case['column']}: {int(case['missing_percent'] * 10)}

            Numeric: {case['column'] if case['dtype']=='numeric' else ''}
            Categorical: {case['column'] if case['dtype']=='categorical' else ''}
            """

            results = run_pipeline(fake_report)

            result = results[0]

            if evaluate_case(case, result):
                correct += 1
            else:
                logger.warning(f"Failed case: {case} → {result}")

        except Exception as e:
            logger.error(f"Crash on case {case}: {str(e)}")

    accuracy = correct / total * 100
    print(f"\n✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")