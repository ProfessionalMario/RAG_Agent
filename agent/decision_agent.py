from core.logger import get_logger
from core.exceptions import AgentError

logger = get_logger(__name__)


class DecisionAgent:
    def decide(self, column_data: dict, rules: list):
        try:
            logger.info(f"Making decision for {column_data['column']}")

            missing = column_data.get("missing_percent", 0)
            dtype = column_data.get("dtype", "")
            skew = column_data.get("skew", "")

            decision = "unknown"
            reason = []

            # Rule-based reasoning (simple + extensible)

            if missing > 40:
                decision = "drop_column"
                reason.append("High missing values")

            elif 10 < missing <= 40:
                decision = "impute"

                if dtype == "numeric":
                    if skew == "high":
                        decision = "impute_median"
                        reason.append("Skewed numeric → median")
                    else:
                        decision = "impute_mean"
                        reason.append("Normal numeric → mean")

                elif dtype == "categorical":
                    decision = "impute_mode"
                    reason.append("Categorical → mode")

            else:
                decision = "keep_column"
                reason.append("Low missing values")

            return {
                "column": column_data["column"],
                "decision": decision,
                "reason": reason,
                "rules_used": rules
            }

        except Exception as e:
            logger.error(f"Decision failed: {str(e)}")
            raise AgentError(f"Decision failed: {str(e)}")