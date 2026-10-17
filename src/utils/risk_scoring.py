from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

from src.detection.fraud_model import load_model, predict as ml_predict
from src.detection.rule_based import RULE_REASONS, apply_fraud_rules
from src.detection.anomaly_detection import detect_anomalies

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_detector(name: str, detector: Callable[..., pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
    """Run one detector, turning any failure into an explicit error.

    Previously every detector was wrapped in a bare ``except Exception`` that
    silently substituted an all-zero score. A missing/corrupt model therefore
    made every transaction look safe with no indication anything was wrong.
    Failing loudly (with the detector's name) makes that impossible to miss.
    """
    try:
        return detector(df)
    except Exception as e:  # noqa: BLE001 - re-raised with context below
        raise RuntimeError(f"{name} detector failed: {e}") from e


def combine_risk_scores(
    df: pd.DataFrame,
    ml_weight: float = 0.5,
    rule_weight: float = 0.3,
    anomaly_weight: float = 0.2,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pandas DataFrame")
    result = df.copy()

    # Each detector contributes a normalised score in [0, 1]. The individual
    # results are ALSO attached to ``result`` so explain_risk() can report why
    # a transaction was flagged (previously only the final score was kept and
    # every explanation came back empty).
    ml_res = _run_detector("ML", ml_predict, result)
    ml_prob = ml_res["fraud_probability"].values
    result["fraud_probability"] = ml_prob

    rule_res = _run_detector("Rule-based", apply_fraud_rules, result)
    # Normalise against the fixed number of possible rules, not the batch
    # maximum, so a transaction's score does not depend on which other
    # transactions happen to be scored alongside it.
    rule_score = rule_res["rules_triggered"].values / max(1, len(RULE_REASONS))
    result["rules_triggered"] = rule_res["rules_triggered"].values
    result["rule_reasons"] = rule_res["rule_reasons"].values

    anom_res = _run_detector("Anomaly", detect_anomalies, result)
    # IsolationForest.score_samples() is the NEGATIVE of the [0, 1] isolation
    # score, so negating it gives a "higher = more anomalous" value that is
    # already bounded and independent of the other rows in the batch. The old
    # per-batch min-max rescaling made a transaction's anomaly contribution
    # depend on the rest of the batch and collapsed to exactly 0 for a single
    # row (max == min), so a lone high-risk request never got flagged.
    anom_score = np.clip(-anom_res["anomaly_score"].values, 0.0, 1.0)
    result["is_anomaly"] = anom_res["is_anomaly"].values
    result["anomaly_score"] = anom_res["anomaly_score"].values

    risk = ml_weight * ml_prob + rule_weight * rule_score + anomaly_weight * anom_score
    risk = np.clip(risk, 0, 1)
    result["risk_score"] = risk
    result["risk_level"] = pd.cut(
        risk,
        bins=[0, 0.3, 0.7, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    result["risk_level"] = result["risk_level"].astype(str)
    return result


def explain_risk(row: pd.Series) -> Dict[str, Any]:
    reasons = []
    if "fraud_probability" in row and row["fraud_probability"] > 0.5:
        reasons.append("High ML fraud probability")
    if "rule_reasons" in row and isinstance(row["rule_reasons"], list) and row["rule_reasons"]:
        reasons.extend(row["rule_reasons"])
    if "is_anomaly" in row and row["is_anomaly"] == 1:
        reasons.append("Anomalous transaction pattern")
    return {
        "risk_score": float(row.get("risk_score", 0)),
        "risk_level": str(row.get("risk_level", "low")),
        "reasons": reasons,
    }


if __name__ == "__main__":
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.csv", nrows=10)
    res = combine_risk_scores(df)
    print(res[["risk_score", "risk_level"]].head())
