from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.predict import predict_records, validate_input
from src.utils import read_events, save_events, severity_for


class FakeModel:
    def predict_proba(self, data):
        return np.array([[0.1, 0.9] for _ in range(len(data))])


META = {
    "features": ["proto", "service", "state", "rate"],
    "numeric": ["rate"],
    "categorical": ["proto", "service", "state"],
    "decision_threshold": 0.5,
    "training_p95": {"rate": 10, "spkts": 10, "dpkts": 10, "sbytes": 10, "dbytes": 10, "dur": 10},
}


def test_severity_boundaries():
    assert severity_for(0, 0.99) == "Low"
    assert severity_for(1, 0.50) == "Medium"
    assert severity_for(1, 0.75) == "High"
    assert severity_for(1, 0.90) == "High"
    assert severity_for(1, 0.90001) == "Critical"


def test_input_validation_and_infinite_value():
    frame = pd.DataFrame([{"proto": "new-protocol", "service": "-", "state": "FIN", "rate": float("inf")}])
    cleaned = validate_input(frame, META)
    assert pd.isna(cleaned.loc[0, "rate"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_input(frame.drop(columns="state"), META)


def test_prediction_and_event_persistence(tmp_path):
    frame = pd.DataFrame([{"proto": "tcp", "service": "http", "state": "FIN", "rate": 15}])
    result = predict_records(frame, FakeModel(), META)
    assert result.loc[0, "prediction"] == "Attack"
    assert result.loc[0, "severity"] == "High"
    assert result.loc[0, "attack_cat"] == "Unknown type"
    assert "high connection rate" in result.loc[0, "suggested_reason"]
    db = tmp_path / "events.sqlite3"
    save_events(result, "batch-1", db)
    save_events(result, "batch-1", db)
    assert len(read_events(db)) == 1
