"""Tests for behavior learning and adaptive duration prediction."""

from datetime import datetime, timedelta

import pandas as pd

from learning_engine import AdaptiveDurationPredictor, learn_behavior
from models import Task


def _sessions(count: int = 20) -> pd.DataFrame:
    rows = []
    base = datetime(2026, 7, 1, 9, 0)
    for index in range(count):
        category = "Coding" if index % 2 == 0 else "Admin"
        planned = 60
        actual = 78 if category == "Coding" else 45
        rows.append(
            {
                "session_id": str(index),
                "category": category,
                "planned_minutes": planned,
                "actual_minutes": actual,
                "start_time": base + timedelta(days=index, hours=index % 3),
                "outcome": "completed" if index % 5 else "skipped",
                "focus_rating": 5 if index % 3 == 0 else 3,
                "priority": 4,
                "energy_required": 4,
            }
        )
    return pd.DataFrame(rows)


def test_learn_behavior_finds_duration_patterns() -> None:
    profile = learn_behavior(_sessions())
    assert profile.total_sessions == 20
    assert profile.duration_factors["Coding"] > 1.0
    assert profile.duration_factors["Admin"] < 1.0
    assert profile.confidence == 0.4
    assert profile.completion_by_weekday
    assert set(profile.preferred_weekdays) == {"Coding", "Admin"}


def test_duration_predictor_trains_with_enough_sessions() -> None:
    sessions = _sessions(30)
    profile = learn_behavior(sessions)
    predictor = AdaptiveDurationPredictor().fit(sessions)
    prediction = predictor.predict(Task("Build API", "Coding", 4, 60, 4), 9, profile)
    assert predictor.is_trained
    assert 45 <= prediction <= 100


def test_duration_predictor_uses_profile_during_cold_start() -> None:
    sessions = _sessions(6)
    profile = learn_behavior(sessions)
    predictor = AdaptiveDurationPredictor().fit(sessions)
    prediction = predictor.predict(Task("Build UI", "Coding", 4, 60, 4), 9, profile)
    assert not predictor.is_trained
    assert prediction > 60
