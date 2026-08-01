"""Behavior-learning utilities for personalized schedule generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from models import Task


@dataclass(frozen=True)
class BehaviorProfile:
    """Explainable behavioral signals learned from focus history."""

    total_sessions: int
    duration_factors: dict[str, float]
    completion_by_hour: dict[int, float]
    focus_by_hour: dict[int, float]
    preferred_hours: dict[str, int]
    confidence: float
    insights: tuple[str, ...]
    completion_by_weekday: dict[int, float] = field(default_factory=dict)
    preferred_weekdays: dict[str, int] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "BehaviorProfile":
        """Return a neutral cold-start profile."""

        return cls(
            total_sessions=0,
            duration_factors={},
            completion_by_hour={},
            focus_by_hour={},
            preferred_hours={},
            confidence=0.0,
            insights=("Complete focus sessions to start personalizing your schedule.",),
            completion_by_weekday={},
            preferred_weekdays={},
        )


def _validate_sessions(sessions: pd.DataFrame) -> None:
    required = {
        "category",
        "planned_minutes",
        "actual_minutes",
        "start_time",
        "outcome",
        "focus_rating",
    }
    missing = sorted(required - set(sessions.columns))
    if missing:
        raise ValueError(f"Session history is missing required columns: {', '.join(missing)}.")


def learn_behavior(sessions: pd.DataFrame) -> BehaviorProfile:
    """Build a smoothed and explainable behavior profile from session history."""

    if not isinstance(sessions, pd.DataFrame):
        raise TypeError("sessions must be a pandas DataFrame.")
    if sessions.empty:
        return BehaviorProfile.empty()
    _validate_sessions(sessions)

    frame = sessions.copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
    frame = frame.dropna(subset=["start_time"])
    if frame.empty:
        return BehaviorProfile.empty()
    frame["start_hour"] = frame["start_time"].dt.hour
    frame["weekday"] = frame["start_time"].dt.weekday
    frame["completed"] = (frame["outcome"] == "completed").astype(float)

    completed = frame[(frame["completed"] == 1.0) & (frame["actual_minutes"] > 0)].copy()
    duration_factors: dict[str, float] = {}
    if not completed.empty:
        completed["duration_ratio"] = (
            completed["actual_minutes"] / completed["planned_minutes"].clip(lower=1)
        ).clip(0.5, 2.0)
        duration_factors = {
            str(category): round(float(group["duration_ratio"].median()), 3)
            for category, group in completed.groupby("category")
        }

    hourly = frame.groupby("start_hour").agg(
        completed=("completed", "sum"),
        sessions=("completed", "count"),
        focus=("focus_rating", "mean"),
    )
    completion_by_hour = {
        int(hour): round(float((row["completed"] + 1.0) / (row["sessions"] + 2.0)), 3)
        for hour, row in hourly.iterrows()
    }
    focus_by_hour = {
        int(hour): round(float(row["focus"]), 3) for hour, row in hourly.iterrows()
    }

    preferred_hours: dict[str, int] = {}
    category_hour = frame.groupby(["category", "start_hour"]).agg(
        completion=("completed", "mean"),
        focus=("focus_rating", "mean"),
        count=("completed", "count"),
    )
    category_hour["fit"] = (
        category_hour["completion"] * 0.65
        + (category_hour["focus"] / 5.0) * 0.35
        + np.minimum(category_hour["count"], 5) * 0.01
    )
    for category in frame["category"].dropna().astype(str).unique():
        subset = category_hour.loc[category]
        preferred_hours[category] = int(subset["fit"].idxmax())

    weekday = frame.groupby("weekday").agg(
        completed=("completed", "sum"),
        sessions=("completed", "count"),
        focus=("focus_rating", "mean"),
    )
    completion_by_weekday = {
        int(day): round(float((row["completed"] + 1.0) / (row["sessions"] + 2.0)), 3)
        for day, row in weekday.iterrows()
    }
    category_weekday = frame.groupby(["category", "weekday"]).agg(
        completion=("completed", "mean"),
        focus=("focus_rating", "mean"),
        count=("completed", "count"),
    )
    category_weekday["fit"] = (
        category_weekday["completion"] * 0.60
        + (category_weekday["focus"] / 5.0) * 0.30
        + np.minimum(category_weekday["count"], 5) * 0.02
    )
    preferred_weekdays: dict[str, int] = {}
    for category in frame["category"].dropna().astype(str).unique():
        subset = category_weekday.loc[category]
        preferred_weekdays[category] = int(subset["fit"].idxmax())

    insights: list[str] = []
    if focus_by_hour:
        best_hour = max(focus_by_hour, key=focus_by_hour.get)  # type: ignore[arg-type]
        insights.append(f"Your strongest focus window starts around {best_hour:02d}:00.")
    if completion_by_weekday:
        weekday_names = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
        best_day = max(completion_by_weekday, key=completion_by_weekday.get)  # type: ignore[arg-type]
        insights.append(f"{weekday_names[best_day]} is currently your strongest completion day.")
    for category, factor in sorted(duration_factors.items(), key=lambda item: abs(item[1] - 1), reverse=True)[:2]:
        difference = abs(factor - 1.0) * 100
        if difference >= 8:
            direction = "longer" if factor > 1 else "faster"
            insights.append(
                f"{category} tasks run about {difference:.0f}% {direction} than your estimates."
            )
    completion_rate = float(frame["completed"].mean())
    insights.append(f"Your observed session completion rate is {completion_rate:.0%}.")

    return BehaviorProfile(
        total_sessions=len(frame),
        duration_factors=duration_factors,
        completion_by_hour=completion_by_hour,
        focus_by_hour=focus_by_hour,
        preferred_hours=preferred_hours,
        confidence=round(min(1.0, len(frame) / 50.0), 3),
        insights=tuple(insights),
        completion_by_weekday=completion_by_weekday,
        preferred_weekdays=preferred_weekdays,
    )


class AdaptiveDurationPredictor:
    """Predict task duration from user history with a transparent fallback."""

    minimum_training_sessions = 12

    def __init__(self) -> None:
        self._model: Pipeline | None = None
        self.training_samples = 0

    @property
    def is_trained(self) -> bool:
        """Return whether a machine-learning model is available."""

        return self._model is not None

    def fit(self, sessions: pd.DataFrame) -> "AdaptiveDurationPredictor":
        """Train on completed sessions when sufficient history exists."""

        if sessions.empty:
            return self
        _validate_sessions(sessions)
        required = {"priority", "energy_required"}
        missing = sorted(required - set(sessions.columns))
        if missing:
            raise ValueError(f"Duration training is missing columns: {', '.join(missing)}.")

        frame = sessions[sessions["outcome"] == "completed"].copy()
        frame = frame[(frame["actual_minutes"] >= 5) & (frame["actual_minutes"] <= 720)]
        frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
        frame = frame.dropna(subset=["start_time"])
        self.training_samples = len(frame)
        if len(frame) < self.minimum_training_sessions:
            self._model = None
            return self

        frame["start_hour"] = frame["start_time"].dt.hour
        features = frame[
            ["category", "planned_minutes", "start_hour", "priority", "energy_required"]
        ]
        target = frame["actual_minutes"].astype(float)
        transformer = ColumnTransformer(
            [("category", OneHotEncoder(handle_unknown="ignore"), ["category"])],
            remainder="passthrough",
        )
        self._model = Pipeline(
            [
                ("features", transformer),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=7,
                        min_samples_leaf=2,
                        random_state=42,
                    ),
                ),
            ]
        )
        self._model.fit(features, target)
        return self

    def predict(self, task: Task, start_hour: int, profile: BehaviorProfile) -> int:
        """Predict realistic duration in minutes for a task and start hour."""

        if not 0 <= start_hour <= 23:
            raise ValueError("start_hour must be between 0 and 23.")
        if self._model is not None:
            features = pd.DataFrame(
                [
                    {
                        "category": task.category,
                        "planned_minutes": task.estimated_minutes,
                        "start_hour": start_hour,
                        "priority": task.priority,
                        "energy_required": task.energy_required,
                    }
                ]
            )
            prediction = float(self._model.predict(features)[0])
        else:
            factor = profile.duration_factors.get(task.category, 1.0)
            prediction = task.estimated_minutes * factor
        return int(np.clip(round(prediction / 5.0) * 5, 5, 720))


def completion_probability(
    task: Task,
    start_hour: int,
    profile: BehaviorProfile,
    weekday: int | None = None,
) -> float:
    """Estimate completion probability with Bayesian-smoothed behavior signals."""

    hourly = profile.completion_by_hour.get(start_hour, 0.65)
    preferred_hour = profile.preferred_hours.get(task.category)
    category_fit = 0.85 if preferred_hour is None else max(0.45, 1.0 - abs(start_hour - preferred_hour) * 0.08)
    preferred_period_fit = 1.0
    if task.preferred_period == "morning" and start_hour >= 12:
        preferred_period_fit = 0.72
    elif task.preferred_period == "afternoon" and not 12 <= start_hour < 17:
        preferred_period_fit = 0.72
    elif task.preferred_period == "evening" and start_hour < 17:
        preferred_period_fit = 0.72
    weekday_fit = profile.completion_by_weekday.get(weekday, 0.65) if weekday is not None else 0.65
    preferred_weekday = profile.preferred_weekdays.get(task.category)
    category_day_fit = 0.80 if preferred_weekday is None or weekday is None else (
        1.0 if preferred_weekday == weekday else 0.62
    )
    probability = (
        hourly * 0.42
        + category_fit * 0.22
        + preferred_period_fit * 0.12
        + weekday_fit * 0.14
        + category_day_fit * 0.10
    )
    return float(np.clip(probability, 0.10, 0.98))
