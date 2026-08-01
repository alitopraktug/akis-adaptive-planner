"""Behavior and productivity analytics for the Akis dashboard."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


def _prepare_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(sessions, pd.DataFrame):
        raise TypeError("sessions must be a pandas DataFrame.")
    if sessions.empty:
        return sessions.copy()
    required = {"start_time", "outcome", "actual_minutes", "planned_minutes", "focus_rating", "category"}
    missing = sorted(required - set(sessions.columns))
    if missing:
        raise ValueError(f"Session analytics is missing columns: {', '.join(missing)}.")
    frame = sessions.copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
    frame = frame.dropna(subset=["start_time"])
    frame["date"] = frame["start_time"].dt.date
    frame["hour"] = frame["start_time"].dt.hour
    frame["completed"] = (frame["outcome"] == "completed").astype(int)
    return frame


def weekly_summary(sessions: pd.DataFrame, today: date | None = None) -> dict[str, Any]:
    """Return primary productivity indicators for the trailing seven days."""

    frame = _prepare_sessions(sessions)
    current_day = today or datetime.now().date()
    if frame.empty:
        return {
            "sessions": 0,
            "completed": 0,
            "completion_rate": 0.0,
            "focus_minutes": 0,
            "average_focus": 0.0,
            "estimate_accuracy": 0.0,
        }
    start_day = current_day - timedelta(days=6)
    week = frame[(frame["date"] >= start_day) & (frame["date"] <= current_day)]
    if week.empty:
        return weekly_summary(pd.DataFrame())
    completed = week[week["completed"] == 1]
    if completed.empty:
        accuracy = 0.0
    else:
        relative_error = (
            (completed["actual_minutes"] - completed["planned_minutes"]).abs()
            / completed["planned_minutes"].clip(lower=1)
        )
        accuracy = float(np.clip(1.0 - relative_error.mean(), 0.0, 1.0))
    return {
        "sessions": len(week),
        "completed": int(week["completed"].sum()),
        "completion_rate": float(week["completed"].mean()),
        "focus_minutes": int(completed["actual_minutes"].sum()),
        "average_focus": float(week["focus_rating"].mean()),
        "estimate_accuracy": accuracy,
    }


def hourly_productivity(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completion and focus quality by hour."""

    frame = _prepare_sessions(sessions)
    if frame.empty:
        return pd.DataFrame(columns=["hour", "completion_rate", "focus_rating", "sessions"])
    result = frame.groupby("hour").agg(
        completion_rate=("completed", "mean"),
        focus_rating=("focus_rating", "mean"),
        sessions=("session_id", "count"),
    ).reset_index()
    return result


def category_performance(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duration accuracy and completion by task category."""

    frame = _prepare_sessions(sessions)
    if frame.empty:
        return pd.DataFrame(
            columns=["category", "sessions", "completion_rate", "planned_minutes", "actual_minutes", "duration_ratio"]
        )
    result = frame.groupby("category").agg(
        sessions=("completed", "count"),
        completion_rate=("completed", "mean"),
        planned_minutes=("planned_minutes", "sum"),
        actual_minutes=("actual_minutes", "sum"),
    ).reset_index()
    result["duration_ratio"] = result["actual_minutes"] / result["planned_minutes"].clip(lower=1)
    return result


def daily_activity(sessions: pd.DataFrame, days: int = 28) -> pd.DataFrame:
    """Return daily planned, completed, and focus-minute trends."""

    if days < 1 or days > 365:
        raise ValueError("days must be between 1 and 365.")
    frame = _prepare_sessions(sessions)
    if frame.empty:
        return pd.DataFrame(columns=["date", "planned", "completed", "focus_minutes"])
    recent_start = frame["date"].max() - timedelta(days=days - 1)
    recent = frame[frame["date"] >= recent_start]
    return recent.groupby("date").agg(
        planned=("completed", "count"),
        completed=("completed", "sum"),
        focus_minutes=("actual_minutes", "sum"),
    ).reset_index()
