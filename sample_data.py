"""Deterministic demo workspace generation for Akis."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np

from database import AkisDatabase
from models import Event, FocusSession, SessionOutcome, Task, UserSettings


def ensure_demo_events(database: AkisDatabase, *, today: date | None = None) -> None:
    """Add a small weekly event set when the workspace has no events."""

    if database.count_events() > 0:
        return
    current_day = today or datetime.now().date()
    monday = current_day - timedelta(days=current_day.weekday())
    event_specs = [
        ("Project Review", 1, time(15, 0), 60, 5, "Meeting", "Studio 2"),
        ("Algorithms Exam", 3, time(10, 0), 120, 5, "Study", "Engineering Hall"),
        ("Career Fair", 5, time(12, 0), 120, 4, "Career", "Main Campus"),
    ]
    for title, day_offset, start_time, duration, importance, category, location in event_specs:
        start = datetime.combine(monday + timedelta(days=day_offset), start_time)
        database.add_event(
            Event(
                title=title,
                start=start,
                end=start + timedelta(minutes=duration),
                importance=importance,
                category=category,
                location=location,
                notes="Demo event — edit your local data by adding real events.",
            )
        )


def seed_demo_data(
    database: AkisDatabase,
    *,
    today: date | None = None,
    random_state: int = 42,
) -> None:
    """Populate an empty database with realistic tasks and behavior history."""

    ensure_demo_events(database, today=today)
    if database.count_tasks() > 0:
        return
    current_day = today or datetime.now().date()
    database.save_settings(UserSettings())

    task_specs = [
        ("Finish adaptive planner architecture", "Coding", 5, 120, 5, 1, "morning"),
        ("Study dynamic programming", "Study", 4, 90, 4, 1, "morning"),
        ("Review project README", "Portfolio", 3, 45, 2, 2, "afternoon"),
        ("Reply to important emails", "Admin", 3, 30, 2, 0, "afternoon"),
        ("Prepare weekly progress notes", "Planning", 3, 40, 2, 4, "afternoon"),
        ("Practice technical interview questions", "Study", 4, 75, 4, 3, "morning"),
        ("Refactor database tests", "Coding", 4, 80, 4, 2, "morning"),
        ("Update portfolio project cards", "Portfolio", 2, 60, 3, 6, "afternoon"),
        ("Evening workout", "Health", 3, 60, 3, 0, "evening"),
    ]
    for title, category, priority, duration, energy, deadline_days, period in task_specs:
        deadline = datetime.combine(
            current_day + timedelta(days=deadline_days), time(18, 0)
        )
        fixed_start = (
            datetime.combine(current_day, time(18, 0)) if title == "Evening workout" else None
        )
        database.add_task(
            Task(
                title=title,
                category=category,
                priority=priority,
                estimated_minutes=duration,
                energy_required=energy,
                deadline=deadline,
                preferred_period=period,  # type: ignore[arg-type]
                fixed_start=fixed_start,
                splittable=duration > 45,
            )
        )

    rng = np.random.default_rng(random_state)
    category_hours = {
        "Coding": [9, 10, 11, 15],
        "Study": [9, 10, 14, 16],
        "Portfolio": [13, 14, 16],
        "Admin": [13, 15, 16],
        "Planning": [11, 14, 17],
        "Health": [17, 18, 19],
    }
    duration_factors = {
        "Coding": 1.18,
        "Study": 1.05,
        "Portfolio": 0.92,
        "Admin": 0.82,
        "Planning": 0.95,
        "Health": 1.00,
    }
    for day_offset in range(1, 29):
        session_day = current_day - timedelta(days=day_offset)
        if session_day.weekday() >= 5 and rng.random() < 0.45:
            continue
        session_count = int(rng.integers(2, 5))
        categories = rng.choice(list(category_hours), size=session_count, replace=False)
        for category_value in categories:
            category = str(category_value)
            hour = int(rng.choice(category_hours[category]))
            planned = int(rng.choice([30, 45, 60, 75, 90]))
            factor = duration_factors[category]
            actual = max(5, int(round(planned * factor + rng.normal(0, 8))))
            strong_hour = hour in category_hours[category][:2]
            completion_probability = 0.90 if strong_hour else 0.70
            completed = bool(rng.random() < completion_probability)
            outcome = SessionOutcome.COMPLETED if completed else SessionOutcome.SKIPPED
            if not completed:
                actual = int(rng.integers(0, max(6, planned // 3)))
            focus = int(np.clip(round((4.4 if strong_hour else 3.2) + rng.normal(0, 0.7)), 1, 5))
            database.record_session(
                FocusSession(
                    task_title=f"Historical {category} session",
                    category=category,
                    planned_minutes=planned,
                    actual_minutes=actual,
                    start_time=datetime.combine(session_day, time(hour, 0)),
                    outcome=outcome,
                    focus_rating=focus,
                    interruption_count=int(rng.integers(0, 4)),
                    priority=int(rng.integers(2, 6)),
                    energy_required=int(rng.integers(2, 6)),
                )
            )
