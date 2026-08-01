"""Tests for adaptive schedule generation."""

from datetime import date, datetime, time, timedelta

import pandas as pd

from learning_engine import AdaptiveDurationPredictor, BehaviorProfile
from models import Event, Task, UserSettings
from scheduler import generate_schedule


def _profile() -> BehaviorProfile:
    return BehaviorProfile(
        total_sessions=20,
        duration_factors={"Coding": 1.0, "Admin": 1.0},
        completion_by_hour={9: 0.9, 10: 0.85, 14: 0.5, 15: 0.5},
        focus_by_hour={9: 4.8, 10: 4.5, 14: 3.0, 15: 3.0},
        preferred_hours={"Coding": 9, "Admin": 14},
        confidence=0.4,
        insights=(),
    )


def test_schedule_has_no_overlapping_blocks() -> None:
    target = date(2026, 8, 3)
    tasks = [
        Task("Deep work", "Coding", 5, 90, 5),
        Task("Email review", "Admin", 3, 30, 2),
        Task(
            "Team meeting",
            "Meeting",
            4,
            60,
            3,
            fixed_start=datetime.combine(target, time(11, 0)),
        ),
    ]
    result = generate_schedule(
        tasks,
        target_date=target,
        settings=UserSettings(day_start_hour=9, day_end_hour=18),
        profile=_profile(),
        predictor=AdaptiveDurationPredictor().fit(pd.DataFrame()),
    )
    blocks = sorted(result.blocks, key=lambda block: block.start)
    assert all(first.end <= second.start for first, second in zip(blocks, blocks[1:]))
    assert not result.unscheduled_tasks


def test_learned_preference_places_coding_in_morning() -> None:
    target = date(2026, 8, 3)
    task = Task("Implement scheduler", "Coding", 4, 60, 5)
    result = generate_schedule(
        [task],
        target_date=target,
        settings=UserSettings(day_start_hour=9, day_end_hour=18),
        profile=_profile(),
        predictor=AdaptiveDurationPredictor().fit(pd.DataFrame()),
    )
    task_block = next(block for block in result.blocks if block.task_id == task.task_id)
    assert task_block.start.hour in {9, 10}


def test_fixed_conflict_is_reported() -> None:
    target = date(2026, 8, 3)
    task = Task(
        "Lunch meeting",
        "Meeting",
        4,
        45,
        3,
        fixed_start=datetime.combine(target, time(13, 0)),
    )
    result = generate_schedule(
        [task],
        target_date=target,
        settings=UserSettings(lunch_start_hour=13, lunch_minutes=45),
        profile=_profile(),
        predictor=AdaptiveDurationPredictor().fit(pd.DataFrame()),
    )
    assert result.unscheduled_tasks == (task,)
    assert any("conflicts" in warning for warning in result.warnings)


def test_unscheduled_task_when_day_is_full() -> None:
    target = date(2026, 8, 3)
    tasks = [Task(f"Long task {index}", "Coding", 3, 180, 4, splittable=False) for index in range(4)]
    result = generate_schedule(
        tasks,
        target_date=target,
        settings=UserSettings(day_start_hour=9, day_end_hour=14, lunch_minutes=0),
        profile=_profile(),
        predictor=AdaptiveDurationPredictor().fit(pd.DataFrame()),
    )
    assert result.unscheduled_tasks
    assert result.utilization <= 1.0


def test_event_is_protected_from_task_placement() -> None:
    target = date(2026, 8, 3)
    event = Event(
        "Project review",
        datetime.combine(target, time(9, 0)),
        datetime.combine(target, time(10, 30)),
        importance=5,
    )
    result = generate_schedule(
        [Task("Deep work", "Coding", 5, 60, 5)],
        target_date=target,
        settings=UserSettings(day_start_hour=9, day_end_hour=18),
        profile=_profile(),
        predictor=AdaptiveDurationPredictor().fit(pd.DataFrame()),
        events=[event],
    )
    event_block = next(block for block in result.blocks if block.kind == "event")
    task_block = next(block for block in result.blocks if block.kind == "task")
    assert event_block.end <= task_block.start or task_block.end <= event_block.start
