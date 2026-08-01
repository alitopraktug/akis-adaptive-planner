"""Tests for weekly distribution and the local planning assistant."""

from datetime import date, datetime, time

import pandas as pd

from assistant import answer_question
from learning_engine import AdaptiveDurationPredictor, BehaviorProfile
from models import Event, Task, UserSettings
from weekly_planner import generate_weekly_plan


def _profile() -> BehaviorProfile:
    return BehaviorProfile(
        total_sessions=30,
        duration_factors={"Coding": 1.0, "Study": 1.0},
        completion_by_hour={9: 0.9, 10: 0.85},
        focus_by_hour={9: 4.8, 10: 4.5},
        preferred_hours={"Coding": 9, "Study": 10},
        confidence=0.6,
        insights=(),
        completion_by_weekday={0: 0.92, 1: 0.65, 2: 0.78, 3: 0.60, 4: 0.70},
        preferred_weekdays={"Coding": 0, "Study": 2},
    )


def _plan():
    monday = date(2026, 8, 3)
    tasks = [
        Task("Build planner", "Coding", 5, 90, 5),
        Task("Study graphs", "Study", 4, 75, 4),
        Task("Write tests", "Coding", 4, 60, 4),
    ]
    events = [
        Event(
            "Project review",
            datetime.combine(monday + pd.Timedelta(days=1), time(14, 0)),
            datetime.combine(monday + pd.Timedelta(days=1), time(15, 0)),
            importance=5,
        )
    ]
    plan = generate_weekly_plan(
        tasks,
        events=events,
        week_start=monday,
        active_weekdays=range(5),
        settings=UserSettings(day_start_hour=9, day_end_hour=18),
        profile=_profile(),
        predictor=AdaptiveDurationPredictor().fit(pd.DataFrame()),
        planning_date=monday,
    )
    return plan, tasks, events


def test_weekly_plan_schedules_each_task_at_most_once() -> None:
    plan, tasks, _ = _plan()
    scheduled_ids = [
        block.task_id
        for schedule in plan.schedules.values()
        for block in schedule.blocks
        if block.task_id is not None
    ]
    assert len(scheduled_ids) == len(set(scheduled_ids))
    assert set(scheduled_ids) == {task.task_id for task in tasks}
    assert any(block.kind == "event" for block in plan.schedules[date(2026, 8, 4)].blocks)


def test_coach_answers_day_and_preference_questions_in_turkish() -> None:
    plan, tasks, events = _plan()
    day_answer = answer_question(
        "Pazartesi nasıl?",
        weekly_plan=plan,
        profile=_profile(),
        tasks=tasks,
        events=events,
    )
    preference_answer = answer_question(
        "Coding için hangi gün daha iyi?",
        weekly_plan=plan,
        profile=_profile(),
        tasks=tasks,
        events=events,
    )
    assert "Pazartesi" in day_answer
    assert "Pazartesi" in preference_answer
    assert "Coding" in preference_answer
