"""Tests for Akis SQLite persistence."""

from datetime import datetime

from database import AkisDatabase
from models import Event, FocusSession, SessionOutcome, Task, TaskStatus, UserSettings


def test_task_round_trip(tmp_path) -> None:
    database = AkisDatabase(tmp_path / "akis.db")
    task = Task(
        title="Write scheduler tests",
        category="Coding",
        priority=5,
        estimated_minutes=60,
        energy_required=4,
    )
    task_id = database.add_task(task)
    loaded = database.get_task(task_id)
    assert loaded == task


def test_status_update_and_session_history(tmp_path) -> None:
    database = AkisDatabase(tmp_path / "akis.db")
    task = Task("Study algorithms", "Study", 4, 45, 4)
    database.add_task(task)
    database.update_task_status(task.task_id, TaskStatus.COMPLETED)
    database.record_session(
        FocusSession(
            task_id=task.task_id,
            task_title=task.title,
            category=task.category,
            planned_minutes=45,
            actual_minutes=50,
            start_time=datetime(2026, 8, 1, 9, 0),
            outcome=SessionOutcome.COMPLETED,
            focus_rating=4,
            priority=task.priority,
            energy_required=task.energy_required,
        )
    )
    assert database.get_task(task.task_id).status == TaskStatus.COMPLETED
    sessions = database.list_sessions()
    assert len(sessions) == 1
    assert sessions.iloc[0]["actual_minutes"] == 50


def test_settings_round_trip(tmp_path) -> None:
    database = AkisDatabase(tmp_path / "akis.db")
    settings = UserSettings(day_start_hour=8, day_end_hour=20, break_minutes=15)
    database.save_settings(settings)
    assert database.load_settings() == settings


def test_export_snapshot_contains_user_state(tmp_path) -> None:
    database = AkisDatabase(tmp_path / "akis.db")
    database.add_task(Task("Plan week", "Planning", 3, 30, 2))
    snapshot = database.export_snapshot()
    assert snapshot["version"] == 2
    assert len(snapshot["tasks"]) == 1
    assert "settings" in snapshot


def test_event_round_trip_and_snapshot(tmp_path) -> None:
    database = AkisDatabase(tmp_path / "akis.db")
    event = Event(
        "Architecture review",
        datetime(2026, 8, 3, 14, 0),
        datetime(2026, 8, 3, 15, 0),
        importance=5,
        location="Studio 2",
    )
    database.add_event(event)
    assert database.list_events() == [event]
    assert database.export_snapshot()["events"][0]["event_id"] == event.event_id
