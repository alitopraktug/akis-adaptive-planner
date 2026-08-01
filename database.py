"""SQLite persistence layer for Akis."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping

import pandas as pd

from models import Event, FocusSession, SessionOutcome, Task, TaskStatus, UserSettings


class AkisDatabase:
    """Small, explicit SQLite repository for tasks, sessions, and settings."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the database schema and query-driven indexes."""

        statements = (
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                priority INTEGER NOT NULL CHECK(priority BETWEEN 1 AND 5),
                estimated_minutes INTEGER NOT NULL CHECK(estimated_minutes BETWEEN 5 AND 720),
                energy_required INTEGER NOT NULL CHECK(energy_required BETWEEN 1 AND 5),
                deadline TEXT,
                preferred_period TEXT NOT NULL,
                splittable INTEGER NOT NULL,
                fixed_start TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                task_id TEXT,
                task_title TEXT NOT NULL,
                category TEXT NOT NULL,
                planned_minutes INTEGER NOT NULL,
                actual_minutes INTEGER NOT NULL,
                start_time TEXT NOT NULL,
                outcome TEXT NOT NULL,
                focus_rating INTEGER NOT NULL CHECK(focus_rating BETWEEN 1 AND 5),
                interruption_count INTEGER NOT NULL DEFAULT 0,
                priority INTEGER NOT NULL,
                energy_required INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                importance INTEGER NOT NULL CHECK(importance BETWEEN 1 AND 5),
                category TEXT NOT NULL,
                location TEXT NOT NULL,
                notes TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_tasks_status_deadline ON tasks(status, deadline)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_category_start ON sessions(category, start_time)",
            "CREATE INDEX IF NOT EXISTS idx_events_start_time ON events(start_time)",
        )
        with self._connection() as connection:
            for statement in statements:
                connection.execute(statement)
            connection.execute("PRAGMA optimize")

    def add_task(self, task: Task) -> str:
        """Insert a validated task and return its identifier."""

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, title, category, priority, estimated_minutes,
                    energy_required, deadline, preferred_period, splittable,
                    fixed_start, status, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.title.strip(),
                    task.category.strip(),
                    task.priority,
                    task.estimated_minutes,
                    task.energy_required,
                    task.deadline.isoformat() if task.deadline else None,
                    task.preferred_period,
                    int(task.splittable),
                    task.fixed_start.isoformat() if task.fixed_start else None,
                    task.status.value,
                    task.created_at.isoformat(),
                    task.completed_at.isoformat() if task.completed_at else None,
                ),
            )
        return task.task_id

    def get_task(self, task_id: str) -> Task | None:
        """Return one task or ``None`` when it does not exist."""

        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._row_to_task(row) if row else None

    def list_tasks(self, status: TaskStatus | None = None) -> list[Task]:
        """Return tasks ordered by urgency and creation time."""

        query = "SELECT * FROM tasks"
        parameters: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status.value,)
        query += " ORDER BY CASE WHEN deadline IS NULL THEN 1 ELSE 0 END, deadline, priority DESC, created_at"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row_to_task(row) for row in rows]

    def count_tasks(self) -> int:
        """Return total task count."""

        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update a task status with a completion timestamp when appropriate."""

        completed_at = datetime.now().isoformat() if status == TaskStatus.COMPLETED else None
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET status = ?, completed_at = ? WHERE task_id = ?",
                (status.value, completed_at, task_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Task '{task_id}' does not exist.")

    def record_session(self, session: FocusSession) -> str:
        """Persist observed task behavior and return the session identifier."""

        if session.task_id is not None and self.get_task(session.task_id) is None:
            raise ValueError(f"Task '{session.task_id}' does not exist.")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, task_id, task_title, category, planned_minutes,
                    actual_minutes, start_time, outcome, focus_rating,
                    interruption_count, priority, energy_required, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.task_id,
                    session.task_title.strip(),
                    session.category.strip(),
                    session.planned_minutes,
                    session.actual_minutes,
                    session.start_time.isoformat(),
                    session.outcome.value,
                    session.focus_rating,
                    session.interruption_count,
                    session.priority,
                    session.energy_required,
                    session.created_at.isoformat(),
                ),
            )
        return session.session_id

    def add_event(self, event: Event) -> str:
        """Insert a validated calendar event and return its identifier."""

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO events (
                    event_id, title, start_time, end_time, importance,
                    category, location, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.title.strip(),
                    event.start.isoformat(),
                    event.end.isoformat(),
                    event.importance,
                    event.category.strip(),
                    event.location.strip(),
                    event.notes.strip(),
                    event.created_at.isoformat(),
                ),
            )
        return event.event_id

    def list_events(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Event]:
        """Return events intersecting an optional half-open time range."""

        clauses: list[str] = []
        parameters: list[str] = []
        if start is not None:
            clauses.append("end_time > ?")
            parameters.append(start.isoformat())
        if end is not None:
            clauses.append("start_time < ?")
            parameters.append(end.isoformat())
        query = "SELECT * FROM events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY start_time, importance DESC"
        with self._connection() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [self._row_to_event(row) for row in rows]

    def count_events(self) -> int:
        """Return total event count."""

        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def list_sessions(self, since: datetime | None = None) -> pd.DataFrame:
        """Return behavior history as a typed DataFrame."""

        query = "SELECT * FROM sessions"
        parameters: tuple[Any, ...] = ()
        if since is not None:
            query += " WHERE start_time >= ?"
            parameters = (since.isoformat(),)
        query += " ORDER BY start_time"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        columns = [
            "session_id",
            "task_id",
            "task_title",
            "category",
            "planned_minutes",
            "actual_minutes",
            "start_time",
            "outcome",
            "focus_rating",
            "interruption_count",
            "priority",
            "energy_required",
            "created_at",
        ]
        frame = pd.DataFrame([dict(row) for row in rows], columns=columns)
        if not frame.empty:
            frame["start_time"] = pd.to_datetime(frame["start_time"])
            frame["created_at"] = pd.to_datetime(frame["created_at"])
        return frame

    def save_settings(self, settings: UserSettings) -> None:
        """Persist scheduling settings as versionable JSON."""

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO settings(setting_key, setting_value, updated_at)
                VALUES ('user_settings', ?, ?)
                ON CONFLICT(setting_key) DO UPDATE SET
                    setting_value = excluded.setting_value,
                    updated_at = excluded.updated_at
                """,
                (json.dumps(settings.to_dict()), datetime.now().isoformat()),
            )

    def load_settings(self) -> UserSettings:
        """Load scheduling settings or return safe defaults."""

        with self._connection() as connection:
            row = connection.execute(
                "SELECT setting_value FROM settings WHERE setting_key = 'user_settings'"
            ).fetchone()
        if row is None:
            return UserSettings()
        try:
            values = json.loads(row["setting_value"])
            return UserSettings(**values)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Stored scheduling settings are invalid.") from exc

    def export_snapshot(self) -> dict[str, Any]:
        """Export all user-owned state as a JSON-compatible dictionary."""

        return {
            "version": 2,
            "exported_at": datetime.now().isoformat(),
            "settings": self.load_settings().to_dict(),
            "tasks": [task.to_dict() for task in self.list_tasks()],
            "events": [event.to_dict() for event in self.list_events()],
            "sessions": self.list_sessions().assign(
                start_time=lambda frame: frame["start_time"].astype(str),
                created_at=lambda frame: frame["created_at"].astype(str),
            ).to_dict(orient="records"),
        }

    @staticmethod
    def _row_to_task(row: Mapping[str, Any]) -> Task:
        return Task(
            task_id=str(row["task_id"]),
            title=str(row["title"]),
            category=str(row["category"]),
            priority=int(row["priority"]),
            estimated_minutes=int(row["estimated_minutes"]),
            energy_required=int(row["energy_required"]),
            deadline=datetime.fromisoformat(row["deadline"]) if row["deadline"] else None,
            preferred_period=str(row["preferred_period"]),  # type: ignore[arg-type]
            splittable=bool(row["splittable"]),
            fixed_start=datetime.fromisoformat(row["fixed_start"]) if row["fixed_start"] else None,
            status=TaskStatus(str(row["status"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
        )

    @staticmethod
    def _row_to_event(row: Mapping[str, Any]) -> Event:
        return Event(
            event_id=str(row["event_id"]),
            title=str(row["title"]),
            start=datetime.fromisoformat(str(row["start_time"])),
            end=datetime.fromisoformat(str(row["end_time"])),
            importance=int(row["importance"]),
            category=str(row["category"]),
            location=str(row["location"]),
            notes=str(row["notes"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )
