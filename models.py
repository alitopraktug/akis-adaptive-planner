"""Domain models used by the Akis adaptive planning engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

PreferredPeriod = Literal["any", "morning", "afternoon", "evening"]


class TaskStatus(str, Enum):
    """Supported task lifecycle states."""

    PENDING = "pending"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    ARCHIVED = "archived"


class SessionOutcome(str, Enum):
    """Supported focus-session outcomes."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Event:
    """A time-bound calendar event protected by the planning engine."""

    title: str
    start: datetime
    end: datetime
    importance: int = 3
    category: str = "Meeting"
    location: str = ""
    notes: str = ""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate event values at the domain boundary."""

        if not self.title.strip():
            raise ValueError("Event title cannot be empty.")
        if self.end <= self.start:
            raise ValueError("Event end time must be later than start time.")
        if not 1 <= self.importance <= 5:
            raise ValueError("Event importance must be between 1 and 5.")
        if not self.category.strip():
            raise ValueError("Event category cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        output = asdict(self)
        for key in ("start", "end", "created_at"):
            output[key] = output[key].isoformat()
        return output


@dataclass(frozen=True)
class Task:
    """A schedulable unit of work."""

    title: str
    category: str
    priority: int
    estimated_minutes: int
    energy_required: int
    deadline: datetime | None = None
    preferred_period: PreferredPeriod = "any"
    splittable: bool = True
    fixed_start: datetime | None = None
    status: TaskStatus = TaskStatus.PENDING
    task_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate task values at the domain boundary."""

        if not self.title.strip():
            raise ValueError("Task title cannot be empty.")
        if not self.category.strip():
            raise ValueError("Task category cannot be empty.")
        if not 1 <= self.priority <= 5:
            raise ValueError("Task priority must be between 1 and 5.")
        if not 5 <= self.estimated_minutes <= 720:
            raise ValueError("Estimated duration must be between 5 and 720 minutes.")
        if not 1 <= self.energy_required <= 5:
            raise ValueError("Energy requirement must be between 1 and 5.")
        if self.preferred_period not in {"any", "morning", "afternoon", "evening"}:
            raise ValueError("Preferred period must be any, morning, afternoon, or evening.")

    @property
    def is_fixed(self) -> bool:
        """Return whether the task has a fixed start time."""

        return self.fixed_start is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        output = asdict(self)
        output["status"] = self.status.value
        for key in ("deadline", "fixed_start", "created_at", "completed_at"):
            value = output[key]
            output[key] = value.isoformat() if value is not None else None
        return output


@dataclass(frozen=True)
class FocusSession:
    """Observed user behavior from one planned or completed work session."""

    task_title: str
    category: str
    planned_minutes: int
    actual_minutes: int
    start_time: datetime
    outcome: SessionOutcome
    focus_rating: int
    interruption_count: int = 0
    priority: int = 3
    energy_required: int = 3
    task_id: str | None = None
    session_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not self.task_title.strip():
            raise ValueError("Session task title cannot be empty.")
        if not self.category.strip():
            raise ValueError("Session category cannot be empty.")
        if not 5 <= self.planned_minutes <= 720:
            raise ValueError("Planned session duration must be between 5 and 720 minutes.")
        if not 0 <= self.actual_minutes <= 1_440:
            raise ValueError("Actual session duration must be between 0 and 1,440 minutes.")
        if not 1 <= self.focus_rating <= 5:
            raise ValueError("Focus rating must be between 1 and 5.")
        if self.interruption_count < 0:
            raise ValueError("Interruption count cannot be negative.")
        if not 1 <= self.priority <= 5 or not 1 <= self.energy_required <= 5:
            raise ValueError("Session priority and energy requirement must be between 1 and 5.")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        output = asdict(self)
        output["outcome"] = self.outcome.value
        output["start_time"] = self.start_time.isoformat()
        output["created_at"] = self.created_at.isoformat()
        return output


@dataclass(frozen=True)
class UserSettings:
    """User-controlled constraints for daily scheduling."""

    day_start_hour: int = 9
    day_end_hour: int = 19
    break_minutes: int = 10
    max_focus_minutes: int = 90
    lunch_start_hour: int = 13
    lunch_minutes: int = 45

    def __post_init__(self) -> None:
        if not 0 <= self.day_start_hour <= 22:
            raise ValueError("Day start hour must be between 0 and 22.")
        if not 1 <= self.day_end_hour <= 23:
            raise ValueError("Day end hour must be between 1 and 23.")
        if self.day_start_hour >= self.day_end_hour:
            raise ValueError("Day end hour must be later than day start hour.")
        if not 0 <= self.break_minutes <= 60:
            raise ValueError("Break duration must be between 0 and 60 minutes.")
        if not 15 <= self.max_focus_minutes <= 240:
            raise ValueError("Maximum focus duration must be between 15 and 240 minutes.")
        if not 0 <= self.lunch_start_hour <= 23:
            raise ValueError("Lunch start hour must be between 0 and 23.")
        if not 0 <= self.lunch_minutes <= 180:
            raise ValueError("Lunch duration must be between 0 and 180 minutes.")

    def to_dict(self) -> dict[str, int]:
        """Return settings as a plain mapping."""

        return asdict(self)


@dataclass(frozen=True)
class ScheduleBlock:
    """One scheduled interval in a generated daily plan."""

    title: str
    start: datetime
    end: datetime
    kind: Literal["task", "fixed", "event", "break", "lunch"]
    task_id: str | None = None
    event_id: str | None = None
    category: str = "Planning"
    score: float = 0.0
    reason: str = ""
    predicted_minutes: int = 0

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("Schedule block end time must be later than start time.")

    @property
    def duration_minutes(self) -> int:
        """Return block duration in whole minutes."""

        return int((self.end - self.start).total_seconds() // 60)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        output = asdict(self)
        output["start"] = self.start.isoformat()
        output["end"] = self.end.isoformat()
        output["duration_minutes"] = self.duration_minutes
        return output


@dataclass(frozen=True)
class ScheduleResult:
    """Complete output from one scheduling run."""

    target_date: date
    blocks: tuple[ScheduleBlock, ...]
    unscheduled_tasks: tuple[Task, ...]
    utilization: float
    task_minutes: int
    available_minutes: int
    warnings: tuple[str, ...] = ()
