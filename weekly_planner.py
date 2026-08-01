"""Behavior-aware weekly planning for Akis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable

from learning_engine import AdaptiveDurationPredictor, BehaviorProfile
from models import Event, ScheduleResult, Task, TaskStatus, UserSettings
from scheduler import generate_schedule


@dataclass(frozen=True)
class WeeklyPlan:
    """A seven-day plan with events and tasks scheduled exactly once."""

    week_start: date
    schedules: dict[date, ScheduleResult]
    events: tuple[Event, ...]
    unscheduled_tasks: tuple[Task, ...]

    @property
    def task_minutes(self) -> int:
        """Return total planned task minutes for the week."""

        return sum(schedule.task_minutes for schedule in self.schedules.values())

    @property
    def average_utilization(self) -> float:
        """Return mean utilization across active plan days."""

        values = [schedule.utilization for schedule in self.schedules.values()]
        return sum(values) / len(values) if values else 0.0


def monday_of(value: date) -> date:
    """Return the Monday anchoring the week containing ``value``."""

    if not isinstance(value, date):
        raise TypeError("value must be a date.")
    return value - timedelta(days=value.weekday())


def _day_capacity(
    day: date,
    settings: UserSettings,
    events: Iterable[Event],
    target_utilization: float,
) -> int:
    work_minutes = (settings.day_end_hour - settings.day_start_hour) * 60
    lunch = settings.lunch_minutes if settings.lunch_minutes > 0 else 0
    event_minutes = 0
    work_start = datetime.combine(day, time(settings.day_start_hour))
    work_end = datetime.combine(day, time(settings.day_end_hour))
    for event in events:
        if event.start.date() != day:
            continue
        overlap_start = max(work_start, event.start)
        overlap_end = min(work_end, event.end)
        if overlap_end > overlap_start:
            event_minutes += int((overlap_end - overlap_start).total_seconds() // 60)
    return max(0, int(max(0, work_minutes - lunch - event_minutes) * target_utilization))


def _day_fit(
    task: Task,
    day: date,
    load_minutes: int,
    capacity_minutes: int,
    profile: BehaviorProfile,
) -> float:
    """Score a day for a task using urgency, history, and remaining capacity."""

    weekday = day.weekday()
    score = task.priority * 16.0
    score += profile.completion_by_weekday.get(weekday, 0.65) * 24.0
    preferred = profile.preferred_weekdays.get(task.category)
    score += 18.0 if preferred == weekday else (8.0 if preferred is None else 0.0)
    if task.deadline is not None:
        days_to_deadline = (task.deadline.date() - day).days
        if days_to_deadline < 0:
            score -= 120.0
        elif days_to_deadline == 0:
            score += 48.0
        elif days_to_deadline <= 2:
            score += 30.0 - days_to_deadline * 6.0
        else:
            score += max(0.0, 12.0 - days_to_deadline)
    load_ratio = load_minutes / max(1, capacity_minutes)
    score -= load_ratio * 42.0
    score -= (day - monday_of(day)).days * 0.25
    return score


def generate_weekly_plan(
    tasks: list[Task],
    *,
    events: list[Event],
    week_start: date,
    active_weekdays: Iterable[int],
    settings: UserSettings,
    profile: BehaviorProfile,
    predictor: AdaptiveDurationPredictor,
    target_utilization: float = 0.74,
    planning_date: date | None = None,
) -> WeeklyPlan:
    """Distribute pending tasks across one week and generate daily schedules."""

    if not 0.25 <= target_utilization <= 1.0:
        raise ValueError("target_utilization must be between 0.25 and 1.0.")
    monday = monday_of(week_start)
    planning_floor = planning_date or date.today()
    active = set(active_weekdays)
    if not active or any(day not in range(7) for day in active):
        raise ValueError("active_weekdays must contain values between 0 and 6.")

    days = [monday + timedelta(days=offset) for offset in range(7)]
    week_end = monday + timedelta(days=7)
    week_events = [event for event in events if monday <= event.start.date() < week_end]
    pending = [task for task in tasks if task.status == TaskStatus.PENDING]
    assignments: dict[date, list[Task]] = {day: [] for day in days}
    assigned_ids: set[str] = set()

    for task in pending:
        if (
            task.fixed_start is not None
            and max(monday, planning_floor) <= task.fixed_start.date() < week_end
        ):
            assignments[task.fixed_start.date()].append(task)
            assigned_ids.add(task.task_id)

    capacities = {
        day: _day_capacity(day, settings, week_events, target_utilization)
        for day in days
    }
    loads = {
        day: sum(task.estimated_minutes for task in assignments[day])
        for day in days
    }
    flexible = [task for task in pending if task.task_id not in assigned_ids and not task.is_fixed]
    flexible.sort(key=lambda task: (-task.priority, task.deadline or datetime.max, task.created_at))

    for task in flexible:
        eligible_days = [
            day
            for day in days
            if day.weekday() in active
            and day >= planning_floor
            and loads[day] + task.estimated_minutes <= capacities[day]
            and (task.deadline is None or day <= task.deadline.date())
        ]
        if not eligible_days:
            continue
        chosen = max(
            eligible_days,
            key=lambda day: _day_fit(task, day, loads[day], capacities[day], profile),
        )
        assignments[chosen].append(task)
        loads[chosen] += task.estimated_minutes
        assigned_ids.add(task.task_id)

    schedules: dict[date, ScheduleResult] = {}
    actually_scheduled: set[str] = set()
    for day in days:
        schedule = generate_schedule(
            assignments[day],
            target_date=day,
            settings=settings,
            profile=profile,
            predictor=predictor,
            events=week_events,
        )
        schedules[day] = schedule
        actually_scheduled.update(
            block.task_id for block in schedule.blocks if block.task_id is not None
        )

    unscheduled = tuple(task for task in pending if task.task_id not in actually_scheduled)
    return WeeklyPlan(
        week_start=monday,
        schedules=schedules,
        events=tuple(sorted(week_events, key=lambda event: event.start)),
        unscheduled_tasks=unscheduled,
    )
