"""Constraint-aware and behavior-adaptive daily scheduling engine."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import math
from typing import Iterable

import numpy as np

from learning_engine import AdaptiveDurationPredictor, BehaviorProfile, completion_probability
from models import Event, ScheduleBlock, ScheduleResult, Task, TaskStatus, UserSettings

SLOT_MINUTES = 15


def _round_up(value: int, step: int = SLOT_MINUTES) -> int:
    return max(step, int(math.ceil(value / step) * step))


def _overlaps(start: datetime, end: datetime, blocks: Iterable[ScheduleBlock]) -> bool:
    return any(start < block.end and end > block.start for block in blocks)


def _period_match(task: Task, hour: int) -> float:
    if task.preferred_period == "any":
        return 1.0
    if task.preferred_period == "morning":
        return 1.0 if 6 <= hour < 12 else 0.45
    if task.preferred_period == "afternoon":
        return 1.0 if 12 <= hour < 17 else 0.45
    return 1.0 if 17 <= hour < 22 else 0.45


def calculate_slot_score(
    task: Task,
    start: datetime,
    target_date: date,
    profile: BehaviorProfile,
) -> float:
    """Calculate an explainable suitability score for a task-time pairing."""

    priority_score = task.priority * 18.0
    deadline_score = 0.0
    if task.deadline is not None:
        hours_until_deadline = (task.deadline - start).total_seconds() / 3_600
        if hours_until_deadline <= 0:
            deadline_score = 48.0
        elif hours_until_deadline <= 24:
            deadline_score = 38.0
        elif hours_until_deadline <= 72:
            deadline_score = 24.0
        elif task.deadline.date() <= target_date + timedelta(days=7):
            deadline_score = 12.0

    probability = completion_probability(task, start.hour, profile, target_date.weekday())
    completion_score = probability * 30.0
    period_score = _period_match(task, start.hour) * 14.0
    focus_level = profile.focus_by_hour.get(start.hour, 3.2)
    energy_gap = abs(task.energy_required - focus_level)
    energy_score = max(0.0, 16.0 - energy_gap * 4.0)
    return round(priority_score + deadline_score + completion_score + period_score + energy_score, 3)


def _reason_for_block(task: Task, start: datetime, profile: BehaviorProfile) -> str:
    reasons: list[str] = []
    if task.priority >= 4:
        reasons.append("high priority")
    if task.deadline and (task.deadline - start).total_seconds() <= 86_400:
        reasons.append("deadline approaching")
    preferred_hour = profile.preferred_hours.get(task.category)
    if preferred_hour is not None and abs(start.hour - preferred_hour) <= 1:
        reasons.append("matches your learned work pattern")
    if _period_match(task, start.hour) == 1.0 and task.preferred_period != "any":
        reasons.append("matches preferred time")
    return ", ".join(reasons) if reasons else "best available fit"


def _candidate_starts(day_start: datetime, day_end: datetime) -> list[datetime]:
    starts: list[datetime] = []
    cursor = day_start
    while cursor < day_end:
        starts.append(cursor)
        cursor += timedelta(minutes=SLOT_MINUTES)
    return starts


def generate_schedule(
    tasks: list[Task],
    *,
    target_date: date,
    settings: UserSettings,
    profile: BehaviorProfile,
    predictor: AdaptiveDurationPredictor,
    events: list[Event] | None = None,
) -> ScheduleResult:
    """Generate a non-overlapping daily plan from tasks and learned behavior."""

    if not isinstance(target_date, date):
        raise TypeError("target_date must be a date.")
    pending = [task for task in tasks if task.status == TaskStatus.PENDING]
    day_start = datetime.combine(target_date, time(settings.day_start_hour))
    day_end = datetime.combine(target_date, time(settings.day_end_hour))
    blocks: list[ScheduleBlock] = []
    warnings: list[str] = []

    day_events = sorted(
        [event for event in (events or []) if event.start.date() == target_date],
        key=lambda event: event.start,
    )
    for event in day_events:
        if _overlaps(event.start, event.end, blocks):
            warnings.append(f"Event '{event.title}' conflicts with another event.")
            continue
        if event.start < day_start or event.end > day_end:
            warnings.append(f"Event '{event.title}' falls outside working hours.")
        blocks.append(
            ScheduleBlock(
                title=event.title,
                start=event.start,
                end=event.end,
                kind="event",
                event_id=event.event_id,
                category=event.category,
                score=250.0 + event.importance,
                reason=f"protected event · importance {event.importance}/5",
            )
        )

    if settings.lunch_minutes > 0 and settings.day_start_hour <= settings.lunch_start_hour < settings.day_end_hour:
        lunch_start = datetime.combine(target_date, time(settings.lunch_start_hour))
        lunch_end = min(day_end, lunch_start + timedelta(minutes=settings.lunch_minutes))
        if not _overlaps(lunch_start, lunch_end, blocks):
            blocks.append(
                ScheduleBlock(
                    title="Lunch",
                    start=lunch_start,
                    end=lunch_end,
                    kind="lunch",
                    reason="protected recovery time",
                )
            )
        else:
            warnings.append("Lunch was omitted because it conflicts with a protected event.")

    fixed_tasks = [
        task for task in pending if task.fixed_start is not None and task.fixed_start.date() == target_date
    ]
    scheduled_ids: set[str] = set()
    for task in sorted(fixed_tasks, key=lambda item: item.fixed_start or day_start):
        assert task.fixed_start is not None
        duration = _round_up(task.estimated_minutes)
        end = task.fixed_start + timedelta(minutes=duration)
        if task.fixed_start < day_start or end > day_end:
            warnings.append(f"Fixed task '{task.title}' falls outside working hours.")
            continue
        if _overlaps(task.fixed_start, end, blocks):
            warnings.append(f"Fixed task '{task.title}' conflicts with another protected block.")
            continue
        blocks.append(
            ScheduleBlock(
                title=task.title,
                start=task.fixed_start,
                end=end,
                kind="fixed",
                task_id=task.task_id,
                category=task.category,
                score=200.0,
                reason="fixed-time commitment",
                predicted_minutes=duration,
            )
        )
        scheduled_ids.add(task.task_id)

    flexible = [task for task in pending if task.task_id not in scheduled_ids and not task.is_fixed]
    flexible.sort(
        key=lambda task: (
            -task.priority,
            task.deadline or datetime.max,
            task.created_at,
        )
    )

    candidate_starts = _candidate_starts(day_start, day_end)
    for task in flexible:
        candidates: list[tuple[float, datetime, int]] = []
        for candidate in candidate_starts:
            predicted = predictor.predict(task, candidate.hour, profile)
            duration = min(predicted, settings.max_focus_minutes) if task.splittable else predicted
            duration = _round_up(duration)
            end = candidate + timedelta(minutes=duration)
            if end > day_end or _overlaps(candidate, end, blocks):
                continue
            score = calculate_slot_score(task, candidate, target_date, profile)
            candidates.append((score, candidate, duration))
        if not candidates:
            continue
        score, chosen_start, duration = max(candidates, key=lambda item: (item[0], -item[1].timestamp()))
        chosen_end = chosen_start + timedelta(minutes=duration)
        blocks.append(
            ScheduleBlock(
                title=task.title,
                start=chosen_start,
                end=chosen_end,
                kind="task",
                task_id=task.task_id,
                category=task.category,
                score=score,
                reason=_reason_for_block(task, chosen_start, profile),
                predicted_minutes=duration,
            )
        )
        scheduled_ids.add(task.task_id)

        if settings.break_minutes > 0:
            break_end = chosen_end + timedelta(minutes=settings.break_minutes)
            if break_end <= day_end and not _overlaps(chosen_end, break_end, blocks):
                blocks.append(
                    ScheduleBlock(
                        title="Recovery break",
                        start=chosen_end,
                        end=break_end,
                        kind="break",
                        reason="protects focus quality",
                    )
                )

    blocks.sort(key=lambda block: block.start)
    task_minutes = sum(block.duration_minutes for block in blocks if block.kind in {"task", "fixed"})
    available_minutes = int((day_end - day_start).total_seconds() // 60)
    protected_minutes = sum(block.duration_minutes for block in blocks if block.kind == "lunch")
    usable_minutes = max(1, available_minutes - protected_minutes)
    utilization = float(np.clip(task_minutes / usable_minutes, 0.0, 1.0))
    unscheduled = tuple(task for task in pending if task.task_id not in scheduled_ids)
    if unscheduled:
        warnings.append(f"{len(unscheduled)} task(s) could not fit within the selected day.")

    return ScheduleResult(
        target_date=target_date,
        blocks=tuple(blocks),
        unscheduled_tasks=unscheduled,
        utilization=round(utilization, 3),
        task_minutes=task_minutes,
        available_minutes=available_minutes,
        warnings=tuple(warnings),
    )
