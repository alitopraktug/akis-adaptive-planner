"""Local, behavior-aware planning assistant for Akis."""

from __future__ import annotations

from datetime import date, timedelta
import re

from learning_engine import BehaviorProfile
from models import Event, Task
from weekly_planner import WeeklyPlan

WEEKDAYS_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
WEEKDAYS_TR = ("Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar")
DAY_ALIASES = {
    0: ("monday", "pazartesi"),
    1: ("tuesday", "salı", "sali"),
    2: ("wednesday", "çarşamba", "carsamba"),
    3: ("thursday", "perşembe", "persembe"),
    4: ("friday", "cuma"),
    5: ("saturday", "cumartesi"),
    6: ("sunday", "pazar"),
}


def _is_turkish(question: str) -> bool:
    normalized = question.casefold()
    markers = ("nasıl", "hangi", "gün", "bugün", "yarın", "plan", "yoğun", "çalış", "öner")
    return any(marker in normalized for marker in markers) or bool(re.search(r"[çğıöşü]", normalized))


def _mentioned_day(question: str) -> int | None:
    normalized = question.casefold()
    for weekday, aliases in DAY_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return weekday
    return None


def _day_summary(day: date, plan: WeeklyPlan, turkish: bool) -> str:
    schedule = plan.schedules[day]
    task_blocks = [block for block in schedule.blocks if block.kind in {"task", "fixed"}]
    event_blocks = [block for block in schedule.blocks if block.kind == "event"]
    name = WEEKDAYS_TR[day.weekday()] if turkish else WEEKDAYS_EN[day.weekday()]
    hours = schedule.task_minutes / 60
    if turkish:
        response = f"{name} günü {len(task_blocks)} görev, {hours:.1f} saat odak çalışması ve {len(event_blocks)} event içeriyor. Doluluk {schedule.utilization:.0%}."
        if event_blocks:
            response += " Önemli zaman blokları: " + ", ".join(
                f"{block.title} ({block.start:%H:%M})" for block in event_blocks
            ) + "."
        if task_blocks:
            response += f" İlk odak görevin {task_blocks[0].title}."
        return response
    response = f"{name} contains {len(task_blocks)} tasks, {hours:.1f} focus hours, and {len(event_blocks)} events. Utilization is {schedule.utilization:.0%}."
    if event_blocks:
        response += " Key time blocks: " + ", ".join(
            f"{block.title} ({block.start:%H:%M})" for block in event_blocks
        ) + "."
    if task_blocks:
        response += f" Your first focus task is {task_blocks[0].title}."
    return response


def _category_in_question(question: str, tasks: list[Task]) -> str | None:
    normalized = question.casefold()
    categories = sorted({task.category for task in tasks}, key=len, reverse=True)
    return next((category for category in categories if category.casefold() in normalized), None)


def _recommended_day(category: str | None, profile: BehaviorProfile) -> int:
    if category and category in profile.preferred_weekdays:
        return profile.preferred_weekdays[category]
    if profile.completion_by_weekday:
        return max(profile.completion_by_weekday, key=profile.completion_by_weekday.get)  # type: ignore[arg-type]
    return 0


def answer_question(
    question: str,
    *,
    weekly_plan: WeeklyPlan,
    profile: BehaviorProfile,
    tasks: list[Task],
    events: list[Event],
) -> str:
    """Answer common planning questions from local schedule and behavior data."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")
    normalized = question.casefold().strip()
    turkish = _is_turkish(question)
    mentioned = _mentioned_day(question)
    if mentioned is not None:
        day = weekly_plan.week_start + timedelta(days=mentioned)
        return _day_summary(day, weekly_plan, turkish)

    if any(word in normalized for word in ("yoğun", "busiest", "dolu")):
        day = max(weekly_plan.schedules, key=lambda value: weekly_plan.schedules[value].utilization)
        return (
            f"En yoğun gün {WEEKDAYS_TR[day.weekday()]}; doluluk {weekly_plan.schedules[day].utilization:.0%}. Yeni iş eklemek yerine daha sakin bir güne taşımanı öneririm."
            if turkish
            else f"{WEEKDAYS_EN[day.weekday()]} is the busiest day at {weekly_plan.schedules[day].utilization:.0%} utilization. Move new work to a quieter day."
        )

    if any(word in normalized for word in ("sakin", "boş", "quiet", "free")):
        day = min(weekly_plan.schedules, key=lambda value: weekly_plan.schedules[value].utilization)
        return (
            f"En sakin gün {WEEKDAYS_TR[day.weekday()]}; doluluk {weekly_plan.schedules[day].utilization:.0%}. Esnek bir görev için en rahat alan burada."
            if turkish
            else f"{WEEKDAYS_EN[day.weekday()]} is the quietest day at {weekly_plan.schedules[day].utilization:.0%} utilization. It has the best room for flexible work."
        )

    category = _category_in_question(question, tasks)
    if category or any(word in normalized for word in ("hangi gün", "which day", "ne zaman", "when")):
        weekday = _recommended_day(category, profile)
        label = WEEKDAYS_TR[weekday] if turkish else WEEKDAYS_EN[weekday]
        subject = category or ("odak çalışması" if turkish else "focus work")
        evidence = profile.completion_by_weekday.get(weekday)
        suffix = f" Gözlenen tamamlama olasılığı {evidence:.0%}." if turkish and evidence else (
            f" Observed completion probability is {evidence:.0%}." if evidence else ""
        )
        return (
            f"{subject} için {label} gününü öneririm. Bu seçim geçmiş tamamlama ve odak kayıtlarınla haftalık yükünü birlikte değerlendiriyor.{suffix}"
            if turkish
            else f"I recommend {label} for {subject}. This combines your completion history, focus pattern, and current weekly load.{suffix}"
        )

    if any(word in normalized for word in ("bugün", "today", "şimdi", "now", "yapmal")):
        today = date.today()
        day = today if today in weekly_plan.schedules else weekly_plan.week_start
        return _day_summary(day, weekly_plan, turkish)

    if any(word in normalized for word in ("event", "etkinlik", "toplantı", "önemli")):
        important = [event for event in weekly_plan.events if event.importance >= 4]
        if not important:
            return "Bu hafta yüksek önemde event görünmüyor." if turkish else "There are no high-importance events this week."
        details = ", ".join(f"{event.title} ({event.start:%a %H:%M})" for event in important)
        return (f"Bu haftanın önemli eventleri: {details}." if turkish else f"This week's important events are: {details}.")

    unscheduled = len(weekly_plan.unscheduled_tasks)
    if turkish:
        return f"Planın hakkında bir günün nasıl göründüğünü, en yoğun veya sakin günü, önemli eventleri ya da bir kategori için en uygun günü sorabilirsin. Şu anda haftaya sığmayan {unscheduled} görev var."
    return f"Ask how a specific day looks, which day is busiest or quietest, about important events, or the best day for a task category. {unscheduled} tasks currently do not fit this week."
