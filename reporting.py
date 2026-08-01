"""Human-readable exports for Akis productivity data."""

from __future__ import annotations

from datetime import UTC, datetime
import json

import pandas as pd

from analytics import category_performance, weekly_summary
from learning_engine import BehaviorProfile
from models import ScheduleResult


def generate_schedule_csv(schedule: ScheduleResult) -> str:
    """Return schedule blocks as CSV text."""

    return pd.DataFrame([block.to_dict() for block in schedule.blocks]).to_csv(index=False)


def generate_weekly_report(
    sessions: pd.DataFrame,
    profile: BehaviorProfile,
    schedule: ScheduleResult | None = None,
) -> str:
    """Return a concise productivity and behavior report."""

    summary = weekly_summary(sessions)
    lines = [
        "AKIS WEEKLY REVIEW",
        "=" * 28,
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "WEEKLY PERFORMANCE",
        f"Sessions: {summary['sessions']}",
        f"Completion rate: {summary['completion_rate']:.0%}",
        f"Focused time: {summary['focus_minutes'] / 60:.1f} hours",
        f"Average focus: {summary['average_focus']:.1f}/5",
        f"Estimate accuracy: {summary['estimate_accuracy']:.0%}",
        "",
        "PERSONAL MODEL",
        f"Behavior samples: {profile.total_sessions}",
        f"Model confidence: {profile.confidence:.0%}",
    ]
    lines.extend(f"- {insight}" for insight in profile.insights)
    if schedule is not None:
        lines.extend(
            [
                "",
                "CURRENT PLAN",
                f"Task time: {schedule.task_minutes} minutes",
                f"Day utilization: {schedule.utilization:.0%}",
                f"Unscheduled tasks: {len(schedule.unscheduled_tasks)}",
            ]
        )
    categories = category_performance(sessions)
    if not categories.empty:
        lines.extend(["", "CATEGORY PERFORMANCE"])
        for row in categories.itertuples(index=False):
            lines.append(
                f"- {row.category}: {row.completion_rate:.0%} completion, "
                f"{row.duration_ratio:.2f}x duration ratio"
            )
    return "\n".join(lines) + "\n"


def generate_data_export_json(snapshot: dict[str, object]) -> str:
    """Serialize a database snapshot with stable formatting."""

    return json.dumps(snapshot, indent=2, ensure_ascii=False, default=str)
