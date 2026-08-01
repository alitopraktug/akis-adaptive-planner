"""Streamlit application for the Akis adaptive daily planner."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from html import escape
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import category_performance, daily_activity, hourly_productivity, weekly_summary
from assistant import answer_question
from database import AkisDatabase
from learning_engine import AdaptiveDurationPredictor, BehaviorProfile, learn_behavior
from models import Event, FocusSession, SessionOutcome, Task, TaskStatus, UserSettings
from reporting import generate_data_export_json, generate_schedule_csv, generate_weekly_report
from sample_data import seed_demo_data
from scheduler import generate_schedule
from weekly_planner import WeeklyPlan, generate_weekly_plan, monday_of

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = APP_DIR / "data" / "akis.db"
ACCENT = "#D96C3F"
INK = "#17251F"
MUTED = "#66736D"
PLOT_COLORS = ["#D96C3F", "#315F59", "#789181", "#D2A85A", "#7A6F9B", "#486C8A"]


def _configure_page() -> None:
    st.set_page_config(
        page_title="Akis | Adaptive Daily Planner",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --akis-ink: #17251F;
            --akis-muted: #66736D;
            --akis-accent: #D96C3F;
            --akis-paper: #F4F1EA;
            --akis-card: #FFFDFC;
            --akis-line: #DDD9D0;
            --akis-green: #315F59;
        }
        .stApp {
            background:
                radial-gradient(circle at 92% 4%, rgba(217,108,63,.07), transparent 24rem),
                var(--akis-paper);
            color: var(--akis-ink);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        [data-testid="stSidebar"] {
            background: var(--akis-ink);
            border-right: 0;
        }
        [data-testid="stSidebar"] * { color: #F8F4EB; }
        [data-testid="stSidebar"] [data-baseweb="select"] * { color: var(--akis-ink); }
        [data-testid="stSidebar"] input { color: var(--akis-ink); }
        [data-testid="stSidebar"] div[data-testid="stMetric"] {
            background: #1B3029; border-color: #30483F; box-shadow: none;
        }
        [data-testid="stSidebar"] div[data-testid="stMetric"] * { color: #F8F4EB !important; }
        [data-testid="stHeader"] { background: rgba(244,241,234,.9); backdrop-filter: blur(12px); }
        [data-testid="stToolbar"] { display: none; }
        .block-container { max-width: 1320px; padding-top: 2.5rem; padding-bottom: 4rem; }
        h1, h2, h3 { color: var(--akis-ink); letter-spacing: -.035em; }
        h1 { font-size: 2.45rem !important; font-weight: 720 !important; }
        .akis-brand { font-size: 1.55rem; font-weight: 760; letter-spacing: .12em; }
        .akis-brand-sub { color: #B9C6BE !important; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; }
        .akis-nav-label {
            color: #80958A !important; font-size: .66rem; font-weight: 750; letter-spacing: .13em;
            margin: 1.6rem 0 .5rem; text-transform: uppercase;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] { gap: .16rem; }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 9px; padding: .43rem .62rem; transition: background .16s ease;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(255,255,255,.07); }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: #284139; box-shadow: inset 3px 0 0 #E27B4E;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: .87rem; font-weight: 610; }
        [data-testid="stSidebar"] div[role="radiogroup"] [data-testid="stMarkdownContainer"] { margin-left: -.2rem; }
        .akis-side-status {
            background: #1B3029; border: 1px solid #30483F; border-radius: 11px;
            padding: .85rem .9rem; margin-top: .25rem;
        }
        .akis-side-status-top { display: flex; align-items: baseline; justify-content: space-between; }
        .akis-side-status-label { color: #9FB0A7 !important; font-size: .66rem; letter-spacing: .1em; }
        .akis-side-status-value { color: #F8F4EB !important; font-size: 1.08rem; font-weight: 760; }
        .akis-side-status-detail { color: #9FB0A7 !important; font-size: .72rem; margin-top: .3rem; }
        .akis-signature {
            display: flex; align-items: center; gap: .75rem; margin-top: 1.6rem; padding-top: 1.25rem;
            border-top: 1px solid rgba(248,244,235,.14);
        }
        .akis-monogram {
            display: grid; place-items: center; width: 42px; height: 42px; flex: 0 0 42px;
            border: 1px solid rgba(248,244,235,.32); border-radius: 50%; color: #F2A47F !important;
            font-size: .78rem; font-weight: 800; letter-spacing: .08em;
        }
        .akis-signature-name { color: #F8F4EB !important; font-size: .84rem; font-weight: 700; line-height: 1.2; }
        .akis-signature-role { color: #9FB0A7 !important; font-size: .68rem; letter-spacing: .06em; margin-top: .2rem; }
        .akis-header-top { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
        .akis-kicker { color: var(--akis-accent); font-size: .72rem; font-weight: 760; letter-spacing: .14em; text-transform: uppercase; }
        .akis-live-chip {
            display: inline-flex; align-items: center; gap: .45rem; border: 1px solid var(--akis-line);
            border-radius: 999px; background: rgba(255,253,252,.72); color: var(--akis-muted);
            padding: .35rem .65rem; font-size: .72rem; font-weight: 620;
        }
        .akis-live-dot { width: 7px; height: 7px; border-radius: 50%; background: #3F816B; box-shadow: 0 0 0 4px rgba(63,129,107,.12); }
        .akis-subtitle { color: var(--akis-muted); font-size: 1rem; max-width: 760px; margin-top: -.5rem; }
        .akis-header { border-bottom: 1px solid var(--akis-line); padding-bottom: 1.35rem; margin-bottom: 1.35rem; }
        div[data-testid="stMetric"] {
            background: var(--akis-card); border: 1px solid var(--akis-line); border-radius: 12px;
            padding: 1rem 1.1rem; box-shadow: 0 5px 22px rgba(23,37,31,.045);
        }
        div[data-testid="stMetricLabel"] { color: var(--akis-muted); }
        .akis-block {
            display: grid; grid-template-columns: 115px 1fr auto; gap: 1rem; align-items: center;
            background: var(--akis-card); border: 1px solid var(--akis-line); border-left: 4px solid var(--akis-accent);
            border-radius: 10px; padding: .9rem 1rem; margin: .55rem 0;
        }
        .akis-block.fixed { border-left-color: #315F59; }
        .akis-block.event { border-left-color: #7A6F9B; background: #F5F1F8; }
        .akis-block.break { border-left-color: #A8AEA8; background: #F0EEE8; }
        .akis-block.lunch { border-left-color: #D2A85A; background: #F8F2E5; }
        .akis-time { font-weight: 720; color: var(--akis-ink); }
        .akis-title { font-weight: 690; color: var(--akis-ink); }
        .akis-reason { color: var(--akis-muted); font-size: .84rem; margin-top: .18rem; }
        .akis-kind {
            border: 1px solid var(--akis-line); border-radius: 999px; padding: .25rem .58rem;
            font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em;
        }
        .akis-note {
            background: #FFFDFC; border: 1px solid var(--akis-line); border-radius: 10px;
            padding: 1rem 1.1rem; margin: .6rem 0;
        }
        .akis-note strong { color: var(--akis-accent); margin-right: .5rem; }
        .akis-week-card {
            background: var(--akis-card); border: 1px solid var(--akis-line); border-radius: 12px;
            padding: 1rem; min-height: 165px; margin-bottom: .75rem;
        }
        .akis-week-day { font-size: .78rem; color: var(--akis-muted); text-transform: uppercase; letter-spacing: .08em; }
        .akis-week-load { font-size: 1.35rem; font-weight: 740; color: var(--akis-ink); margin: .2rem 0 .6rem; }
        .akis-event-row {
            background: var(--akis-card); border: 1px solid var(--akis-line); border-left: 4px solid #7A6F9B;
            border-radius: 10px; padding: .9rem 1rem; margin: .55rem 0;
        }
        .akis-model-card {
            background: var(--akis-card); border: 1px solid var(--akis-line); border-radius: 12px;
            padding: 1.1rem; min-height: 128px;
        }
        .akis-model-label { color: var(--akis-muted); font-size: .76rem; letter-spacing: .08em; text-transform: uppercase; }
        .akis-model-value { color: var(--akis-ink); font-size: 1.45rem; font-weight: 740; margin-top: .35rem; }
        .akis-model-detail { color: var(--akis-muted); font-size: .84rem; margin-top: .35rem; }
        .akis-footer {
            display: flex; justify-content: space-between; align-items: center; gap: 1rem;
            border-top: 1px solid var(--akis-line); margin-top: 3.5rem; padding: 1.3rem .15rem .2rem;
            color: var(--akis-muted); font-size: .78rem;
        }
        .akis-footer-mark { color: var(--akis-ink); font-weight: 800; letter-spacing: .12em; }
        .akis-footer-maker { color: var(--akis-ink); font-weight: 650; }
        .stTabs [data-baseweb="tab-list"] { gap: .12rem; border-bottom: 1px solid var(--akis-line); }
        .stTabs [data-baseweb="tab"] { height: 3rem; padding: 0 .75rem; color: var(--akis-muted); }
        .stTabs [aria-selected="true"] { color: var(--akis-accent) !important; font-weight: 700; }
        .stButton > button, .stDownloadButton > button {
            border-radius: 9px; border-color: #C9C5BC; min-height: 2.7rem; font-weight: 620;
        }
        .akis-quick-actions { color: var(--akis-muted); font-size: .78rem; margin: .15rem 0 .45rem; }
        .stButton > button[kind="primary"] { background: var(--akis-accent); border-color: var(--akis-accent); }
        [data-testid="stDataFrame"] { border: 1px solid var(--akis-line); border-radius: 10px; overflow: hidden; }
        footer { visibility: hidden; }
        footer.akis-footer { visibility: visible; }
        @media (max-width: 700px) {
            .akis-block { grid-template-columns: 1fr; gap: .25rem; }
            .akis-kind { width: fit-content; }
            .akis-footer { align-items: flex-start; flex-direction: column; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def _database(path: str) -> AkisDatabase:
    """Return one database repository per configured path."""

    return AkisDatabase(path)


def _get_database() -> AkisDatabase:
    path = os.environ.get("AKIS_DB_PATH", str(DEFAULT_DB_PATH))
    return _database(path)


def _header(profile: BehaviorProfile, page: str) -> None:
    page_copy = {
        "Today": ("Today", "Start with what matters.", "A realistic plan for today, with protected events and recovery time."),
        "Daily Planner": ("Daily planning", "Build a day you can finish.", "Choose a date and let Akis place focused work around real commitments."),
        "Weekly Planner": ("Weekly planning", "See the week before it gets busy.", "Balance tasks across your strongest days without scheduling anything twice."),
        "Important Events": ("Calendar", "Protect the moments that cannot move.", "Keep exams, reviews, meetings, and personal commitments visible in every plan."),
        "Task Inbox": ("Task management", "Turn loose work into a clear queue.", "Capture tasks quickly, then add the detail Akis needs to schedule them well."),
        "Focus Session": ("Focus", "Record the work, not the intention.", "Log what actually happened so estimates and recommendations improve over time."),
        "Insights": ("Personal analytics", "Understand your working rhythm.", "Explore completion patterns, focus quality, weekly progress, and the personal model."),
        "Akis Coach": ("Planning assistant", "Ask your schedule a question.", "Get direct answers grounded in your tasks, events, workload, and behavior history."),
        "Settings": ("Preferences and data", "Make the system fit your boundaries.", "Control work hours, recovery rules, local storage, and data exports."),
    }
    kicker, title, subtitle = page_copy[page]
    model_label = "Learning" if profile.confidence < 0.5 else "Model active"
    st.markdown(
        f"""
        <div class="akis-header">
            <div class="akis-header-top">
                <div class="akis-kicker">{kicker}</div>
                <div class="akis-live-chip"><span class="akis-live-dot"></span>{model_label} · {profile.total_sessions} signals</div>
            </div>
            <h1>{title}</h1>
            <div class="akis-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar(profile: BehaviorProfile, settings: UserSettings) -> str:
    st.sidebar.markdown('<div class="akis-brand">AKIŞ</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="akis-brand-sub">Adaptive daily planner</div>', unsafe_allow_html=True
    )
    st.sidebar.markdown('<div class="akis-nav-label">Workspace</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Workspace",
        [
            "Today",
            "Daily Planner",
            "Weekly Planner",
            "Important Events",
            "Task Inbox",
            "Focus Session",
            "Insights",
            "Akis Coach",
            "Settings",
        ],
        label_visibility="collapsed",
        key="main_nav",
    )
    st.sidebar.markdown('<div class="akis-nav-label">System status</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div class="akis-side-status">
            <div class="akis-side-status-top">
                <span class="akis-side-status-label">PERSONAL MODEL</span>
                <span class="akis-side-status-value">{profile.confidence:.0%}</span>
            </div>
            <div class="akis-side-status-detail">{profile.total_sessions} behavior signals · local only</div>
        </div>
        <div class="akis-side-status">
            <div class="akis-side-status-top">
                <span class="akis-side-status-label">WORKING DAY</span>
                <span class="akis-side-status-value">{settings.day_start_hour:02d}:00–{settings.day_end_hour:02d}:00</span>
            </div>
            <div class="akis-side-status-detail">{settings.max_focus_minutes} min focus · {settings.break_minutes} min recovery</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div class="akis-signature">
            <div class="akis-monogram">ATT</div>
            <div>
                <div class="akis-signature-name">Ali Toprak Tuğtekin</div>
                <div class="akis-signature-role">DESIGNED & BUILT BY ATT</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


def _schedule_frame(schedule) -> pd.DataFrame:
    rows = []
    for block in schedule.blocks:
        rows.append(
            {
                "Start": block.start,
                "End": block.end,
                "Block": block.title,
                "Kind": block.kind.title(),
                "Category": block.category,
                "Reason": block.reason,
                "Duration": block.duration_minutes,
            }
        )
    return pd.DataFrame(rows)


def _render_timeline_cards(schedule) -> None:
    if not schedule.blocks:
        st.info("No blocks are scheduled for this day. Add tasks in Task Inbox.")
        return
    for block in schedule.blocks:
        reason = escape(block.reason)
        category = escape(block.category)
        st.markdown(
            f"""
            <div class="akis-block {block.kind}">
                <div class="akis-time">{block.start.strftime('%H:%M')}–{block.end.strftime('%H:%M')}</div>
                <div>
                    <div class="akis-title">{escape(block.title)}</div>
                    <div class="akis-reason">{category} · {reason}</div>
                </div>
                <div class="akis-kind">{escape(block.kind)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _coach_notes(schedule, profile: BehaviorProfile) -> list[str]:
    notes: list[str] = []
    if schedule.utilization >= 0.88:
        notes.append("Your day is tightly packed. Keep the protected breaks instead of adding another task.")
    elif schedule.utilization <= 0.45:
        notes.append("You still have meaningful capacity. Consider one medium-priority task or a longer recovery window.")
    if schedule.unscheduled_tasks:
        notes.append(
            f"{len(schedule.unscheduled_tasks)} task(s) did not fit. Move the lowest-priority item before extending your day."
        )
    for category, factor in profile.duration_factors.items():
        if factor >= 1.15:
            notes.append(f"Add buffer to {category} work; it typically takes {factor:.2f}x your estimate.")
            break
    if profile.insights:
        notes.append(profile.insights[0])
    return notes[:3]


def _navigate(page: str) -> None:
    """Switch the primary workspace view from a quick-action button."""

    st.session_state["main_nav"] = page


def _render_today(schedule, profile: BehaviorProfile, summary: dict[str, object]) -> None:
    st.markdown('<div class="akis-quick-actions">QUICK ACTIONS</div>', unsafe_allow_html=True)
    actions = st.columns(4)
    actions[0].button("Plan another day", width="stretch", on_click=_navigate, args=("Daily Planner",))
    actions[1].button("Add a task", width="stretch", on_click=_navigate, args=("Task Inbox",))
    actions[2].button("Add an event", width="stretch", on_click=_navigate, args=("Important Events",))
    actions[3].button("Ask Akis Coach", width="stretch", on_click=_navigate, args=("Akis Coach",))
    st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
    metrics = st.columns(4)
    metrics[0].metric("Planned focus", f"{schedule.task_minutes / 60:.1f} h")
    metrics[1].metric("Day utilization", f"{schedule.utilization:.0%}")
    metrics[2].metric("Weekly completion", f"{float(summary['completion_rate']):.0%}")
    metrics[3].metric("Model confidence", f"{profile.confidence:.0%}")
    left, right = st.columns([1.75, 1])
    with left:
        st.markdown("#### Today’s timeline")
        _render_timeline_cards(schedule)
    with right:
        st.markdown("#### Coach notes")
        for index, note in enumerate(_coach_notes(schedule, profile), start=1):
            st.markdown(
                f'<div class="akis-note"><strong>{index:02d}</strong>{escape(note)}</div>',
                unsafe_allow_html=True,
            )
        if schedule.warnings:
            st.markdown("#### Planning warnings")
            for warning in schedule.warnings:
                st.warning(warning)


def _render_task_inbox(database: AkisDatabase, pending_tasks: list[Task]) -> None:
    left, right = st.columns([1, 1.35])
    with left:
        st.markdown("#### Add a task")
        with st.form("task_form", clear_on_submit=True):
            title = st.text_input("Task title", placeholder="Prepare project presentation")
            category = st.selectbox(
                "Category", ["Coding", "Study", "Portfolio", "Admin", "Planning", "Health", "Personal", "Meeting"]
            )
            first, second = st.columns(2)
            priority = first.select_slider("Priority", options=[1, 2, 3, 4, 5], value=3)
            duration = second.number_input("Estimated minutes", min_value=5, max_value=720, value=45, step=5)
            energy = st.select_slider("Energy required", options=[1, 2, 3, 4, 5], value=3)
            preferred = st.selectbox("Preferred period", ["any", "morning", "afternoon", "evening"])
            deadline_enabled = st.checkbox("Set a deadline", value=True)
            deadline_date = st.date_input("Deadline date", value=date.today())
            deadline_time = st.time_input("Deadline time", value=time(18, 0))
            fixed_enabled = st.checkbox("Fixed start time")
            fixed_date = st.date_input("Fixed date", value=date.today(), disabled=not fixed_enabled)
            fixed_time = st.time_input("Fixed time", value=time(14, 0), disabled=not fixed_enabled)
            splittable = st.checkbox("Can be split into focus blocks", value=True)
            submitted = st.form_submit_button("Add task", type="primary", width="stretch")
        if submitted:
            try:
                database.add_task(
                    Task(
                        title=title,
                        category=category,
                        priority=int(priority),
                        estimated_minutes=int(duration),
                        energy_required=int(energy),
                        deadline=datetime.combine(deadline_date, deadline_time) if deadline_enabled else None,
                        preferred_period=preferred,  # type: ignore[arg-type]
                        splittable=splittable,
                        fixed_start=datetime.combine(fixed_date, fixed_time) if fixed_enabled else None,
                    )
                )
                st.success("Task added to your planning queue.")
                st.rerun()
            except (TypeError, ValueError) as exc:
                st.error(str(exc))

        st.markdown("#### Quick capture")
        with st.form("brain_dump", clear_on_submit=True):
            brain_dump = st.text_area(
                "One task per line",
                placeholder="Review pull request\nPractice algorithms\nPlan tomorrow",
                height=120,
            )
            capture = st.form_submit_button("Capture tasks", width="stretch")
        if capture:
            lines = [line.strip() for line in brain_dump.splitlines() if line.strip()]
            if not lines:
                st.error("Enter at least one task.")
            else:
                for line in lines[:25]:
                    database.add_task(Task(line, "Inbox", 3, 45, 3))
                st.success(f"Captured {len(lines[:25])} task(s).")
                st.rerun()

    with right:
        st.markdown("#### Planning queue")
        if not pending_tasks:
            st.info("Your planning queue is empty.")
            return
        frame = pd.DataFrame(
            [
                {
                    "Task": task.title,
                    "Category": task.category,
                    "Priority": task.priority,
                    "Estimate": task.estimated_minutes,
                    "Energy": task.energy_required,
                    "Deadline": task.deadline,
                    "Preferred": task.preferred_period.title(),
                    "Fixed": task.fixed_start,
                }
                for task in pending_tasks
            ]
        )
        st.dataframe(
            frame,
            hide_index=True,
            width="stretch",
            column_config={
                "Estimate": st.column_config.NumberColumn("Estimate", format="%d min"),
                "Deadline": st.column_config.DatetimeColumn(format="MMM D, HH:mm"),
                "Fixed": st.column_config.DatetimeColumn(format="MMM D, HH:mm"),
            },
        )


def _render_smart_schedule(schedule) -> None:
    top = st.columns(3)
    top[0].metric("Scheduled tasks", sum(block.kind in {"task", "fixed"} for block in schedule.blocks))
    top[1].metric("Protected breaks", sum(block.kind in {"break", "lunch"} for block in schedule.blocks))
    top[2].metric("Unscheduled", len(schedule.unscheduled_tasks))
    frame = _schedule_frame(schedule)
    if frame.empty:
        st.info("No schedule could be generated.")
        return
    frame["Timeline"] = "Today"
    figure = px.timeline(
        frame,
        x_start="Start",
        x_end="End",
        y="Timeline",
        color="Kind",
        hover_name="Block",
        hover_data={"Category": True, "Reason": True, "Duration": True, "Timeline": False},
        color_discrete_map={
            "Task": ACCENT,
            "Fixed": "#315F59",
            "Event": "#7A6F9B",
            "Break": "#A8AEA8",
            "Lunch": "#D2A85A",
        },
    )
    figure.update_yaxes(visible=False)
    figure.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=25, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="",
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.markdown("#### Why each task was placed")
    explanations = frame[frame["Kind"].isin(["Task", "Fixed"])][
        ["Block", "Start", "End", "Category", "Reason", "Duration"]
    ]
    st.dataframe(
        explanations,
        hide_index=True,
        width="stretch",
        column_config={
            "Start": st.column_config.DatetimeColumn(format="HH:mm"),
            "End": st.column_config.DatetimeColumn(format="HH:mm"),
            "Duration": st.column_config.NumberColumn(format="%d min"),
        },
    )
    if schedule.unscheduled_tasks:
        st.markdown("#### Could not fit today")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Task": task.title, "Priority": task.priority, "Minutes": task.estimated_minutes, "Deadline": task.deadline}
                    for task in schedule.unscheduled_tasks
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def _render_daily_planner(
    tasks: list[Task],
    events: list[Event],
    settings: UserSettings,
    profile: BehaviorProfile,
    predictor: AdaptiveDurationPredictor,
) -> None:
    """Render a date-driven planning workspace separate from the home view."""

    selected_day = st.date_input("Plan date", value=date.today(), key="daily_plan_date")
    schedule = generate_schedule(
        tasks,
        target_date=selected_day,
        settings=settings,
        profile=profile,
        predictor=predictor,
        events=events,
    )
    st.caption("The plan protects calendar events, deadlines, lunch, and recovery before placing flexible work.")
    _render_smart_schedule(schedule)
    st.markdown("#### Day timeline")
    _render_timeline_cards(schedule)
    st.download_button(
        "Download selected day",
        data=generate_schedule_csv(schedule).encode("utf-8"),
        file_name=f"akis_{selected_day.isoformat()}_schedule.csv",
        mime="text/csv",
    )


def _weekly_plan_frame(plan: WeeklyPlan) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day, schedule in plan.schedules.items():
        rows.append(
            {
                "Date": day,
                "Day": day.strftime("%A"),
                "Focus hours": round(schedule.task_minutes / 60, 2),
                "Utilization": schedule.utilization,
                "Tasks": sum(block.kind in {"task", "fixed"} for block in schedule.blocks),
                "Events": sum(block.kind == "event" for block in schedule.blocks),
            }
        )
    return pd.DataFrame(rows)


def _render_weekly_planner(
    tasks: list[Task],
    events: list[Event],
    settings: UserSettings,
    profile: BehaviorProfile,
    predictor: AdaptiveDurationPredictor,
) -> WeeklyPlan:
    """Render a configurable seven-day workload plan."""

    controls = st.columns([1, 1.5])
    anchor = controls[0].date_input("Week containing", value=date.today(), key="week_anchor")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    selected_names = controls[1].multiselect(
        "Planning days",
        day_names,
        default=day_names[:6],
        key="weekly_active_days",
    )
    active = [day_names.index(name) for name in selected_names] or [0]
    plan = generate_weekly_plan(
        tasks,
        events=events,
        week_start=anchor,
        active_weekdays=active,
        settings=settings,
        profile=profile,
        predictor=predictor,
    )
    frame = _weekly_plan_frame(plan)
    metrics = st.columns(4)
    metrics[0].metric("Planned focus", f"{plan.task_minutes / 60:.1f} h")
    metrics[1].metric("Weekly events", len(plan.events))
    metrics[2].metric("Average load", f"{plan.average_utilization:.0%}")
    metrics[3].metric("Unscheduled", len(plan.unscheduled_tasks))

    figure = px.bar(
        frame,
        x="Day",
        y="Focus hours",
        color="Utilization",
        text="Focus hours",
        color_continuous_scale=["#DCE5DF", "#D2A85A", ACCENT],
        hover_data=["Tasks", "Events", "Date"],
    )
    figure.update_layout(
        height=310,
        margin=dict(l=15, r=15, t=20, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis_title="",
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    st.markdown("#### Week at a glance")
    columns = st.columns(3)
    for index, (day, schedule) in enumerate(plan.schedules.items()):
        task_blocks = [block for block in schedule.blocks if block.kind in {"task", "fixed"}]
        event_blocks = [block for block in schedule.blocks if block.kind == "event"]
        details = [f"{block.start:%H:%M} · {escape(block.title)}" for block in event_blocks[:2]]
        details.extend(escape(block.title) for block in task_blocks[:2])
        preview = "<br>".join(details) if details else "Open capacity"
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="akis-week-card">
                    <div class="akis-week-day">{day:%A · %b %d}</div>
                    <div class="akis-week-load">{schedule.task_minutes / 60:.1f} h focus</div>
                    <div class="akis-reason">{preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"Open {day:%A} timeline"):
                _render_timeline_cards(schedule)
    if plan.unscheduled_tasks:
        st.warning(f"{len(plan.unscheduled_tasks)} task(s) need a later week or more capacity.")
    return plan


def _render_important_events(database: AkisDatabase, events: list[Event]) -> None:
    """Render event capture and the important weekly event list."""

    left, right = st.columns([1, 1.45])
    with left:
        st.markdown("#### Add an event")
        with st.form("event_form", clear_on_submit=True):
            title = st.text_input("Event title", placeholder="Project review")
            event_date = st.date_input("Date", value=date.today(), key="event_date")
            times = st.columns(2)
            start_time = times[0].time_input("Starts", value=time(14, 0))
            end_time = times[1].time_input("Ends", value=time(15, 0))
            importance = st.select_slider("Importance", options=[1, 2, 3, 4, 5], value=4)
            category = st.selectbox("Event category", ["Meeting", "Study", "Career", "Health", "Personal"])
            location = st.text_input("Location", placeholder="Optional")
            notes = st.text_area("Notes", placeholder="Preparation or context")
            submitted = st.form_submit_button("Add event", type="primary", width="stretch")
        if submitted:
            try:
                database.add_event(
                    Event(
                        title=title,
                        start=datetime.combine(event_date, start_time),
                        end=datetime.combine(event_date, end_time),
                        importance=int(importance),
                        category=category,
                        location=location,
                        notes=notes,
                    )
                )
                st.success("Event added and protected in future plans.")
                st.rerun()
            except (TypeError, ValueError) as exc:
                st.error(str(exc))
    with right:
        st.markdown("#### Important this week")
        week_start = monday_of(date.today())
        week_end = week_start + timedelta(days=7)
        weekly_events = [event for event in events if week_start <= event.start.date() < week_end]
        important = sorted(weekly_events, key=lambda event: (-event.importance, event.start))
        if not important:
            st.info("No events are registered for this week.")
        for event in important:
            location = f" · {escape(event.location)}" if event.location else ""
            st.markdown(
                f"""
                <div class="akis-event-row">
                    <div class="akis-week-day">{event.start:%A · %b %d · %H:%M–}{event.end:%H:%M}</div>
                    <div class="akis-title">{escape(event.title)}</div>
                    <div class="akis-reason">Importance {event.importance}/5 · {escape(event.category)}{location}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_coach(
    plan: WeeklyPlan,
    profile: BehaviorProfile,
    tasks: list[Task],
    events: list[Event],
) -> None:
    """Render the local planning conversation interface."""

    st.caption("A local, behavior-aware planning assistant. It uses your Akis data and does not call an external AI service.")
    suggestions = st.columns(4)
    prompts = ["How does today look?", "Which day is best for Coding?", "What is my busiest day?", "Show important events"]
    suggested_prompt: str | None = None
    for column, prompt in zip(suggestions, prompts):
        if column.button(prompt, width="stretch"):
            suggested_prompt = prompt
    history = st.session_state.setdefault(
        "coach_history",
        [{"role": "assistant", "content": "Ask me about a day, weekly workload, important events, or the best time for a type of work."}],
    )
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input("Ask Akis Coach about your week")
    resolved = question or suggested_prompt
    if resolved:
        response = answer_question(
            resolved,
            weekly_plan=plan,
            profile=profile,
            tasks=tasks,
            events=events,
        )
        history.extend(
            [
                {"role": "user", "content": resolved},
                {"role": "assistant", "content": response},
            ]
        )
        st.rerun()


def _render_focus_session(database: AkisDatabase, pending_tasks: list[Task], predictor, profile) -> None:
    if not pending_tasks:
        st.info("Add a task before logging a focus session.")
        return
    task_map = {task.task_id: task for task in pending_tasks}
    selected_id = st.selectbox(
        "Task",
        list(task_map),
        format_func=lambda task_id: f"{task_map[task_id].title} · {task_map[task_id].category}",
    )
    task = task_map[selected_id]
    predicted = predictor.predict(task, datetime.now().hour, profile)
    cards = st.columns(3)
    cards[0].metric("Original estimate", f"{task.estimated_minutes} min")
    cards[1].metric("Adaptive estimate", f"{predicted} min", delta=f"{predicted - task.estimated_minutes:+d} min")
    cards[2].metric("Energy required", f"{task.energy_required}/5")
    st.caption("Log what actually happened. Every session improves future duration and time-slot recommendations.")
    with st.form("focus_log"):
        first, second = st.columns(2)
        session_date = first.date_input("Session date", value=date.today())
        session_time = second.time_input("Start time", value=datetime.now().time().replace(second=0, microsecond=0))
        actual = st.number_input("Actual minutes", min_value=0, max_value=1_440, value=predicted, step=5)
        outcome = st.segmented_control(
            "Outcome", options=["completed", "partial", "skipped"], default="completed"
        )
        focus_rating = st.select_slider("Focus quality", options=[1, 2, 3, 4, 5], value=4)
        interruptions = st.number_input("Interruptions", min_value=0, max_value=50, value=0)
        save = st.form_submit_button("Save focus session", type="primary", width="stretch")
    if save:
        try:
            resolved_outcome = SessionOutcome(outcome or "completed")
            database.record_session(
                FocusSession(
                    task_id=task.task_id,
                    task_title=task.title,
                    category=task.category,
                    planned_minutes=predicted,
                    actual_minutes=int(actual),
                    start_time=datetime.combine(session_date, session_time),
                    outcome=resolved_outcome,
                    focus_rating=int(focus_rating),
                    interruption_count=int(interruptions),
                    priority=task.priority,
                    energy_required=task.energy_required,
                )
            )
            if resolved_outcome == SessionOutcome.COMPLETED:
                database.update_task_status(task.task_id, TaskStatus.COMPLETED)
            elif resolved_outcome == SessionOutcome.SKIPPED:
                database.update_task_status(task.task_id, TaskStatus.SKIPPED)
            st.success("Session saved. Your personal model has new evidence.")
            st.rerun()
        except (TypeError, ValueError) as exc:
            st.error(str(exc))


def _render_behavior_insights(sessions: pd.DataFrame) -> None:
    hourly = hourly_productivity(sessions)
    categories = category_performance(sessions)
    if hourly.empty:
        st.info("Complete focus sessions to unlock behavior analytics.")
        return
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Focus rhythm by hour")
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=hourly["hour"],
                y=hourly["completion_rate"] * 100,
                name="Completion rate",
                marker_color=ACCENT,
                opacity=0.78,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=hourly["hour"],
                y=hourly["focus_rating"] * 20,
                name="Focus quality",
                mode="lines+markers",
                line=dict(color="#315F59", width=3),
            )
        )
        figure.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Score (%)",
            xaxis_title="Start hour",
            legend_title_text="",
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("#### Category reliability")
        category_figure = px.bar(
            categories.sort_values("completion_rate"),
            x="completion_rate",
            y="category",
            orientation="h",
            color="duration_ratio",
            color_continuous_scale=["#789181", "#D2A85A", "#D96C3F"],
            labels={"completion_rate": "Completion rate", "category": "", "duration_ratio": "Duration ratio"},
        )
        category_figure.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        category_figure.update_xaxes(tickformat=".0%")
        st.plotly_chart(category_figure, width="stretch", config={"displayModeBar": False})
    st.dataframe(
        categories,
        hide_index=True,
        width="stretch",
        column_config={
            "completion_rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "duration_ratio": st.column_config.NumberColumn(format="%.2fx"),
        },
    )


def _render_weekly_review(sessions: pd.DataFrame, summary: dict[str, object]) -> None:
    cards = st.columns(5)
    cards[0].metric("Sessions", int(summary["sessions"]))
    cards[1].metric("Completed", int(summary["completed"]))
    cards[2].metric("Focus time", f"{int(summary['focus_minutes']) / 60:.1f} h")
    cards[3].metric("Average focus", f"{float(summary['average_focus']):.1f}/5")
    cards[4].metric("Estimate accuracy", f"{float(summary['estimate_accuracy']):.0%}")
    activity = daily_activity(sessions)
    if activity.empty:
        st.info("No weekly activity is available yet.")
        return
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("#### Planned vs completed")
        figure = go.Figure()
        figure.add_trace(go.Bar(x=activity["date"], y=activity["planned"], name="Planned", marker_color="#C9C5BC"))
        figure.add_trace(go.Bar(x=activity["date"], y=activity["completed"], name="Completed", marker_color=ACCENT))
        figure.update_layout(
            barmode="group",
            height=350,
            margin=dict(l=20, r=20, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
            yaxis_title="Sessions",
            xaxis_title="",
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("#### Focus time")
        focus_figure = px.area(
            activity,
            x="date",
            y="focus_minutes",
            markers=True,
            color_discrete_sequence=["#315F59"],
            labels={"focus_minutes": "Minutes", "date": ""},
        )
        focus_figure.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(focus_figure, width="stretch", config={"displayModeBar": False})


def _model_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="akis-model-card">
            <div class="akis-model-label">{escape(label)}</div>
            <div class="akis-model-value">{escape(value)}</div>
            <div class="akis-model-detail">{escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_personal_model(profile: BehaviorProfile, predictor: AdaptiveDurationPredictor) -> None:
    st.caption("Every value below comes from your recorded sessions. No external profile or hidden API is used.")
    best_hour = max(profile.focus_by_hour, key=profile.focus_by_hour.get) if profile.focus_by_hour else None
    columns = st.columns(4)
    with columns[0]:
        _model_card("Model state", "Adaptive" if predictor.is_trained else "Calibrating", f"{predictor.training_samples} completed samples")
    with columns[1]:
        _model_card("Confidence", f"{profile.confidence:.0%}", "Reaches full confidence after 50 observations")
    with columns[2]:
        _model_card("Strongest hour", f"{best_hour:02d}:00" if best_hour is not None else "Unknown", "Based on focus quality")
    with columns[3]:
        _model_card("Known categories", str(len(profile.duration_factors)), "Categories with duration evidence")
    st.markdown("#### What Akis has learned")
    for index, insight in enumerate(profile.insights, start=1):
        st.markdown(
            f'<div class="akis-note"><strong>{index:02d}</strong>{escape(insight)}</div>',
            unsafe_allow_html=True,
        )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Duration calibration")
        if profile.duration_factors:
            factors = pd.DataFrame(
                [{"Category": category, "Actual / estimated": factor} for category, factor in profile.duration_factors.items()]
            )
            fig = px.bar(
                factors.sort_values("Actual / estimated"),
                x="Actual / estimated",
                y="Category",
                orientation="h",
                color="Actual / estimated",
                color_continuous_scale=["#789181", "#D2A85A", "#D96C3F"],
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color=INK)
            fig.update_layout(height=330, margin=dict(l=20, r=20, t=20, b=30), coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("#### Preferred start times")
        if profile.preferred_hours:
            preferences = pd.DataFrame(
                [{"Category": category, "Preferred hour": hour} for category, hour in profile.preferred_hours.items()]
            ).sort_values("Preferred hour")
            fig = px.scatter(
                preferences,
                x="Preferred hour",
                y="Category",
                size=[18] * len(preferences),
                color="Category",
                color_discrete_sequence=PLOT_COLORS,
            )
            fig.update_layout(height=330, margin=dict(l=20, r=20, t=20, b=30), showlegend=False)
            fig.update_xaxes(dtick=1, range=[6, 21])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _render_settings(database: AkisDatabase, settings: UserSettings, sessions, profile, schedule) -> None:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Scheduling preferences")
        with st.form("settings_form"):
            day_start = st.slider("Day starts", 5, 14, settings.day_start_hour)
            day_end = st.slider("Day ends", 15, 23, settings.day_end_hour)
            break_minutes = st.slider("Recovery break", 0, 30, settings.break_minutes, step=5)
            max_focus = st.slider("Maximum focus block", 30, 180, settings.max_focus_minutes, step=15)
            lunch_hour = st.slider("Lunch starts", 10, 16, settings.lunch_start_hour)
            lunch_minutes = st.slider("Lunch duration", 0, 120, settings.lunch_minutes, step=15)
            save_settings = st.form_submit_button("Save preferences", type="primary", width="stretch")
        if save_settings:
            try:
                database.save_settings(
                    UserSettings(
                        day_start_hour=day_start,
                        day_end_hour=day_end,
                        break_minutes=break_minutes,
                        max_focus_minutes=max_focus,
                        lunch_start_hour=lunch_hour,
                        lunch_minutes=lunch_minutes,
                    )
                )
                st.success("Scheduling preferences saved.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with right:
        st.markdown("#### Your data")
        st.write("Akis stores tasks, settings, and behavior history in a local SQLite file. Export your complete profile at any time.")
        snapshot = database.export_snapshot()
        st.download_button(
            "Download complete data export",
            data=generate_data_export_json(snapshot).encode("utf-8"),
            file_name="akis_data_export.json",
            mime="application/json",
            width="stretch",
        )
        st.download_button(
            "Download weekly review",
            data=generate_weekly_report(sessions, profile, schedule).encode("utf-8"),
            file_name="akis_weekly_review.txt",
            mime="text/plain",
            width="stretch",
        )
        st.download_button(
            "Download today's schedule",
            data=generate_schedule_csv(schedule).encode("utf-8"),
            file_name="akis_schedule.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption(f"Local database: {database.path.name}")


def main() -> None:
    """Run the Akis Streamlit application."""

    _configure_page()
    database = _get_database()
    seed_demo_data(database)
    settings = database.load_settings()
    sessions = database.list_sessions()
    profile = learn_behavior(sessions)
    predictor = AdaptiveDurationPredictor().fit(sessions)
    pending_tasks = database.list_tasks(TaskStatus.PENDING)
    events = database.list_events()
    schedule = generate_schedule(
        pending_tasks,
        target_date=date.today(),
        settings=settings,
        profile=profile,
        predictor=predictor,
        events=events,
    )
    summary = weekly_summary(sessions)

    page = _sidebar(profile, settings)
    _header(profile, page)
    if page == "Today":
        _render_today(schedule, profile, summary)
    elif page == "Daily Planner":
        _render_daily_planner(pending_tasks, events, settings, profile, predictor)
    elif page == "Weekly Planner":
        _render_weekly_planner(pending_tasks, events, settings, profile, predictor)
    elif page == "Important Events":
        _render_important_events(database, events)
    elif page == "Task Inbox":
        _render_task_inbox(database, pending_tasks)
    elif page == "Focus Session":
        _render_focus_session(database, pending_tasks, predictor, profile)
    elif page == "Insights":
        insight_tabs = st.tabs(["Behavior", "Weekly Review", "Personal Model"])
        with insight_tabs[0]:
            _render_behavior_insights(sessions)
        with insight_tabs[1]:
            _render_weekly_review(sessions, summary)
        with insight_tabs[2]:
            _render_personal_model(profile, predictor)
    elif page == "Akis Coach":
        current_week_plan = generate_weekly_plan(
            pending_tasks,
            events=events,
            week_start=date.today(),
            active_weekdays=range(6),
            settings=settings,
            profile=profile,
            predictor=predictor,
        )
        _render_coach(current_week_plan, profile, pending_tasks, events)
    elif page == "Settings":
        _render_settings(database, settings, sessions, profile, schedule)
    st.markdown(
        """
        <footer class="akis-footer">
            <span><span class="akis-footer-mark">AKIS</span> · Adaptive planning, grounded in real behavior.</span>
            <span>Made by <span class="akis-footer-maker">Ali Toprak Tuğtekin</span> · ATT</span>
        </footer>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
