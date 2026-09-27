<div align="center">

# AKIS

### A local-first planner that learns when your work actually gets done

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Planning_Workspace-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Adaptive_Model-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Local_First-003B57?logo=sqlite&logoColor=white)](https://sqlite.org/)
[![pytest](https://img.shields.io/badge/pytest-15_Tests-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-315F59.svg)](LICENSE)

Akis turns tasks, fixed commitments, and observed work habits into realistic daily and weekly plans. It protects important events, explains every placement, and keeps all personal data on the user's device.

**Designed and built by [Ali Toprak Tuğtekin](https://github.com/alitopraktug) · ATT**

</div>

---

## Why Akis

Traditional planners treat every hour as equally productive and every estimate as accurate. Akis records what happened during real focus sessions and gradually learns:

- Which hours and weekdays produce the strongest completion rate
- Which day fits each task category
- How much time different types of work actually require
- When a weekly plan is overloaded
- Where a task can move without creating a conflict

The first schedule uses transparent rules. After enough completed sessions, a Random Forest duration model becomes active. No external API, account, secret key, or cloud database is required.

## Product Areas

| View | Purpose |
|---|---|
| **Today** | Current timeline, capacity, warnings, and concise coaching notes |
| **Daily Planner** | Generate and export a schedule for any selected date |
| **Weekly Planner** | Distribute tasks once across chosen workdays and inspect daily load |
| **Important Events** | Capture high-value commitments and protect them from task placement |
| **Task Inbox** | Add structured tasks or capture a quick multi-line brain dump |
| **Focus Session** | Record actual duration, outcome, focus quality, and interruptions |
| **Insights** | Explore productive hours, weekly progress, category reliability, and the personal model |
| **Akis Coach** | Ask about specific days, workload, events, or the best day for a category |
| **Settings** | Configure work boundaries and export local data |

Primary views use a persistent sidebar instead of a crowded horizontal tab bar. The Today dashboard also provides direct shortcuts to planning, task capture, event capture, and Akis Coach.

## Planning Engine

Daily placement combines task priority, deadline urgency, completion probability, energy fit, preferred period, learned start hour, and learned weekday. Fixed tasks and events are inserted first as protected intervals; lunch and recovery blocks are then preserved wherever constraints allow.

```text
Tasks + Events + Work Boundaries
                |
                v
       Weekly Day Assignment
                |
                v
     Constraint-Aware Timelines
                |
                v
   Focus Outcomes and Real Duration
                |
                v
 Hour + Weekday Behavior Profile
                |
                v
     Better Future Recommendations
```

The weekly engine uses a target load instead of filling every available minute. A task is assigned at most once, deadlines are respected during day selection, and unscheduled work remains visible instead of silently disappearing.

## Akis Coach

Akis Coach is a deterministic, behavior-aware assistant rather than a remote language model. It reads the generated weekly plan and the local behavior profile to answer questions such as:

```text
How does Monday look?
Which day is best for Coding?
What is my busiest day?
Show important events.
Pazartesi nasıl?
Coding için hangi gün daha iyi?
```

Responses are grounded in the current schedule, event list, category preferences, and observed completion rates. The assistant supports common planning questions in English and Turkish without sending data outside the application.

## Architecture

```text
akis-adaptive-planner/
|
|-- app.py                  # Streamlit product interface
|-- models.py               # Validated domain models
|-- database.py             # SQLite persistence and exports
|-- scheduler.py            # Constraint-aware daily scheduling
|-- weekly_planner.py       # Behavior-aware weekly distribution
|-- learning_engine.py      # Personal profile and duration model
|-- assistant.py            # Local planning conversation engine
|-- analytics.py            # Productivity aggregations
|-- sample_data.py          # Deterministic demo workspace
|-- reporting.py            # CSV, TXT, and JSON exports
|
|-- data/                   # Local database directory
|-- tests/                  # Engine, persistence, and assistant tests
|-- requirements.txt
|-- README.md
`-- .gitignore
```

The scheduling, learning, assistant, analytics, and persistence layers are independent of Streamlit and can be tested or reused separately.

## Technology

- Python
- Streamlit
- SQLite
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Pytest

## Installation

```bash
git clone https://github.com/alitopraktug/akis-adaptive-planner.git
cd akis-adaptive-planner
python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate

# macOS or Linux
source venv/bin/activate
```

Install dependencies and run the application:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Akis creates a deterministic demo workspace on first launch so the planning, event, analytics, and assistant views are immediately testable.

## Testing

```bash
python -m pytest -q
```

The suite covers SQLite round trips, event persistence, data export, behavior learning, duration prediction, daily conflicts, protected events, weekly task uniqueness, weekday preferences, and Turkish assistant answers.

## Privacy and Scope

- All data is stored in a local SQLite file.
- No external calendar, account, API, or paid service is used.
- The learned profile is inspectable and exportable.
- Akis recommends plans; it does not send notifications or modify an external calendar.
- Reliable personalization improves as real focus sessions are recorded.

## License

Released under the [MIT License](LICENSE).

## Developer

**Ali Toprak Tuğtekin**

Computer Engineer

Building practical software at the intersection of machine learning, personal analytics, and thoughtful product design. Akis is designed, developed, and maintained under the **ATT** signature.

- GitHub: [alitopraktug](https://github.com/alitopraktug)
- LinkedIn: [Ali Toprak Tuğtekin](https://linkedin.com/in/ali-toprak-tuğtekin-b97b11341)
