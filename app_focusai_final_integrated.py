import streamlit as st
import pandas as pd
import joblib
import os
import time
import calendar
import json
from datetime import date, datetime, timedelta

# ============================================================
# FOCUSAI — FINAL STUDENT VERSION
# ============================================================

MODEL_FILE = "focusai_best_model.pkl"
FALLBACK_MODEL_FILE = "focusai_tuned_model.pkl"
REWARDS_FILE = "focusai_daily_rewards.json"

st.set_page_config(
    page_title="FocusAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# DAILY REWARD STORAGE (persists across refreshes/restarts)
# ============================================================

def load_daily_rewards():
    try:
        if os.path.exists(REWARDS_FILE):
            with open(REWARDS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def save_daily_rewards(data):
    temp_file = REWARDS_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_file, REWARDS_FILE)


def award_points(points, reason, day=None):
    """Add points to the selected calendar day and persist them locally."""
    points = int(points)
    if points <= 0:
        return
    reward_day = (day or date.today()).isoformat()
    ledger = st.session_state.daily_rewards
    entry = ledger.setdefault(reward_day, {"total": 0, "events": []})
    entry["total"] = int(entry.get("total", 0)) + points
    entry.setdefault("events", []).append({
        "time": datetime.now().strftime("%H:%M"),
        "points": points,
        "reason": reason,
    })
    save_daily_rewards(ledger)
    st.session_state.focus_points = int(ledger.get(date.today().isoformat(), {}).get("total", 0))


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "tasks": [],
    "study_log": {},
    "prediction": None,
    "prediction_confidence": None,
    "prediction_reasons": [],
    "challenge": None,
    "active_session": False,
    "session_task": None,
    "session_duration": 25,
    "session_end": None,
    "session_started_at": None,
    "session_paused_remaining": None,
    "session_total_seconds": 25 * 60,
    "completed_sessions": 0,
    "session_message": None,
    "focus_points": 0,
    "daily_rewards": load_daily_rewards(),
    "challenge_completed_date": None,
    "calendar_month": date.today().replace(day=1),
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Points shown in the header are always for today; older days remain in the ledger.
st.session_state.focus_points = int(st.session_state.daily_rewards.get(date.today().isoformat(), {}).get("total", 0))

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    # Prefer the newly trained balanced model; fall back to the older filename.
    for candidate in (MODEL_FILE, FALLBACK_MODEL_FILE):
        if os.path.exists(candidate):
            return joblib.load(candidate), candidate
    return None, None

model, loaded_model_file = load_model()

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
    .rewards-banner {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 18px 22px;
        border: 1px solid rgba(99,102,241,.35);
        border-radius: 18px;
        background: linear-gradient(115deg, rgba(99,102,241,.16), rgba(16,185,129,.12), rgba(245,158,11,.10));
        box-shadow: 0 5px 18px rgba(50,60,100,.08);
        margin: 10px 0 20px 0;
    }
    .rewards-trophy { font-size: 42px; line-height: 1; }
    .rewards-copy { min-width: 0; }
    .rewards-kicker { font-size: 12px; opacity: .75; letter-spacing: .12em; font-weight: 700; }
    .rewards-total { font-size: 27px; font-weight: 800; line-height: 1.35; color: var(--text-color); }
    .rewards-details { font-size: 13px; opacity: .8; margin-top: 3px; }
    .rewards-details span { padding: 0 4px; }

    /* Larger, colorful Streamlit navigation tabs */
    div[data-testid="stTabs"] > div[role="tablist"] {
        gap: 8px;
        border-bottom: 2px solid rgba(99,102,241,.22);
        padding: 8px 6px 0 6px;
        background: linear-gradient(90deg, rgba(99,102,241,.07), rgba(236,72,153,.06), rgba(16,185,129,.06));
        border-radius: 14px 14px 0 0;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        font-size: 16px !important;
        font-weight: 650 !important;
        padding: 13px 16px !important;
        min-height: 52px;
        border-radius: 11px 11px 0 0;
        transition: background .15s ease, transform .15s ease;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        background: rgba(99,102,241,.12);
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: linear-gradient(120deg, rgba(99,102,241,.18), rgba(16,185,129,.13));
        border-bottom: 3px solid #6366f1;
    }
    @media (max-width: 700px) {
        div[data-testid="stTabs"] button[role="tab"] { font-size: 14px !important; padding: 10px 9px !important; }
        .rewards-total { font-size: 23px; }
        .rewards-details { font-size: 11px; }
    }

    .main-title {
        font-size: 42px;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.70;
        margin-bottom: 22px;
    }

    .result-box {
        padding: 22px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        margin: 12px 0 20px 0;
    }

    .challenge-box {
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.28);
        margin: 10px 0 18px 0;
    }

    .active-box {
        padding: 16px 20px;
        border-radius: 15px;
        border: 2px solid rgba(80,140,220,0.55);
        margin: 8px 0 18px 0;
    }

    .small-muted {
        opacity: 0.65;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def today_key():
    return date.today().isoformat()


def add_study_minutes(minutes, day=None):
    day = day or date.today()
    key = day.isoformat()
    st.session_state.study_log[key] = (
        st.session_state.study_log.get(key, 0) + int(minutes)
    )


def completed_task_count():
    return sum(1 for t in st.session_state.tasks if t["completed"])


def total_task_count():
    return len(st.session_state.tasks)


def task_completion_rate():
    total = total_task_count()
    if total == 0:
        return 0
    return round(completed_task_count() / total * 100)


def weekly_minutes():
    total = 0
    start = date.today() - timedelta(days=6)

    for i in range(7):
        d = start + timedelta(days=i)
        total += st.session_state.study_log.get(d.isoformat(), 0)

    return total


def current_streak():
    streak = 0
    d = date.today()

    while st.session_state.study_log.get(d.isoformat(), 0) > 0:
        streak += 1
        d -= timedelta(days=1)

    return streak


def create_challenge(screen_time, study_hours, session_length,
                     completion_pct, deadline_urgency,
                     multitasking_level, task_difficulty, prediction):

    if screen_time >= 7:
        return {
            "title": "Phone-Free Focus",
            "text": "Complete your next focused study session without checking your phone.",
            "minutes": min(max(session_length, 25), 45),
            "reason": "Your reported screen time is relatively high."
        }

    if completion_pct < 70:
        return {
            "title": "Finish One Thing",
            "text": "Choose one small pending task and finish it before starting another.",
            "minutes": 25,
            "reason": "Your current task-completion pattern leaves some planned work unfinished."
        }

    if multitasking_level == "High":
        return {
            "title": "One Task Only",
            "text": "Work on one task for one uninterrupted focus session.",
            "minutes": 25,
            "reason": "You reported frequent multitasking."
        }

    if study_hours <= 3:
        return {
            "title": "Add One Focus Block",
            "text": "Complete one extra focused study block today.",
            "minutes": 30,
            "reason": "Your reported daily study time is relatively low."
        }

    if deadline_urgency == "High":
        return {
            "title": "Beat the Deadline",
            "text": "Start your most urgent task before doing anything else.",
            "minutes": 25,
            "reason": "You reported high deadline urgency."
        }

    if task_difficulty == "Hard":
        return {
            "title": "Break the Hard Task",
            "text": "Work on the hardest task for one focused session, starting with its smallest step.",
            "minutes": 30,
            "reason": "Difficult tasks can become easier to start when broken into smaller steps."
        }

    if prediction == "High":
        return {
            "title": "Start Before You Feel Ready",
            "text": "Do one short distraction-free session instead of waiting for motivation.",
            "minutes": 25,
            "reason": "FocusAI detected a higher procrastination-risk pattern."
        }

    return {
        "title": "Keep the Momentum",
        "text": "Complete one focused session and protect the routine you already have.",
        "minutes": 30,
        "reason": "Your current pattern is relatively balanced."
    }


def start_focus(duration, task_name=None):
    st.session_state.active_session = True
    st.session_state.session_task = task_name
    st.session_state.session_duration = duration
    st.session_state.session_total_seconds = duration * 60
    st.session_state.session_end = time.time() + duration * 60
    st.session_state.session_started_at = time.time()
    st.session_state.session_paused_remaining = None


def pause_focus():
    if st.session_state.active_session and st.session_state.session_end:
        remaining = max(0, int(st.session_state.session_end - time.time()))
        st.session_state.session_paused_remaining = remaining
        st.session_state.session_end = None


def resume_focus():
    remaining = st.session_state.session_paused_remaining
    if remaining is not None and remaining > 0:
        st.session_state.session_end = time.time() + remaining
        st.session_state.session_paused_remaining = None


def finish_focus(completed=True):
    total = st.session_state.session_total_seconds
    elapsed = total

    if st.session_state.session_end:
        elapsed = total - max(0, int(st.session_state.session_end - time.time()))
    elif st.session_state.session_paused_remaining is not None:
        elapsed = total - st.session_state.session_paused_remaining

    # Record the actual time spent even if the student ends a session early.
    # Ignore very short accidental starts under one minute.
    minutes = max(0, round(elapsed / 60))

    if minutes > 0:
        add_study_minutes(minutes)

        task_name = st.session_state.session_task
        if task_name:
            for task in st.session_state.tasks:
                if task["task"] == task_name:
                    task["study_minutes"] = task.get("study_minutes", 0) + minutes
                    break

    if completed:
        st.session_state.completed_sessions += 1
        # Award points once when this focus session is finished.
        award_points(5, "Completed focus timer session")

    st.session_state.active_session = False
    st.session_state.session_task = None
    st.session_state.session_end = None
    st.session_state.session_started_at = None
    st.session_state.session_paused_remaining = None
    st.session_state.session_total_seconds = 25 * 60


def remaining_seconds():
    if not st.session_state.active_session:
        return 0

    if st.session_state.session_end:
        return max(0, int(st.session_state.session_end - time.time()))

    if st.session_state.session_paused_remaining is not None:
        return st.session_state.session_paused_remaining

    return 0


def format_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">🧠 FocusAI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Understand your study habits. Build better focus.</div>',
    unsafe_allow_html=True
)

# Reward points are kept for this app session.
st.markdown(
    f"""
    <div class="rewards-banner">
      <div class="rewards-trophy">🏆</div>
      <div class="rewards-copy">
        <div class="rewards-kicker">FOCUS REWARDS · TODAY</div>
        <div class="rewards-total">{st.session_state.focus_points} Focus Points</div>
        <div class="rewards-details">+10 per completed task <span>·</span> +5 per finished focus session <span>·</span> +20 daily challenge</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# ACTIVE FOCUS SESSION — ALWAYS VISIBLE
# ============================================================

if st.session_state.active_session:
    remaining = remaining_seconds()

    if remaining <= 0:
        finish_focus(True)
        st.success("🎉 Focus session complete! Your study time has been added to your progress.")
        st.rerun()

    task_label = st.session_state.session_task or "Focus session"

    st.markdown(
        f'<div class="active-box"><b>⏱️ Active Focus Session</b><br>'
        f'{task_label} · <b>{format_time(remaining)}</b></div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    if st.session_state.session_end:
        if b.button("⏸️ Pause", use_container_width=True):
            pause_focus()
            st.rerun()
    else:
        if b.button("▶️ Resume", use_container_width=True):
            resume_focus()
            st.rerun()

    if a.button("🔄 Restart", use_container_width=True):
        start_focus(st.session_state.session_duration, st.session_state.session_task)
        st.rerun()

    if c.button("✖️ End Session", use_container_width=True):
        finish_focus(False)
        st.rerun()

    # Refresh while running so countdown stays alive.
    if st.session_state.session_end:
        time.sleep(1)
        st.rerun()

# ============================================================
# NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 AI Predictor",
    "✅ My Tasks",
    "⏱️ Focus Timer",
    "📅 Study Calendar",
    "📊 My Progress"
])

# ============================================================
# TAB 1 — AI PREDICTOR
# ============================================================

with tab1:
    st.header("🔮 Procrastination Predictor")

    st.write(
        "Answer simple questions about your current routine. "
        "FocusAI will turn your answers into the inputs used by its trained model."
    )

    col1, col2 = st.columns(2)

    with col1:
        sleep_hours = st.selectbox(
            "😴 How much do you usually sleep per day?",
            options=[4, 5, 6, 7, 8, 9, 10, 11, 12],
            index=3,
            format_func=lambda x: f"{x} hours"
        )

        study_hours = st.selectbox(
            "📚 How many hours do you study per day?",
            options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            index=5,
            format_func=lambda x: f"{x} hours"
        )

        screen_time = st.selectbox(
            "📱 How many hours of screen time do you usually have?",
            options=list(range(0, 16)),
            index=5,
            format_func=lambda x: f"{x} hours"
        )

        study_session_length = st.radio(
            "⏱️ How long is a typical study session?",
            options=[25, 45, 60, 90, 120],
            index=2,
            format_func=lambda x: f"{x} min",
            horizontal=True
        )

        completion_pct = st.radio(
            "✅ Roughly what percentage of your planned work do you finish?",
            options=[25, 50, 60, 75, 85, 100],
            index=3,
            format_func=lambda x: f"{x}%",
            horizontal=True
        )

    with col2:
        deadline_urgency = st.selectbox(
            "⏰ How often do deadlines feel urgent?",
            ["Low", "Medium", "High"]
        )

        multitasking_level = st.selectbox(
            "🔀 How often do you switch between tasks while studying?",
            ["Low", "Medium", "High"]
        )

        peak_productivity_time = st.selectbox(
            "🌟 When do you usually feel most productive?",
            ["Morning", "Afternoon", "Night"]
        )

        task_difficulty = st.selectbox(
            "📖 How difficult is the work you are currently doing?",
            ["Easy", "Medium", "Hard"]
        )

        distraction_control = st.selectbox(
            "🎯 How easy is it for you to stay focused once you start?",
            options=["Very difficult", "Difficult", "Okay", "Easy", "Very easy"],
            index=2
        )

    # Convert natural answers into model features.
    focus_map = {
        "Very difficult": 2,
        "Difficult": 4,
        "Okay": 6,
        "Easy": 8,
        "Very easy": 10
    }

    focus_score = focus_map[distraction_control]

    # Productivity is estimated from behaviour instead of asking the student
    # to invent a "productivity score".
    productivity_score = (
        0.35 * (completion_pct / 10)
        + 0.25 * min(study_hours, 10)
        + 0.20 * min(study_session_length / 20, 10)
        + 0.20 * focus_score
    )
    productivity_score = max(1.0, min(10.0, productivity_score))

    st.caption(
        "FocusAI uses your answers to estimate the model inputs behind the prediction."
    )

    if st.button("🔮 Analyse My Risk", use_container_width=True, type="primary"):

        if model is None:
            st.error(
                f"Could not find {MODEL_FILE} (or {FALLBACK_MODEL_FILE}). Put the model .pkl file in the same folder as this app."
            )
        else:
            input_data = pd.DataFrame([{
                "sleep_hours": sleep_hours,
                "study_hours": study_hours,
                "screen_time": screen_time,
                "study_session_length": study_session_length,
                # The training dataset includes this feature; use the student's
                # reported typical session length as a practical estimate.
                "avg_study_session_length": study_session_length,
                "focus_score": focus_score,
                "productivity_score": productivity_score,
                "task_completion_rate": completion_pct,
                "deadline_urgency": deadline_urgency,
                "multitasking_level": multitasking_level,
                "peak_productivity_time": peak_productivity_time,
                "task_difficulty": task_difficulty
            }])

            try:
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                classes = list(model.classes_)
                probability_map = dict(zip(classes, probabilities))
                confidence = probability_map.get(prediction, 0) * 100

                st.session_state.prediction = prediction
                st.session_state.prediction_confidence = confidence

                reasons = []

                if screen_time >= 7:
                    reasons.append(
                        f"📱 Screen time — {screen_time:.1f} hrs/day\n"
                        "Your screen time is quite high compared with your study time, "
                        "which may make consistency harder."
                    )
                elif screen_time <= 3:
                    reasons.append(
                        f"📱 Screen time — {screen_time:.1f} hrs/day\n"
                        "Your reported screen time leaves more room for study."
                    )

                if study_session_length <= 25:
                    reasons.append(
                        f"⏱️ Study sessions — {study_session_length} min\n"
                        "Your sessions are short. A focused 25-minute block can be a useful starting point."
                    )
                elif study_session_length >= 90:
                    reasons.append(
                        f"⏱️ Study sessions — {study_session_length} min\n"
                        "Your sessions are long, so planned breaks can help you maintain consistency."
                    )

                if completion_pct < 70:
                    reasons.append(
                        f"🎯 Task completion — {completion_pct}%\n"
                        "Some planned work is being left unfinished. Smaller task steps may help."
                    )
                elif completion_pct >= 85:
                    reasons.append(
                        f"🎯 Task completion — {completion_pct}%\n"
                        "You are completing most of your planned work, which is a strong routine signal."
                    )

                if sleep_hours < 6:
                    reasons.append(
                        f"😴 Sleep — {sleep_hours:.1f} hrs/day\n"
                        "Your reported sleep is relatively low, so keeping a consistent routine may help."
                    )

                if multitasking_level == "High":
                    reasons.append(
                        "🔀 Multitasking\n"
                        "You reported switching between tasks frequently. One-task focus blocks may help."
                    )

                if deadline_urgency == "High":
                    reasons.append(
                        "⏰ Deadline pressure\n"
                        "You reported high deadline urgency. Starting important work earlier may reduce pressure."
                    )

                if task_difficulty == "Hard":
                    reasons.append(
                        "📖 Task difficulty\n"
                        "Breaking difficult work into smaller milestones can make starting easier."
                    )

                if not reasons:
                    reasons.append(
                        f"📊 Your reported routine — {study_hours} study hrs, "
                        f"{screen_time} screen hrs, {completion_pct}% task completion\n"
                        "No single input crossed the app's simple habit flags. "
                        "This is a summary of your answers, not proof of what caused the model prediction."
                    )

                st.session_state.prediction_reasons = reasons

                st.session_state.challenge = create_challenge(
                    screen_time,
                    study_hours,
                    study_session_length,
                    completion_pct,
                    deadline_urgency,
                    multitasking_level,
                    task_difficulty,
                    prediction
                )

            except Exception as e:
                st.error("The model could not process these inputs.")
                st.exception(e)

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if st.session_state.prediction:

        prediction = st.session_state.prediction
        confidence = st.session_state.prediction_confidence

        if prediction == "High":
            st.error("🔴 HIGH PROCRASTINATION RISK")
            result_message = (
                "Your current pattern shows a higher tendency toward procrastination. "
                "A structured focus session can help you regain momentum."
            )
        elif prediction == "Medium":
            st.warning("🟡 MEDIUM PROCRASTINATION RISK")
            result_message = (
                "Your current pattern shows some signs of procrastination, "
                "but there is clear room to improve consistency."
            )
        else:
            st.success("🟢 LOW PROCRASTINATION RISK")
            result_message = (
                "Your current pattern looks fairly consistent. "
                "Keep building on the habits that are working."
            )

        st.markdown(
            f'<div class="result-box"><b>Model estimated probability for this class: {confidence:.0f}%</b><br>'
            f'{result_message}</div>',
            unsafe_allow_html=True
        )

       

    # --------------------------------------------------------
    # EXPLANATIONS
    # --------------------------------------------------------
    if st.session_state.prediction:
        st.subheader("🔎 Why FocusAI gave you this result")
        st.caption("These are personalized habit insights based on the answers you entered. They are not a claim that each factor caused the model's prediction; the model's class probability is shown separately above.")

        reasons = st.session_state.prediction_reasons or [
            "✨ Study pattern\nYour result is based on the routine details you entered."
        ]

        for reason in reasons[:4]:
            parts = str(reason).split("\n", 1)
            title = parts[0].strip() or "FocusAI insight"
            explanation = (
                parts[1].strip()
                if len(parts) > 1 and parts[1].strip()
                else "This result is based on the routine details you provided."
            )
            with st.expander(title):
                st.write(explanation)

        st.subheader("💡 Your personalized improvement plan")
        recommendations = []
        if screen_time >= 7:
            recommendations.append(("Reduce one screen-time trigger", "Put your phone on Do Not Disturb and keep it out of reach for your next focus block."))
        if completion_pct < 70:
            recommendations.append(("Shrink the next task", "Rewrite one unfinished task as a step that takes 10–15 minutes, then complete only that step."))
        if multitasking_level == "High":
            recommendations.append(("Use one-task mode", "Keep only the materials for one subject open until the timer ends."))
        if sleep_hours < 7:
            recommendations.append(("Protect your rest", "Plan a realistic stopping time tonight; avoid trying to fix tiredness by adding more study hours."))
        if deadline_urgency == "High":
            recommendations.append(("Start with the nearest deadline", "Write the very first action for your most urgent task and begin with that."))
        if task_difficulty == "Hard":
            recommendations.append(("Make hard work easier to start", "Do one worked example or the first three questions before judging the whole task."))
        if study_hours <= 3:
            recommendations.append(("Build gradually", "Add one manageable 25-minute focus block rather than planning a sudden huge increase."))
        if not recommendations:
            recommendations.append(("Protect what is working", "Choose your most important task, do one focused block, and record the time in your calendar."))

        for title, advice in recommendations[:4]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(advice)

        # ----------------------------------------------------
        # CHALLENGE
        # ----------------------------------------------------
        challenge = st.session_state.challenge
        if challenge:
            st.subheader("🎯 Today's Focus Challenge")
            st.markdown(
                f'<div class="challenge-box">'
                f'<h3>{challenge["title"]}</h3>'
                f'<p>{challenge["text"]}</p>'
                f'<p class="small-muted">{challenge["reason"]}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            today = date.today().isoformat()
            if st.session_state.challenge_completed_date == today:
                st.success("✅ Today's challenge completed! You earned 20 Focus Points. Come back tomorrow for a new challenge.")
            else:
                challenge_col1, challenge_col2 = st.columns(2)
                with challenge_col1:
                    if st.button(
                        f"▶️ Start {challenge['minutes']}-minute Challenge",
                        use_container_width=True,
                        key="start_daily_challenge"
                    ):
                        start_focus(challenge["minutes"], challenge.get("title", "Today's Focus Challenge"))
                        st.rerun()
                with challenge_col2:
                    if st.button(
                        "🏆 Mark challenge complete (+20 points)",
                        use_container_width=True,
                        key="complete_daily_challenge"
                    ):
                        st.session_state.challenge_completed_date = today
                        award_points(20, "Completed daily challenge")
                        st.success("Amazing! You earned 20 Focus Points for today's challenge.")
                        st.rerun()

            st.caption("Tip: mark the challenge complete only after you've genuinely tried it. FocusAI rewards progress, not perfection.")

    # --------------------------------------------------------
    # TECHNICAL DETAILS
    # --------------------------------------------------------
    with st.expander("🧠 About FocusAI — Technical details"):
        st.write(
            "FocusAI uses a trained machine-learning classification model "
            "to predict Low, Medium, or High procrastination risk."
        )
        st.write(
            "The current held-out evaluation was 79.75% test accuracy and 76.22% macro F1. "
            "These are preliminary dataset test results; possible target leakage has not been ruled out. "
            "Performance depends on the dataset and evaluation method; this is not a "
            "guarantee about an individual student's behaviour."
        )
        st.write(
            "The model uses behavioural inputs such as sleep, study time, screen time, "
            "study-session length and its average, focus, productivity, task completion and study-pattern information."
        )
        st.caption(
            "Prediction confidence is the model's estimated probability for its chosen class, "
            "not a guarantee that the prediction is correct."
        )

# ============================================================
# TAB 2 — TASKS
# ============================================================

with tab2:
    st.header("✅ My Study Tasks")

    st.write("Plan your work, break it into smaller steps and focus on one thing at a time.")

    with st.form("task_form", clear_on_submit=True):
        task_name = st.text_input(
            "Task",
            placeholder="e.g. Finish Electrochemistry PYQs"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            task_deadline = st.date_input(
                "Deadline",
                value=date.today()
            )

        with c2:
            task_priority = st.selectbox(
                "Priority",
                ["Low", "Medium", "High"]
            )

        with c3:
            estimated_minutes = st.number_input(
                "Estimated time (minutes)",
                min_value=5,
                max_value=600,
                value=30,
                step=5
            )

        subtasks_text = st.text_area(
            "Subtasks",
            placeholder="One subtask per line\nRead notes\nSolve examples\nDo PYQs"
        )

        submitted = st.form_submit_button(
            "➕ Add Task",
            use_container_width=True
        )

        if submitted and task_name.strip():
            subtasks = [
                {
                    "text": x.strip(),
                    "completed": False
                }
                for x in subtasks_text.splitlines()
                if x.strip()
            ]

            st.session_state.tasks.append({
                "task": task_name.strip(),
                "completed": False,
                "deadline": task_deadline.isoformat(),
                "priority": task_priority,
                "estimated_minutes": int(estimated_minutes),
                "subtasks": subtasks,
                "study_minutes": 0
            })

            st.success("Task added!")
            st.rerun()

    st.divider()

    total = total_task_count()
    completed = completed_task_count()

    if total:
        st.progress(completed / total)
        st.write(f"**{completed}/{total} tasks completed**")

    if not st.session_state.tasks:
        st.info("📋 No tasks yet. Add your first study task above.")

    else:
        for i, task in enumerate(st.session_state.tasks):

            deadline = date.fromisoformat(task["deadline"])
            days_left = (deadline - date.today()).days

            if days_left < 0:
                deadline_text = "🔴 Overdue"
            elif days_left == 0:
                deadline_text = "🟠 Due today"
            elif days_left == 1:
                deadline_text = "🟡 Due tomorrow"
            else:
                deadline_text = f"📅 Due in {days_left} days"

            with st.container(border=True):
                top1, top2 = st.columns([5, 1])

                with top1:
                    completed_before = bool(task["completed"])
                    completed_value = st.checkbox(
                        task["task"],
                        value=completed_before,
                        key=f"complete_{i}"
                    )

                    # Award points only the first time this task is completed.
                    if completed_value and not completed_before and not task.get("points_awarded", False):
                        award_points(10, "Completed task")
                        st.session_state.tasks[i]["points_awarded"] = True
                        st.toast("🏆 +10 Focus Points! Task completed.")
                    st.session_state.tasks[i]["completed"] = completed_value

                with top2:
                    if st.button("🗑️", key=f"delete_{i}"):
                        st.session_state.tasks.pop(i)
                        st.rerun()

                st.caption(
                    f"{deadline_text} · Priority: {task['priority']} · "
                    f"Estimated: {task['estimated_minutes']} min · "
                    f"Focused: {task.get('study_minutes', 0)} min"
                )

                if task["subtasks"]:
                    st.write("**Subtasks**")

                    for j, subtask in enumerate(task["subtasks"]):
                        sub_value = st.checkbox(
                            subtask["text"],
                            value=subtask["completed"],
                            key=f"sub_{i}_{j}"
                        )
                        st.session_state.tasks[i]["subtasks"][j]["completed"] = sub_value

                if not task["completed"]:
                    if st.button(
                        "🎯 Focus on this task",
                        key=f"focus_task_{i}",
                        use_container_width=True
                    ):
                        start_focus(
                            task["estimated_minutes"],
                            task["task"]
                        )
                        st.rerun()

# ============================================================
# TAB 3 — TIMER
# ============================================================

with tab3:
    st.header("⏱️ Focus Timer")

    st.write("Choose a session length and focus on one task.")

    if st.session_state.active_session:
        remaining = remaining_seconds()

        st.metric(
            "Current session",
            format_time(remaining)
        )

        st.write(
            f"**Task:** {st.session_state.session_task or 'General study'}"
        )

        if st.session_state.session_end:
            st.info("Your timer is running. The active session also stays visible above the tabs.")
        else:
            st.warning("Timer paused.")

    else:
        duration = st.radio(
            "Choose your focus time",
            [25, 45, 60, 90],
            horizontal=True
        )

        custom_subject_option = "✏️ Enter a subject or task..."
        selected_task_names = ["General study", custom_subject_option] + [
            t["task"]
            for t in st.session_state.tasks
            if not t["completed"]
        ]

        selected_task = st.selectbox(
            "What are you working on?",
            selected_task_names,
            key="focus_timer_task_choice",
            help="Choose one of your existing tasks or enter any subject yourself."
        )

        custom_subject = ""
        if selected_task == custom_subject_option:
            custom_subject = st.text_input(
                "Enter your subject or task",
                placeholder="e.g. Physics, Electrostatics, Chemistry, Maths...",
                key="focus_timer_custom_subject",
                max_chars=100
            ).strip()
            if custom_subject:
                st.caption(f"Your focus session will be for: **{custom_subject}**")

        if st.button(
            f"▶️ Start {duration}-minute Focus Session",
            use_container_width=True,
            type="primary"
        ):
            if selected_task == custom_subject_option:
                if not custom_subject:
                    st.warning("Please enter a subject or task before starting the timer.")
                    st.stop()
                task_name = custom_subject
            else:
                task_name = None if selected_task == "General study" else selected_task

            start_focus(duration, task_name)
            st.rerun()

        st.divider()

        st.write("**How the timer works**")
        st.write("• Starting a session creates an active focus session.")
        st.write("• The timer remains visible while you move through the app.")
        st.write("• Completed sessions are automatically added to your study calendar.")
        st.write("• If you focus on a task, its focused minutes are also recorded.")

# ============================================================
# TAB 4 — CALENDAR
# ============================================================

with tab4:
    st.header("📅 Study Calendar")
    st.write(
        "See your whole month at a glance. Timer sessions are added automatically "
        "when you finish or end a session, and you can log offline study manually."
    )

    # Month navigation
    month_value = st.session_state.calendar_month
    nav_prev, nav_title, nav_next = st.columns([1, 4, 1])

    with nav_prev:
        if st.button("← Previous", key="calendar_prev_month", use_container_width=True):
            year, month = month_value.year, month_value.month
            if month == 1:
                st.session_state.calendar_month = date(year - 1, 12, 1)
            else:
                st.session_state.calendar_month = date(year, month - 1, 1)
            st.rerun()

    with nav_title:
        st.markdown(
            f"<h3 style='text-align:center;margin-top:4px'>{month_value.strftime('%B %Y')}</h3>",
            unsafe_allow_html=True
        )

    with nav_next:
        if st.button("Next →", key="calendar_next_month", use_container_width=True):
            year, month = month_value.year, month_value.month
            if month == 12:
                st.session_state.calendar_month = date(year + 1, 1, 1)
            else:
                st.session_state.calendar_month = date(year, month + 1, 1)
            st.rerun()

    # Whole-month calendar grid, with each day's recorded hours shown in its cell.
    cal = calendar.Calendar(firstweekday=0)
    month_weeks = cal.monthdatescalendar(month_value.year, month_value.month)
    weekday_cols = st.columns(7)
    for col, weekday in zip(weekday_cols, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        col.markdown(f"**{weekday}**")

    for week_index, week in enumerate(month_weeks):
        day_cols = st.columns(7)
        for col, day in zip(day_cols, week):
            is_current_month = day.month == month_value.month
            minutes = st.session_state.study_log.get(day.isoformat(), 0)
            hours = minutes / 60
            if not is_current_month:
                col.markdown(
                    f"<div style='min-height:84px;padding:8px;border:1px solid #e5e7eb;"
                    f"border-radius:8px;opacity:0.35'><b>{day.day}</b><br> </div>",
                    unsafe_allow_html=True
                )
            else:
                if minutes >= 180:
                    bg = "#dcfce7"
                elif minutes > 0:
                    bg = "#dbeafe"
                else:
                    bg = "#f3f4f6"
                border = "#86efac" if day == date.today() else "#d1d5db"
                col.markdown(
                    f"<div style='min-height:84px;padding:8px;border:1px solid {border};"
                    f"border-radius:8px;background:{bg};color:#1f2937'>"
                    f"<b>{day.day}</b><br><span style='font-size:13px'>"
                    f"{hours:.1f} study hr</span></div>",
                    unsafe_allow_html=True
                )

    month_total = sum(
        minutes for key, minutes in st.session_state.study_log.items()
        if key.startswith(f"{month_value.year:04d}-{month_value.month:02d}-")
    )
    st.metric("Study time this month", f"{month_total // 60}h {month_total % 60}m")

    st.divider()
    st.subheader("✍️ Log study time manually")
    st.caption(
        "Use this if you studied offline or forgot to run the timer. "
        "Manual entries are added to the existing total for that day, not substituted for timer time."
    )

    manual_col1, manual_col2, manual_col3 = st.columns([2, 2, 1])
    with manual_col1:
        manual_day = st.date_input(
            "Study date",
            value=date.today(),
            key="manual_study_day"
        )
    with manual_col2:
        manual_hours = st.number_input(
            "Hours studied offline",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.25,
            key="manual_study_hours"
        )
    with manual_col3:
        st.write("")
        st.write("")
        if st.button("➕ Add time", key="add_manual_study_time", use_container_width=True):
            added_minutes = int(round(manual_hours * 60))
            if added_minutes <= 0:
                st.warning("Enter a study duration greater than zero.")
            else:
                add_study_minutes(added_minutes, manual_day)
                earned_points = (added_minutes // 15) * 2
                if earned_points > 0:
                    award_points(earned_points, "Manual study time logged", manual_day)
                st.success(
                    f"Added {added_minutes} minutes to {manual_day.strftime('%d %b %Y')}. "
                    + (f"🏆 +{earned_points} Focus Points earned!" if earned_points else "Log 15+ minutes to earn Focus Points.")
                )
                st.rerun()

    st.divider()
    st.subheader("📋 Daily study details")
    selected_day = st.date_input(
        "Choose a day to inspect",
        value=date.today(),
        key="calendar_detail_day"
    )
    selected_minutes = st.session_state.study_log.get(selected_day.isoformat(), 0)
    st.info(
        f"{selected_day.strftime('%A, %d %B %Y')}: "
        f"{selected_minutes // 60}h {selected_minutes % 60}m recorded."
    )

# ============================================================
# TAB 5 — PROGRESS
# ============================================================

with tab5:
    today_rewards = st.session_state.daily_rewards.get(today_key(), {"total": 0, "events": []})
    st.subheader("🏆 Today's Focus Rewards")
    reward_col1, reward_col2, reward_col3 = st.columns(3)
    with reward_col1:
        st.metric("✨ Points earned today", int(today_rewards.get("total", 0)))
    with reward_col2:
        st.metric("🎯 Reward actions", len(today_rewards.get("events", [])))
    with reward_col3:
        all_time_points = sum(int(v.get("total", 0)) for v in st.session_state.daily_rewards.values())
        st.metric("🌟 All-time points", all_time_points)
    st.caption("Today's points reset to 0 at the start of a new day. Previous days remain saved in your local rewards history.")
    with st.expander("📅 Daily points history", expanded=False):
        history_rows = []
        for day_key, entry in sorted(st.session_state.daily_rewards.items(), reverse=True):
            history_rows.append({"Date": day_key, "Points earned": int(entry.get("total", 0)), "Reward actions": len(entry.get("events", []))})
        if history_rows:
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Your rewards history will appear here after you earn your first points.")
    st.divider()
    st.header("📊 Your Progress Dashboard")
    st.caption("Small, consistent sessions count. Track your effort, celebrate milestones, and choose one next step.")

    total_minutes = sum(st.session_state.study_log.values())
    week_minutes = weekly_minutes()
    completion = task_completion_rate()
    streak = current_streak()
    total_tasks = total_task_count()
    done_tasks = completed_task_count()
    study_days = sum(1 for value in st.session_state.study_log.values() if value > 0)

    # A user-adjustable goal makes the dashboard feel personal and actionable.
    goal_col, goal_note_col = st.columns([1, 2])
    with goal_col:
        weekly_goal_hours = st.number_input(
            "🎯 Weekly study goal (hours)", min_value=1, max_value=80,
            value=15, step=1, key="progress_weekly_goal_hours"
        )
    weekly_goal_minutes = weekly_goal_hours * 60
    goal_progress = min(week_minutes / weekly_goal_minutes, 1.0) if weekly_goal_minutes else 0
    with goal_note_col:
        st.markdown("### Your week, your pace")
        st.write(f"You've logged **{week_minutes / 60:.1f} of {weekly_goal_hours} hours** this week.")
        st.progress(goal_progress)
        if week_minutes >= weekly_goal_minutes:
            st.success("Weekly goal reached! Celebrate the consistency that got you here.")
        else:
            remaining = max(0, weekly_goal_minutes - week_minutes)
            st.caption(f"{remaining // 60}h {remaining % 60}m left to reach your goal. One focused session moves the bar.")

    st.divider()
    st.subheader("🏁 Your key numbers")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📚 Total recorded", f"{total_minutes // 60}h {total_minutes % 60}m", help="All study time recorded in your calendar and manual logs.")
    with c2:
        st.metric("📅 Last 7 days", f"{week_minutes // 60}h {week_minutes % 60}m")
    with c3:
        st.metric("✅ Tasks completed", f"{done_tasks}/{total_tasks}", help="Based on tasks marked complete in My Tasks.")
    with c4:
        st.metric("🔥 Current streak", f"{streak} day{'s' if streak != 1 else ''}")

    st.divider()
    st.subheader("📈 Your last 7 days")
    start = date.today() - timedelta(days=6)
    chart_rows = []
    for i in range(7):
        d = start + timedelta(days=i)
        minutes = st.session_state.study_log.get(d.isoformat(), 0)
        chart_rows.append({"Day": d.strftime("%a · %d %b"), "Study hours": round(minutes / 60, 2)})
    chart_df = pd.DataFrame(chart_rows).set_index("Day")
    st.bar_chart(chart_df, use_container_width=True)
    recorded_days_week = sum(1 for row in chart_rows if row["Study hours"] > 0)
    best_day = max(chart_rows, key=lambda row: row["Study hours"])
    info_col, best_col = st.columns(2)
    with info_col:
        st.markdown("**Consistency this week**")
        st.write(f"You recorded study time on **{recorded_days_week}/7 days**.")
    with best_col:
        st.markdown("**Your strongest day**")
        st.write(f"**{best_day['Day']}** — {best_day['Study hours']:.1f} hours")

    st.divider()
    st.subheader("🏆 Milestones")
    milestones = [
        ("First steps", total_minutes >= 25, "Log your first 25 minutes of study"),
        ("Hour builder", total_minutes >= 60, "Reach 1 total study hour"),
        ("Three-day rhythm", streak >= 3, "Record study time 3 days in a row"),
        ("Focus timer user", st.session_state.completed_sessions >= 1, "Complete a timer session"),
        ("Task finisher", done_tasks >= 1, "Mark a task as complete"),
        ("Challenge accepted", st.session_state.challenge_completed_date == today_key(), "Complete today's Focus Challenge"),
    ]
    achieved = sum(1 for _, unlocked, _ in milestones if unlocked)
    st.write(f"**{achieved} of {len(milestones)} milestones unlocked**")
    st.progress(achieved / len(milestones))
    milestone_cols = st.columns(3)
    for i, (name, unlocked, description) in enumerate(milestones):
        with milestone_cols[i % 3]:
            if unlocked:
                st.success(f"🏅 **{name}**\n\n{description}")
            else:
                st.info(f"🔒 **{name}**\n\n{description}")

    st.divider()
    st.subheader("🧭 Your next best step")
    next_steps = []
    if week_minutes == 0:
        next_steps.append("Start with one manageable 15–25 minute session and log it in the Study Calendar.")
    elif goal_progress < 1:
        next_steps.append(f"Plan one focused block next. You need {max(1, (weekly_goal_minutes - week_minutes + 24) // 25)} more 25-minute block(s) to get closer to your weekly goal.")
    else:
        next_steps.append("You've reached your weekly target. Keep your routine sustainable rather than adding hours just for the number.")
    if total_tasks == 0:
        next_steps.append("Add one specific task in My Tasks so you can track completion, not just time spent.")
    elif completion < 70:
        next_steps.append("Choose one unfinished task and shrink it into a clear next action you can complete in 10–15 minutes.")
    else:
        next_steps.append("Your task completion is moving well. Keep your task list realistic and celebrate finished work.")
    if streak == 0:
        next_steps.append("A streak starts with today: record any honest study session in the calendar.")
    elif streak >= 3:
        next_steps.append(f"You're on a {streak}-day streak. Protect the habit, but remember rest days are allowed.")
    for step in next_steps:
        st.markdown(f"- {step}")

    st.divider()
    st.subheader("💡 FocusAI reflection")
    if st.session_state.prediction:
        risk = st.session_state.prediction
        if risk == "High":
            st.warning("Your latest prediction indicates higher procrastination risk based on the information entered. Use this as a prompt to add structure: pick one task, set a short timer, and reduce distractions.")
        elif risk == "Medium":
            st.info("Your latest prediction indicates moderate procrastination risk. A realistic task list and repeatable focus blocks may help you build consistency.")
        else:
            st.success("Your latest prediction indicates lower procrastination risk. Keep using your calendar and timer to maintain the habits that support you.")
        st.caption("This is a model estimate, not a diagnosis or a guarantee. Your logged progress reflects recorded activity only.")
    else:
        st.info("Run a prediction in the AI Predictor tab to connect your latest result with your study routine. You can still use this dashboard without it.")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FocusAI • Student productivity assistant • "
    "Prediction + Tasks + Focus Timer + Calendar + Progress"
)
