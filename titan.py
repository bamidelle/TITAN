# titan_full_vA.py
"""
TITAN — Full single-file Streamlit backend (Option A: auto-move pipeline but always allow manual override)
Features:
- SQLite-backed persistence (sqlite3)
- Lead Capture, Pipeline Board, Analytics & SLA, CPA/ROI pages
- Top 5 Priority leads (black cards)
- KPI cards (2 rows), date range selector at top-right
- Notification bell with dismissable dropdown & count
- Audit trail (lead_history)
- Imports/Exports (CSV, Excel fallback)
- Internal ML model (train & score) — no user tuning
- AI Observation & Recommendation + Report Summary
- SLA enforcement (>0 hours)
- Search & quick filters, mobile-friendly layout
- Auto stage progression (when flags change) but manual override allowed
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import plotly.express as px
from datetime import datetime, timedelta, date, timezone
import io, os, math, json, base64, traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------
# CONFIG
# ----------------------------
DB_FILE = "titan_vA.db"
MODEL_FILE = "titan_vA_model.joblib"
DEFAULT_SLA_HOURS = 24
PIPELINE_STAGES = [
    "New",
    "Contacted",
    "Inspection Scheduled",
    "Inspection Completed",
    "Estimate Submitted",
    "Qualified",
    "Won",
    "Lost"
]
NOTIFICATION_TABLE = "notifications"  # simple notifications table

# ----------------------------
# DB UTILITIES (sqlite3)
# ----------------------------
def get_conn():
    # ensures DB file exists in current working directory
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    # leads table
    c.execute("""
    CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id TEXT UNIQUE,
        created_at TEXT,
        source TEXT,
        source_details TEXT,
        contact_name TEXT,
        contact_phone TEXT,
        contact_email TEXT,
        property_address TEXT,
        damage_type TEXT,
        assigned_to TEXT,
        notes TEXT,
        estimated_value REAL DEFAULT 0,
        stage TEXT DEFAULT 'New',
        sla_hours INTEGER DEFAULT ?,
        sla_entered_at TEXT,
        contacted INTEGER DEFAULT 0,
        inspection_scheduled INTEGER DEFAULT 0,
        inspection_scheduled_at TEXT,
        inspection_completed INTEGER DEFAULT 0,
        estimate_submitted INTEGER DEFAULT 0,
        estimate_submitted_at TEXT,
        awarded_date TEXT,
        awarded_invoice TEXT,
        lost_date TEXT,
        qualified INTEGER DEFAULT 0,
        ad_cost REAL DEFAULT 0,
        converted INTEGER DEFAULT 0,
        score REAL DEFAULT 0.0
    );
    """, (DEFAULT_SLA_HOURS,))
    # history / audit
    c.execute("""
    CREATE TABLE IF NOT EXISTS lead_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id TEXT,
        who TEXT,
        field TEXT,
        old_value TEXT,
        new_value TEXT,
        timestamp TEXT
    );
    """)
    # notifications
    c.execute(f"""
    CREATE TABLE IF NOT EXISTS {NOTIFICATION_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        message TEXT,
        created_at TEXT,
        read INTEGER DEFAULT 0
    );
    """)
    conn.commit()
    conn.close()

init_db()

# ----------------------------
# BASIC HELPERS
# ----------------------------
def now_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def safe_float(x, default=0.0):
    try:
        return float(x or 0.0)
    except:
        return default

def format_money(x):
    try:
        return f"${float(x):,.2f}"
    except:
        return "$0.00"

# ----------------------------
# LEAD CRUD / AUDIT / NOTIFS
# ----------------------------
def fetch_leads(start_date=None, end_date=None):
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM leads ORDER BY created_at DESC", conn, parse_dates=["created_at","sla_entered_at","inspection_scheduled_at","estimate_submitted_at","awarded_date","lost_date"])
    conn.close()
    if df.empty:
        # define columns to avoid KeyErrors downstream
        cols = ["id","lead_id","created_at","source","source_details","contact_name","contact_phone","contact_email","property_address","damage_type",
                "assigned_to","notes","estimated_value","stage","sla_hours","sla_entered_at","contacted","inspection_scheduled","inspection_scheduled_at",
                "inspection_completed","estimate_submitted","estimate_submitted_at","awarded_date","awarded_invoice","lost_date","qualified","ad_cost","converted","score"]
        return pd.DataFrame(columns=cols)
    # cast numeric columns
    for col in ["estimated_value","ad_cost","score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ["contacted","inspection_scheduled","inspection_completed","estimate_submitted","qualified","converted","sla_hours"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    # filter by date if provided (dates are inclusive)
    if start_date:
        df = df[df["created_at"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["created_at"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    return df.reset_index(drop=True)

def add_history(lead_id, who, field, old, new):
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("INSERT INTO lead_history (lead_id,who,field,old_value,new_value,timestamp) VALUES (?,?,?,?,?,?)",
                  (lead_id, who, field, str(old), str(new), now_iso()))
        conn.commit(); conn.close()
    except Exception:
        pass

def add_notification(title, message):
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute(f"INSERT INTO {NOTIFICATION_TABLE} (title,message,created_at,read) VALUES (?,?,?,0)", (title, message, now_iso()))
        conn.commit(); conn.close()
    except Exception:
        pass

def fetch_notifications(limit=50):
    conn = get_conn(); c = conn.cursor()
    c.execute(f"SELECT id,title,message,created_at,read FROM {NOTIFICATION_TABLE} ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    # convert to dict list
    notifs = [{"id": r[0], "title": r[1], "message": r[2], "created_at": r[3], "read": bool(r[4])} for r in rows]
    return notifs

def mark_notification_read(nid):
    conn = get_conn(); c = conn.cursor()
    c.execute(f"UPDATE {NOTIFICATION_TABLE} SET read=1 WHERE id=?", (nid,))
    conn.commit(); conn.close()

def save_lead(payload, who="admin"):
    """
    Upsert a lead. payload is a dict of fields to set. Returns lead_id.
    Behavior:
      - If lead_id not supplied: create a new lead_id and insert.
      - If existing lead: update provided fields; create audit entries for changed fields.
      - Auto-move stage forward if flags indicate progression (Option A) but manual stage value in payload overrides auto-move.
    """
    # normalize
    p = payload.copy()
    now = now_iso()
    if not p.get("lead_id"):
        p["lead_id"] = f"L{int(datetime.utcnow().timestamp())}"
    # ensure SLA hours valid
    if "sla_hours" in p:
        try:
            if int(p.get("sla_hours") or DEFAULT_SLA_HOURS) <= 0:
                raise ValueError("SLA must be > 0")
        except Exception:
            p["sla_hours"] = DEFAULT_SLA_HOURS

    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT * FROM leads WHERE lead_id=?", (p["lead_id"],))
    existing = c.fetchone()
    # Helper to map row->dict using cursor description
    if not existing:
        # Insert — fill defaults
        c.execute("""
        INSERT INTO leads (lead_id, created_at, source, source_details, contact_name, contact_phone, contact_email,
                           property_address, damage_type, assigned_to, notes, estimated_value, stage, sla_hours, sla_entered_at,
                           contacted, inspection_scheduled, inspection_scheduled_at, inspection_completed, estimate_submitted, estimate_submitted_at,
                           awarded_date, awarded_invoice, lost_date, qualified, ad_cost, converted, score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            p.get("lead_id"),
            p.get("created_at", now),
            p.get("source"),
            p.get("source_details"),
            p.get("contact_name"),
            p.get("contact_phone"),
            p.get("contact_email"),
            p.get("property_address"),
            p.get("damage_type"),
            p.get("assigned_to"),
            p.get("notes"),
            safe_float(p.get("estimated_value")),
            p.get("stage", "New"),
            int(p.get("sla_hours") or DEFAULT_SLA_HOURS),
            p.get("sla_entered_at") or now,
            int(bool(p.get("contacted"))),
            int(bool(p.get("inspection_scheduled"))),
            p.get("inspection_scheduled_at"),
            int(bool(p.get("inspection_completed"))),
            int(bool(p.get("estimate_submitted"))),
            p.get("estimate_submitted_at"),
            p.get("awarded_date"),
            p.get("awarded_invoice"),
            p.get("lost_date"),
            int(bool(p.get("qualified"))),
            safe_float(p.get("ad_cost")),
            int(bool(p.get("converted"))),
            safe_float(p.get("score"))
        ))
        c.execute("INSERT INTO lead_history (lead_id,who,field,old_value,new_value,timestamp) VALUES (?,?,?,?,?,?)",
                  (p.get("lead_id"), who, "create", "", p.get("stage", "New"), now))
        conn.commit()
        conn.close()
        return p.get("lead_id")
    else:
        # Update: fetch current values
        # get column names
        col_names = [d[0] for d in c.description]
        # But easier: read current row to dict via select
        conn.close()
        # read current values via fetch_leads for that lead
        df = fetch_leads()
        cur = df[df["lead_id"] == p["lead_id"]]
        if cur.empty:
            # weird — but insert fallback
            return save_lead({**p}, who=who)
        cur = cur.iloc[0].to_dict()

        # Auto-progression logic (Option A)
        # If certain flags provided (inspection_completed, estimate_submitted, contacted...), propose an auto-stage
        # But we allow manual stage override in payload: if payload has 'stage', use it exactly
        auto_stage = None
        try:
            # if inspection_completed true -> move to Inspection Completed (unless manual stage provided)
            if p.get("inspection_completed") or p.get("inspection_completed") == 1:
                auto_stage = "Inspection Completed"
            elif p.get("inspection_scheduled") or p.get("inspection_scheduled") == 1:
                auto_stage = "Inspection Scheduled"
            elif p.get("estimate_submitted") or p.get("estimate_submitted") == 1:
                auto_stage = "Estimate Submitted"
            elif p.get("contacted") or p.get("contacted") == 1:
                auto_stage = "Contacted"
        except Exception:
            auto_stage = None

        # If user supplied stage explicitly, that takes precedence (manual override)
        if p.get("stage"):
            chosen_stage = p.get("stage")
        else:
            chosen_stage = auto_stage or cur.get("stage") or "New"

        # fields allowed to update
        up_fields = ["source","source_details","contact_name","contact_phone","contact_email","property_address","damage_type","assigned_to","notes",
                     "estimated_value","stage","sla_hours","sla_entered_at","contacted","inspection_scheduled","inspection_scheduled_at",
                     "inspection_completed","estimate_submitted","estimate_submitted_at","awarded_date","awarded_invoice","lost_date","qualified","ad_cost","converted","score"]
        conn = get_conn(); c = conn.cursor()
        for f in up_fields:
            if f in p:
                old_val = cur.get(f)
                new_val = p.get(f)
                # convert booleans to ints when needed
                if f in ["contacted","inspection_scheduled","inspection_completed","estimate_submitted","qualified","converted"]:
                    new_val = int(bool(new_val))
                if f in ["estimated_value","ad_cost","score"]:
                    new_val = safe_float(new_val)
                if f == "sla_hours":
                    try:
                        new_val = int(new_val)
                        if new_val <= 0:
                            new_val = DEFAULT_SLA_HOURS
                    except:
                        new_val = DEFAULT_SLA_HOURS
                if str(old_val) != str(new_val):
                    c.execute(f"UPDATE leads SET {f} = ? WHERE lead_id = ?", (new_val, p.get("lead_id")))
                    add_history(p.get("lead_id"), who, f, old_val, new_val)
        # ensure chosen_stage stored too (if differs from computed)
        try:
            c.execute("SELECT stage FROM leads WHERE lead_id=?", (p.get("lead_id"),))
            cur_stage = c.fetchone()[0]
            if cur_stage != chosen_stage:
                c.execute("UPDATE leads SET stage=? WHERE lead_id=?", (chosen_stage, p.get("lead_id")))
                add_history(p.get("lead_id"), who, "stage", cur_stage, chosen_stage)
        except Exception:
            pass

        conn.commit(); conn.close()
        # generate notification when SLA breached or high-value lead not contacted
        try:
            if int(p.get("estimated_value", 0)) >= 10000 and not p.get("contacted"):
                add_notification("High value lead not contacted", f"Lead {p.get('lead_id')} estimated {p.get('estimated_value')}")
        except:
            pass
        return p.get("lead_id")

def delete_lead(lead_id):
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("DELETE FROM leads WHERE lead_id=?", (lead_id,))
        c.execute("INSERT INTO lead_history (lead_id, who, field, old_value, new_value, timestamp) VALUES (?,?,?,?,?,?)",
                  (lead_id, "admin", "deleted", "", "", now_iso()))
        conn.commit(); conn.close()
        return True
    except Exception:
        return False

# ----------------------------
# ML: Train / Load / Score (internal only)
# ----------------------------
def train_internal_model():
    df = fetch_leads()
    if df.empty or df["converted"].nunique() < 2:
        return None, "Not enough labeled conversion data to train"
    df2 = df.copy()
    df2["created_at"] = pd.to_datetime(df2["created_at"], errors="coerce")
    df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
    X = pd.get_dummies(df2[["source","stage"]].astype(str))
    X["ad_cost"] = df2["ad_cost"]
    X["estimated_value"] = df2["estimated_value"]
    X["age_days"] = df2["age_days"]
    y = df2["converted"].fillna(0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_FILE)
    acc = model.score(X_test, y_test)
    return model, acc

def load_internal_model():
    if os.path.exists(MODEL_FILE):
        try:
            obj = joblib.load(MODEL_FILE)
            return obj.get("model") if isinstance(obj, dict) else obj
        except Exception:
            try:
                return joblib.load(MODEL_FILE)
            except:
                return None
    return None

def score_leads_and_persist():
    model_bundle = load_internal_model()
    if model_bundle is None:
        return "No model found"
    model = model_bundle if not isinstance(model_bundle, dict) else model_bundle.get("model")
    df = fetch_leads()
    if df.empty:
        return "No leads to score"
    # build X with fallback columns
    df2 = df.copy()
    df2["created_at"] = pd.to_datetime(df2["created_at"], errors="coerce")
    df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
    X = pd.get_dummies(df2[["source","stage"]].astype(str))
    # ensure columns align with model if available
    if hasattr(model, "n_features_in_"):
        # attempt to reconstruct columns: try to load saved columns from saved model_bundle dict
        try:
            saved = joblib.load(MODEL_FILE)
            cols = saved.get("columns") if isinstance(saved, dict) else None
            if cols:
                X = X.reindex(columns=cols, fill_value=0)
        except:
            pass
    X["ad_cost"] = df2["ad_cost"]
    X["estimated_value"] = df2["estimated_value"]
    # predict_proba if available
    try:
        probs = model.predict_proba(X)[:,1]
    except:
        probs = model.predict(X)
        # normalize if necessary
        if probs.max() > 1:
            probs = probs / probs.max()
    # write back to DB
    conn = get_conn(); c = conn.cursor()
    for lead_id, score in zip(df2["lead_id"], probs):
        c.execute("UPDATE leads SET score = ? WHERE lead_id = ?", (float(score), lead_id))
    conn.commit(); conn.close()
    return f"Scored {len(probs)} leads"

# ----------------------------
# SLA / Prioritization helpers
# ----------------------------
def remaining_sla_seconds(sla_entered_at, sla_hours):
    try:
        if not sla_entered_at:
            sla_entered = datetime.utcnow()
        else:
            sla_entered = pd.to_datetime(sla_entered_at).to_pydatetime()
        deadline = sla_entered + timedelta(hours=int(sla_hours or DEFAULT_SLA_HOURS))
        remain = (deadline - datetime.utcnow()).total_seconds()
        return max(remain, 0.0)
    except Exception:
        return float("inf")

def is_overdue(sla_entered_at, sla_hours):
    return remaining_sla_seconds(sla_entered_at, sla_hours) <= 0

def compute_priority(row, weights=None):
    # weights dict optional; else use defaults
    if weights is None:
        weights = {"score":0.6, "value":0.25, "sla":0.15, "value_baseline":5000.0}
    try:
        ml = float(row.get("score") or 0.0)
    except:
        ml = 0.0
    try:
        val = float(row.get("estimated_value") or 0.0)
    except:
        val = 0.0
    value_score = min(1.0, val / max(1.0, float(weights.get("value_baseline",5000.0))))
    sla_h = row.get("sla_hours") or DEFAULT_SLA_HOURS
    rem_h = remaining_sla_seconds(row.get("sla_entered_at"), sla_h) / 3600.0
    sla_score = max(0.0, (72.0 - min(rem_h,72.0)) / 72.0)
    total = ml * weights.get("score",0.6) + value_score * weights.get("value",0.25) + sla_score * weights.get("sla",0.15)
    total = max(0.0, min(1.0, total))
    return total

# ----------------------------
# EXPORT helpers
# ----------------------------
def df_to_excel_bytes(df):
    towrite = io.BytesIO()
    try:
        df.to_excel(towrite, index=False, engine="openpyxl")
    except Exception:
        # fallback to csv bytes in Excel-compatible mime if openpyxl missing
        towrite = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    towrite.seek(0)
    return towrite.read()

# ----------------------------
# UI - CSS
# ----------------------------
st.set_page_config(page_title="TITAN — Pipeline (Option A)", layout="wide", initial_sidebar_state="expanded")
APP_CSS = """
<style>
:root{--bg:#ffffff;--card:#0b1220;--muted:#6b7280;--white:#ffffff;}
body, .stApp { background: var(--bg); color: #0b1220; font-family: 'Comfortaa', "Helvetica", Arial, sans-serif; }
.kpi-card { background: #000; color: #fff; border-radius:12px; padding:12px; min-height:80px; }
.kpi-title { color:#fff; font-weight:700; font-size:13px; }
.kpi-value { font-weight:900; font-size:22px; margin-top:6px; }
.priority-card { background:#000; color:#fff; padding:12px; border-radius:12px; min-width:220px; }
.small-muted { color:#6b7280; font-size:12px; }
.btn-small { padding:8px 12px; border-radius:8px; }
.top-right-controls { text-align:right; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# ----------------------------
# TOP bar: Title left, Date + Bell right
# ----------------------------
top_l, top_s, top_r = st.columns([3,1,2])
with top_l:
    st.markdown("<h2 style='margin:6px 0'>TITAN — Lead Pipeline (Option A)</h2>", unsafe_allow_html=True)

# Prepare session dates
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=29)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

with top_r:
    sd = st.date_input("Start date", value=st.session_state.start_date, key="ui_start_date")
    ed = st.date_input("End date", value=st.session_state.end_date, key="ui_end_date")
    st.session_state.start_date = sd
    st.session_state.end_date = ed

# Notification bell & dropdown (simple)
def notification_ui():
    notifs = fetch_notifications(limit=50)
    unread_count = sum(1 for n in notifs if not n["read"])
    # bell
    st.markdown(f"<div style='text-align:right;font-size:16px'>🔔 <span style='background:#ef4444;color:white;padding:4px 8px;border-radius:12px'>{unread_count}</span></div>", unsafe_allow_html=True)
    # a button to open notifications area in sidebar (or expand)
    if st.button("Show notifications"):
        with st.expander(f"Notifications ({len(notifs)})", expanded=True):
            for n in notifs:
                col1, col2 = st.columns([8,1])
                with col1:
                    st.write(f"**{n['title']}** — {n['message']}")
                    st.caption(n["created_at"])
                with col2:
                    if not n["read"]:
                        if st.button("Mark read", key=f"mark_{n['id']}"):
                            mark_notification_read(n["id"])
                            st.experimental_rerun()
                    else:
                        st.write("✅")

with top_r:
    notification_ui()

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigate")
nav = st.sidebar.radio("Go to", ["Pipeline Board", "Lead Capture", "Analytics & SLA", "CPA & ROI", "Reports & AI", "Exports / Imports", "Settings", "ML (internal)"])

# ----------------------------
# Pipeline Board Page
# ----------------------------
if nav == "Pipeline Board":
    st.header("TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("<em>High-level pipeline performance at a glance. Use filters and cards to drill into details.</em>", unsafe_allow_html=True)

    s = fetch_leads(start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    df = s.copy()

    if df.empty:
        st.info("No leads yet. Add leads on the Lead Capture page.")
    else:
        # show KPIs (7 cards)
        total_leads = len(df)
        sla_success_count = int(df[df["contacted"] == 1].shape[0])
        sla_success_pct = (sla_success_count / total_leads * 100) if total_leads else 0.0
        qualified_leads = int(df[df["qualified"] == 1].shape[0])
        qualification_pct = (qualified_leads / total_leads * 100) if total_leads else 0.0
        awarded_count = int(df[df["stage"] == "Won"].shape[0])
        lost_count = int(df[df["stage"] == "Lost"].shape[0])
        closed = awarded_count + lost_count
        conversion_rate = (awarded_count / closed * 100) if closed else 0.0
        inspection_scheduled_count = int(df[df["inspection_scheduled"] == 1].shape[0])
        inspection_pct = (inspection_scheduled_count / qualified_leads * 100) if qualified_leads else 0.0
        estimate_sent_count = int(df[df["estimate_submitted"] == 1].shape[0])
        pipeline_job_value = float(df["estimated_value"].sum())
        active_leads = total_leads - (awarded_count + lost_count)

        KPI_ITEMS = [
            ("Active Leads", f"{active_leads}", "#2563eb", "Leads currently in pipeline"),
            ("SLA Success", f"{sla_success_pct:.1f}%", "#0ea5a4", "Leads contacted within SLA"),
            ("Qualification Rate", f"{qualification_pct:.1f}%", "#a855f7", "Leads marked qualified"),
            ("Conversion Rate", f"{conversion_rate:.1f}%", "#f97316", "Won / Closed"),
            ("Inspections Booked", f"{inspection_pct:.1f}%", "#ef4444", "Qualified → Scheduled"),
            ("Estimates Sent", f"{estimate_sent_count}", "#6d28d9", "Estimates submitted"),
            ("Pipeline Job Value", f"{format_money(pipeline_job_value)}", "#22c55e", "Total pipeline job value")
        ]

        # Render KPI cards: 4 top, 3 bottom with spacing
        top_cols = st.columns(4, gap="large")
        for (title, value, color, note), c in zip(KPI_ITEMS[:4], top_cols):
            pct = 50
            c.markdown(f"""
                <div class="kpi-card" style="background:#000; padding:12px; border-radius:10px;">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value" style="color:{color};">{value}</div>
                    <div style="height:8px; background:#e6e6e6; border-radius:6px; margin-top:10px;">
                      <div style="height:100%; width:{pct}%; background:{color}; border-radius:6px;"></div>
                    </div>
                    <div class="small-muted" style="margin-top:8px;">{note}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        bottom_cols = st.columns(3, gap="large")
        for (title, value, color, note), c in zip(KPI_ITEMS[4:], bottom_cols):
            pct = 40
            c.markdown(f"""
                <div class="kpi-card" style="background:#000; padding:12px; border-radius:10px;">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value" style="color:{color};">{value}</div>
                    <div style="height:8px; background:#e6e6e6; border-radius:6px; margin-top:10px;">
                      <div style="height:100%; width:{pct}%; background:{color}; border-radius:6px;"></div>
                    </div>
                    <div class="small-muted" style="margin-top:8px;">{note}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Pipeline Stages (DONUT in Analytics; for pipeline board keep stage counts as a horizontal bar or mini chart)
        st.markdown("### Lead Pipeline Stages")
        st.markdown("<em>Distribution of leads across pipeline stages. Use this to spot stage drop-offs quickly.</em>", unsafe_allow_html=True)
        stage_counts = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
        pie_df = pd.DataFrame({"stage": stage_counts.index, "count": stage_counts.values})
        # use pie/donut here as requested (but in earlier instructions pipeline donut was removed from pipeline; user wants donut in analytics — we show a compact bar here)
        fig_stage = px.bar(pie_df, x="stage", y="count", title="Leads per Stage")
        st.plotly_chart(fig_stage, use_container_width=True)

        st.markdown("---")

        # TOP 5 priority leads (black cards)
        st.markdown("### TOP 5 PRIORITY LEADS")
        st.markdown("<em>Highest urgency leads by priority score (0–1). Address these first.</em>", unsafe_allow_html=True)

        df["priority_score"] = df.apply(lambda r: compute_priority(r), axis=1)
        df["hours_left"] = df.apply(lambda r: int(remaining_sla_seconds(r.get("sla_entered_at"), r.get("sla_hours"))/3600.0) if r.get("sla_hours") not in (None,0) else 9999, axis=1)
        pr_df = df.sort_values("priority_score", ascending=False).head(5)

        if pr_df.empty:
            st.info("No priority leads to display.")
        else:
            pr_cols = st.columns(len(pr_df))
            for c, (_, r) in zip(pr_cols, pr_df.iterrows()):
                score = r["priority_score"]
                if score >= 0.7:
                    priority_color = "#ef4444"; label = "🔴 CRITICAL"
                elif score >= 0.45:
                    priority_color = "#f97316"; label = "🟠 HIGH"
                else:
                    priority_color = "#22c55e"; label = "🟢 NORMAL"
                hours_left = r["hours_left"]
                money = format_money(r["estimated_value"])
                sla_html = f"<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>" if is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) else f"<div style='color:#ef4444;font-weight:700;'>⏳ {hours_left}h left</div>"
                c.markdown(f"""
                <div class="priority-card">
                  <div style="font-weight:800; font-size:14px; color:white;">#{r['lead_id']} — {r.get('contact_name') or 'No name'}</div>
                  <div style="margin-top:6px; color:{priority_color}; font-weight:700;">{label}</div>
                  <div style="margin-top:8px; color:#fff; font-size:13px;">{r.get('damage_type','Unknown').title()} | {money}</div>
                  <div style="margin-top:8px;">{sla_html}</div>
                  <div style="margin-top:6px; color:white;">Priority score: <strong>{score:.2f}</strong></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # All Leads table with expand to edit
        st.markdown("### 📋 All Leads (expand a card to edit / change status)")
        st.markdown("<em>Expand a lead to edit details, change status, upload invoice when awarded, and create estimates.</em>", unsafe_allow_html=True)
        search_q = st.text_input("Search leads (id, contact, address, notes)", value="", key="search_leads")
        view_df = df.copy()
        if search_q:
            ql = search_q.lower()
            view_df = view_df[view_df.apply(lambda r: ql in str(r.get("lead_id","")).lower() or ql in str(r.get("contact_name","")).lower() or ql in str(r.get("property_address","")).lower() or ql in str(r.get("notes","")).lower(), axis=1)]
        for _, lead in view_df.sort_values("created_at", ascending=False).iterrows():
            est_val_display = format_money(lead.get("estimated_value", 0))
            card_title = f"#{lead['lead_id']} — {lead.get('contact_name') or 'No name'} — {lead.get('damage_type') or 'Unknown'} — {est_val_display}"
            with st.expander(card_title, expanded=False):
                colA, colB = st.columns([3, 1])
                with colA:
                    st.write(f"**Source:** {lead.get('source') or '—'}   |   **Assigned:** {lead.get('assigned_to') or '—'}")
                    st.write(f"**Address:** {lead.get('property_address') or '—'}")
                    st.write(f"**Notes:** {lead.get('notes') or '—'}")
                    st.write(f"**Created:** {pd.to_datetime(lead.get('created_at')).strftime('%Y-%m-%d %H:%M') if lead.get('created_at') else '—'}")
                with colB:
                    entered = lead.get("sla_entered_at") or lead.get("created_at")
                    try:
                        entered_dt = pd.to_datetime(entered)
                    except:
                        entered_dt = pd.to_datetime(now_iso())
                    deadline = entered_dt + pd.Timedelta(hours=int(lead.get("sla_hours") or DEFAULT_SLA_HOURS))
                    remaining = deadline - pd.Timestamp.utcnow()
                    if remaining.total_seconds() <= 0 and lead.get("stage") not in ("Won","Lost"):
                        sla_status_html = "<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>"
                    else:
                        try:
                            hours = int(remaining.total_seconds() // 3600)
                            mins = int((remaining.total_seconds() % 3600) // 60)
                            sla_status_html = f"<div style='color:#111827;font-weight:700;'>⏳ {hours}h {mins}m</div>"
                        except:
                            sla_status_html = "<div style='color:#111827;font-weight:700;'>—</div>"
                    st.markdown(f"<div style='text-align:right;'><div style='display:inline-block; padding:6px 12px; border-radius:20px; background: #111; color:#fff; font-weight:700;'>{lead.get('stage') or '—'}</div><div style='margin-top:8px;'>{sla_status_html}</div></div>", unsafe_allow_html=True)

                st.markdown("---")
                # Quick contact / action buttons
                qc1, qc2, qc3, qc4 = st.columns([1,1,1,3])
                phone = (lead.get("contact_phone") or "").strip()
                email = (lead.get("contact_email") or "").strip()
                if phone:
                    with qc1:
                        st.markdown(f"<a href='tel:{phone}'><button class='btn-small' style='background:#2563eb;color:white;border-radius:8px;padding:8px 12px;'>📞 Call</button></a>", unsafe_allow_html=True)
                    with qc2:
                        wa_number = phone.lstrip("+").replace(" ", "").replace("-", "")
                        wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20following%20up%20on%20your%20restoration%20request."
                        st.markdown(f"<a href='{wa_link}' target='_blank'><button class='btn-small' style='background:#22c55e;color:#000;border-radius:8px;padding:8px 12px;'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                else:
                    qc1.write(" "); qc2.write(" ")
                if email:
                    with qc3:
                        st.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button class='btn-small' style='background:transparent;color:#000;border:1px solid #e5e7eb;border-radius:8px;padding:8px 12px;'>✉️ Email</button></a>", unsafe_allow_html=True)
                else:
                    qc3.write(" ")
                qc4.write("")

                st.markdown("---")

                # Lead update form
                with st.form(f"update_lead_{lead['lead_id']}"):
                    st.markdown("#### Update Lead")
                    u1,u2 = st.columns(2)
                    with u1:
                        new_stage = st.selectbox("Status", PIPELINE_STAGES, index=PIPELINE_STAGES.index(lead.get("stage")) if lead.get("stage") in PIPELINE_STAGES else 0, key=f"status_{lead['lead_id']}")
                        new_assigned = st.text_input("Assigned to", value=lead.get("assigned_to") or "", key=f"assign_{lead['lead_id']}")
                        new_contacted = st.checkbox("Contacted", value=bool(lead.get("contacted")), key=f"contacted_{lead['lead_id']}")
                    with u2:
                        insp_sched = st.checkbox("Inspection Scheduled", value=bool(lead.get("inspection_scheduled")), key=f"insp_sched_{lead['lead_id']}")
                        insp_comp = st.checkbox("Inspection Completed", value=bool(lead.get("inspection_completed")), key=f"insp_comp_{lead['lead_id']}")
                        est_sub = st.checkbox("Estimate Submitted", value=bool(lead.get("estimate_submitted")), key=f"est_sub_{lead['lead_id']}")
                    new_notes = st.text_area("Notes", value=lead.get("notes") or "", key=f"notes_{lead['lead_id']}")
                    new_est_val = st.number_input("Job Value Estimate (USD)", value=float(lead.get("estimated_value") or 0.0), min_value=0.0, step=100.0, key=f"estval_{lead['lead_id']}")
                    new_ad_cost = st.number_input("Cost to acquire lead (USD)", value=float(lead.get("ad_cost") or 0.0), min_value=0.0, step=1.0, key=f"adcost_{lead['lead_id']}")

                    award_invoice = None
                    award_comment = None
                    lost_comment = None
                    if new_stage == "Won":
                        st.markdown("**Award details**")
                        award_comment = st.text_area("Award comment", key=f"award_comment_{lead['lead_id']}")
                        award_invoice = st.file_uploader("Upload Invoice (optional)", type=["pdf","jpg","png","xlsx","csv"], key=f"award_inv_{lead['lead_id']}")
                    elif new_stage == "Lost":
                        st.markdown("**Lost details**")
                        lost_comment = st.text_area("Lost comment", key=f"lost_comment_{lead['lead_id']}")

                    if st.form_submit_button("💾 Update Lead"):
                        try:
                            payload = {
                                "lead_id": lead.get("lead_id"),
                                "stage": new_stage,
                                "assigned_to": new_assigned,
                                "contacted": 1 if new_contacted else 0,
                                "inspection_scheduled": 1 if insp_sched else 0,
                                "inspection_completed": 1 if insp_comp else 0,
                                "estimate_submitted": 1 if est_sub else 0,
                                "notes": new_notes,
                                "estimated_value": float(new_est_val or 0.0),
                                "ad_cost": float(new_ad_cost or 0.0),
                                "sla_entered_at": lead.get("sla_entered_at") or lead.get("created_at")
                            }
                            # set awarded/lost extras
                            if new_stage == "Won":
                                payload["awarded_date"] = now_iso()
                                payload["awarded_invoice"] = None
                                if award_invoice:
                                    # save file locally
                                    try:
                                        fbytes = award_invoice.getbuffer()
                                        fname = f"award_{lead.get('lead_id')}_{int(datetime.utcnow().timestamp())}_{award_invoice.name}"
                                        pth = os.path.join(os.getcwd(), fname)
                                        with open(pth, "wb") as f:
                                            f.write(fbytes)
                                        payload["awarded_invoice"] = pth
                                    except Exception:
                                        pass
                            if new_stage == "Lost":
                                payload["lost_date"] = now_iso()
                                payload["lost_comment"] = lost_comment

                            # Option A auto-progression: when flags are changed above we set stage accordingly, but here manual stage chosen overrides
                            # Because payload contains explicit 'stage', save_lead will respect manual override
                            save_lead(payload, who=st.session_state.get("profile", {}).get("name", "admin"))
                            st.success(f"Lead {lead.get('lead_id')} updated.")
                            # create notification for awarded or lost
                            if new_stage == "Won":
                                add_notification("Lead Awarded", f"Lead {lead.get('lead_id')} has been marked Won.")
                            if new_stage == "Lost":
                                add_notification("Lead Lost", f"Lead {lead.get('lead_id')} has been marked Lost.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error("Failed to update lead: " + str(e))
                            st.write(traceback.format_exc())

# ----------------------------
# Lead Capture page
# ----------------------------
elif nav == "Lead Capture":
    st.header("📇 Lead Capture")
    st.markdown("<em>Create or update a lead. SLA Response time must be greater than 0 hours.</em>", unsafe_allow_html=True)

    with st.form("lead_form_new", clear_on_submit=True):
        lead_id = st.text_input("Lead ID (optional)")
        source = st.selectbox("Lead Source", ["Google Ads", "Organic Search", "Referral", "Phone", "Insurance", "Facebook", "Instagram", "LinkedIn", "Other"])
        source_details = st.text_input("Source details (UTM / notes)")
        contact_name = st.text_input("Contact name")
        contact_phone = st.text_input("Contact phone")
        contact_email = st.text_input("Contact email")
        property_address = st.text_input("Property address")
        damage_type = st.selectbox("Damage type", ["water", "fire", "mold", "contents", "reconstruction", "other"])
        assigned_to = st.text_input("Assigned to")
        estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
        ad_cost = st.number_input("Cost to acquire lead (USD)", min_value=0.0, value=0.0, step=1.0)
        sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=DEFAULT_SLA_HOURS, step=1, help="SLA Response time must be greater than 0 hours.")
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Create / Update Lead")
        if submitted:
            if sla_hours <= 0:
                st.error("SLA must be greater than 0 hours.")
            else:
                payload = {
                    "lead_id": lead_id.strip() if lead_id else None,
                    "created_at": now_iso(),
                    "source": source,
                    "source_details": source_details,
                    "contact_name": contact_name,
                    "contact_phone": contact_phone,
                    "contact_email": contact_email,
                    "property_address": property_address,
                    "damage_type": damage_type,
                    "assigned_to": assigned_to,
                    "notes": notes,
                    "estimated_value": float(estimated_value or 0.0),
                    "ad_cost": float(ad_cost or 0.0),
                    "sla_hours": int(sla_hours),
                    "sla_entered_at": now_iso(),
                    "stage": "New",
                    "converted": 0,
                    "score": 0.0
                }
                newid = save_lead(payload, who=st.session_state.get("profile", {}).get("name", "admin"))
                st.success(f"Lead created (ID: {newid})")
                st.experimental_rerun()

# ----------------------------
# Analytics & SLA page
# ----------------------------
elif nav == "Analytics & SLA":
    st.header("📈 Analytics — Cost vs Conversions & SLA")
    st.markdown("<em>Select date range (top-right) to filter. CPA and ROI pages show details.</em>", unsafe_allow_html=True)

    df = fetch_leads(start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    if df.empty:
        st.info("No leads in selected range.")
    else:
        # Pipeline stages donut
        st.markdown("### Lead Pipeline Stages (Donut)")
        stage_counts = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
        pie_df = pd.DataFrame({"stage": stage_counts.index, "count": stage_counts.values})
        fig = px.pie(pie_df, names="stage", values="count", hole=0.45, title="Pipeline Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Cost vs Conversions by source
        st.markdown("### Cost vs Conversions")
        df["won"] = df["stage"].apply(lambda s: 1 if s == "Won" else 0)
        agg = df.groupby("source").agg(total_spend=("ad_cost","sum"), conversions=("won","sum")).reset_index()
        if not agg.empty:
            fig2 = px.bar(agg, x="source", y=["total_spend","conversions"], barmode="group", title="Total Marketing Spend vs Conversions by Source")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No spend data available.")

        # SLA overdue trend (last 30 days)
        st.markdown("### SLA Overdue Trend (30 days)")
        today = date.today()
        dates = [today - timedelta(days=i) for i in range(29, -1, -1)]
        rows = []
        for d in dates:
            start_dt = pd.to_datetime(d)
            end_dt = pd.to_datetime(d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            day_df = df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)]
            overdue_count = int(day_df.apply(lambda r: 1 if (is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost")) else 0, axis=1).sum())
            rows.append({"date": d, "overdue": overdue_count})
        ts_df = pd.DataFrame(rows)
        fig3 = px.line(ts_df, x="date", y="overdue", markers=True, title="SLA Overdue Count (30d)")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### Current Overdue Leads")
        overdue_df = df[df.apply(lambda r: is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost"), axis=1)]
        if overdue_df.empty:
            st.success("No SLA overdue leads.")
        else:
            st.dataframe(overdue_df[["lead_id","contact_name","stage","estimated_value","ad_cost","sla_hours"]])

# ----------------------------
# CPA & ROI page
# ----------------------------
elif nav == "CPA & ROI":
    st.header("💰 CPA & ROI")
    df = fetch_leads(start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    if df.empty:
        st.info("No data to compute CPA/ROI.")
    else:
        awarded_df = df[df["stage"]=="Won"].copy()
        total_spend = df["ad_cost"].sum()
        conversions = len(awarded_df)
        cpa = (total_spend / conversions) if conversions else None
        pipeline_value = df["estimated_value"].sum()
        roi = ((pipeline_value - total_spend) / total_spend * 100) if total_spend else None

        # display metrics with colored fonts (no extra stylized card)
        cols = st.columns(3)
        cols[0].markdown(f"<div style='font-size:13px;color:white;background:#111;padding:12px;border-radius:10px;'>✅ <b>Conversions (Won)</b><div style='font-size:20px;margin-top:8px;color:#2563eb;font-weight:800;'>{conversions}</div></div>", unsafe_allow_html=True)
        cols[1].markdown(f"<div style='font-size:13px;color:white;background:#111;padding:12px;border-radius:10px;'>🎯 <b>CPA</b><div style='font-size:20px;margin-top:8px;color:#f97316;font-weight:800;'>{format_money(cpa) if cpa else '$0.00'}</div></div>", unsafe_allow_html=True)
        cols[2].markdown(f"<div style='font-size:13px;color:white;background:#111;padding:12px;border-radius:10px;'>📈 <b>ROI</b><div style='font-size:18px;margin-top:8px;color:#22c55e;font-weight:800;'>{f'{roi:.1f}% ({format_money(pipeline_value)})' if roi is not None else '—'}</div></div>", unsafe_allow_html=True)

        # Chart: Total Marketing Spend vs Number of conversions over time (monthly)
        st.markdown("---")
        st.markdown("### Spend vs Conversions Over Time")
        if not df.empty:
            df["created_month"] = pd.to_datetime(df["created_at"]).dt.to_period("M").astype(str)
            monthly = df.groupby("created_month").agg(spend=("ad_cost","sum"), conversions=("stage", lambda x: (x=="Won").sum())).reset_index()
            if not monthly.empty:
                fig = px.line(monthly, x="created_month", y=["spend","conversions"], markers=True, title="Total Marketing Spend vs Conversions (monthly)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No monthly data.")
        else:
            st.info("No leads to chart.")

# ----------------------------
# Reports & AI Observations
# ----------------------------
elif nav == "Reports & AI":
    st.header("🧠 AI Observation & Recommendation")
    df = fetch_leads()
    if df.empty:
        st.info("No leads to analyze.")
    else:
        notes = []
        # bottleneck stage
        bottleneck = df["stage"].value_counts().idxmax()
        notes.append(f"Most leads currently in '{bottleneck}' stage — investigate stage handoffs.")
        # high spend channels
        high_spend = df.groupby("source")["ad_cost"].sum().sort_values(ascending=False).head(3)
        notes.append(f"Top spend channels: {', '.join(high_spend.index.astype(str).tolist())}")
        # SLA failures
        overdue = df[df.apply(lambda r: is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost"), axis=1)]
        notes.append(f"SLA overdue leads: {len(overdue)}")
        # ROI
        spend = df["ad_cost"].sum()
        rev = df["estimated_value"].sum()
        roi = ((rev - spend)/spend*100) if spend else None
        if roi is not None and roi < 20:
            notes.append("ROI below 20% — consider optimizing campaigns and qualification.")
        else:
            notes.append("ROI healthy or no spend data.")

        st.subheader("Observations")
        for n in notes:
            st.write("- " + n)

        st.markdown("---")
        st.subheader("Executive Summary")
        summary = "\n".join(notes)
        st.text_area("Executive Summary", value=summary, height=150)
        st.download_button("Download Summary", summary.encode("utf-8"), file_name="titan_report_summary.txt", mime="text/plain")

# ----------------------------
# Exports / Imports
# ----------------------------
elif nav == "Exports / Imports":
    st.header("📤 Exports & Imports")
    df = fetch_leads(start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    if df.empty:
        st.info("No leads to export for selected range.")
    else:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads (CSV)", csv_bytes, file_name="leads_export.csv", mime="text/csv")
        # Excel fallback
        try:
            try:
                import openpyxl  # ensure available
                excel_bytes = df_to_file_bytes_excel(df)
                st.download_button("Download leads (Excel)", excel_bytes, file_name="leads_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:
                # fallback to csv link
                st.info("Excel export unavailable (openpyxl). CSV is provided.")
        except Exception:
            st.info("Excel export unavailable. CSV is provided.")

    st.markdown("---")
    uploaded = st.file_uploader("Import leads CSV (must contain lead_id column)", type=["csv"])
    if uploaded:
        try:
            imp = pd.read_csv(uploaded)
            if "lead_id" not in imp.columns:
                st.error("CSV must contain 'lead_id' column.")
            else:
                count = 0
                for _, r in imp.iterrows():
                    payload = {
                        "lead_id": str(r.get("lead_id")),
                        "created_at": pd.to_datetime(r.get("created_at")).isoformat() if not pd.isna(r.get("created_at")) else now_iso(),
                        "source": r.get("source"),
                        "source_details": r.get("source_details"),
                        "contact_name": r.get("contact_name"),
                        "contact_phone": r.get("contact_phone"),
                        "contact_email": r.get("contact_email"),
                        "property_address": r.get("property_address"),
                        "damage_type": r.get("damage_type"),
                        "assigned_to": r.get("assigned_to"),
                        "notes": r.get("notes"),
                        "estimated_value": safe_float(r.get("estimated_value")),
                        "stage": r.get("stage") or "New",
                        "sla_hours": int(r.get("sla_hours") or DEFAULT_SLA_HOURS),
                        "sla_entered_at": r.get("sla_entered_at") or now_iso(),
                        "converted": int(r.get("converted") or 0),
                        "ad_cost": safe_float(r.get("ad_cost")),
                        "score": safe_float(r.get("score"))
                    }
                    save_lead(payload, who="import")
                    count += 1
                st.success(f"Imported {count} leads.")
        except Exception as e:
            st.error("Import failed: " + str(e))

# ----------------------------
# Settings
# ----------------------------
elif nav == "Settings":
    st.header("⚙️ Settings & Users (internal)")
    st.markdown("<em>Set your display name and role (stored in session for audit)</em>", unsafe_allow_html=True)
    if "profile" not in st.session_state:
        st.session_state.profile = {"name": "", "role": "Admin"}
    name = st.text_input("Your name", value=st.session_state.profile.get("name",""))
    role = st.selectbox("Role", ["Admin","Estimator","Manager","Viewer"], index=["Admin","Estimator","Manager","Viewer"].index(st.session_state.profile.get("role","Admin")) if st.session_state.profile.get("role") in ["Admin","Estimator","Manager","Viewer"] else 0)
    if st.button("Save profile"):
        st.session_state.profile["name"] = name
        st.session_state.profile["role"] = role
        st.success("Profile saved to session.")

    st.markdown("---")
    st.subheader("Audit Trail (last 100)")
    conn = get_conn(); hist = pd.read_sql_query("SELECT * FROM lead_history ORDER BY timestamp DESC LIMIT 100", conn)
    conn.close()
    if hist.empty:
        st.info("No history yet.")
    else:
        st.dataframe(hist)

# ----------------------------
# ML internal
# ----------------------------
elif nav == "ML (internal)":
    st.header("🧠 Internal ML — Lead Scoring (no user tuning)")
    st.markdown("<em>Train on historic converted vs not converted leads. Model saved internally. Use 'Score leads' to persist predictions.</em>", unsafe_allow_html=True)
    if st.button("Train internal model"):
        model, msg = train_internal_model()
        if model is None:
            st.error(f"Training failed or not possible: {msg}")
        else:
            st.success(f"Model trained — approx accuracy: {msg:.3f}" if isinstance(msg, float) else f"{msg}")
    if st.button("Score leads & persist"):
        res = score_leads_and_persist()
        if res:
            st.success(str(res))
        else:
            st.error("Scoring failed or no model.")

# ----------------------------
# Footer / safety
# ----------------------------
else:
    st.info("Select a page from the left navigation to begin.")

st.markdown("---")
st.markdown("<div style='font-size:12px;color:#666'>TITAN — single-file backend. Use with care. For production, move DB to a durable storage.</div>", unsafe_allow_html=True)
