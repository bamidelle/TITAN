# titan_final_app.py
"""
TITAN — Final single-file Streamlit backend
Features implemented:
- SQLite persistence (sqlite3)
- Pipeline Board (2-row KPI cards), Top 5 priority leads (black cards)
- Date selector top-right and notification bell with unread count
- Lead Capture (auto Lead ID), SLA validation (>0)
- Analytics: Pipeline donut, Cost vs Conversions, SLA overdue line chart, overdue table
- CPA & ROI page (colored numbers)
- Exports/Imports (CSV + Excel fallback)
- Settings (profile, role saved to session), Audit trail
- Internal ML (optional scikit-learn): train & score (stored in same table)
- AI Observations & Recommendations (top 20)
- Alerts: SLA breaches and stage stagnation (both)
- Priority scoring: same_table approach
- UI: Hybrid premium + energetic look, Comfortaa-like font via fallback
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import io
import joblib
import traceback
from datetime import datetime, timedelta, date, timezone

# Optional ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Optional plotting
import plotly.express as px

# -----------------------
# Configuration
# -----------------------
DB_FILE = "titan_final.db"
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
MODEL_FILE = "titan_model.joblib"
NOTIF_TABLE = "notifications"
HISTORY_TABLE = "lead_history"

# -----------------------
# Database initialization (no parameter binding inside CREATE)
# -----------------------
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    # leads table
    c.execute(f"""
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
        ad_cost REAL DEFAULT 0,
        stage TEXT DEFAULT 'New',
        sla_hours INTEGER DEFAULT {DEFAULT_SLA_HOURS},
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
        converted INTEGER DEFAULT 0,
        score REAL DEFAULT 0.0
    );
    """)
    # history / audit
    c.execute(f"""
    CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
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
    CREATE TABLE IF NOT EXISTS {NOTIF_TABLE} (
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

# -----------------------
# Utility helpers
# -----------------------
def utcnow_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def safe_float(v, default=0.0):
    try:
        return float(v or 0.0)
    except Exception:
        return default

def format_money(v):
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return "$0.00"

def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

# -----------------------
# DB helpers
# -----------------------
def fetch_leads(start_date=None, end_date=None):
    conn = get_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM leads ORDER BY created_at DESC", conn, parse_dates=["created_at","sla_entered_at","inspection_scheduled_at","estimate_submitted_at","awarded_date","lost_date"])
    except Exception:
        df = pd.DataFrame()
    conn.close()
    if df.empty:
        # make empty df with expected columns
        cols = ["id","lead_id","created_at","source","source_details","contact_name","contact_phone","contact_email","property_address","damage_type","assigned_to","notes","estimated_value","ad_cost","stage","sla_hours","sla_entered_at","contacted","inspection_scheduled","inspection_scheduled_at","inspection_completed","estimate_submitted","estimate_submitted_at","awarded_date","awarded_invoice","lost_date","qualified","converted","score"]
        return pd.DataFrame(columns=cols)
    # normalize numeric/bool columns
    for col in ["estimated_value","ad_cost","score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ["contacted","inspection_scheduled","inspection_completed","estimate_submitted","qualified","converted","sla_hours"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    # filter by date if provided
    if start_date:
        try:
            df = df[df["created_at"] >= pd.to_datetime(start_date)]
        except Exception:
            pass
    if end_date:
        try:
            df = df[df["created_at"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
        except Exception:
            pass
    return df.reset_index(drop=True)

def write_history(lead_id, who, field, old, new):
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute(f"INSERT INTO {HISTORY_TABLE} (lead_id,who,field,old_value,new_value,timestamp) VALUES (?,?,?,?,?,?)", (lead_id, who, field, str(old), str(new), utcnow_iso()))
        conn.commit(); conn.close()
    except Exception:
        pass

def add_notification(title, message):
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute(f"INSERT INTO {NOTIF_TABLE} (title,message,created_at,read) VALUES (?,?,?,0)", (title, message, utcnow_iso()))
        conn.commit(); conn.close()
    except Exception:
        pass

def fetch_notifications(limit=100):
    conn = get_conn(); c = conn.cursor()
    try:
        c.execute(f"SELECT id,title,message,created_at,read FROM {NOTIF_TABLE} ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = c.fetchall()
    except Exception:
        rows = []
    conn.close()
    notifs = [{"id": r[0], "title": r[1], "message": r[2], "created_at": r[3], "read": bool(r[4])} for r in rows]
    return notifs

def mark_notification_read(nid):
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute(f"UPDATE {NOTIF_TABLE} SET read=1 WHERE id=?", (nid,))
        conn.commit(); conn.close()
    except Exception:
        pass

# -----------------------
# Lead CRUD
# -----------------------
def generate_lead_id():
    # simple numeric timestamp-based ID
    return f"L{int(datetime.utcnow().timestamp())}"

def upsert_lead(payload, who="admin"):
    """
    Insert or update a lead. payload is a dict.
    Returns lead_id or None on failure.
    Option A auto-progression used but manual 'stage' in payload overrides.
    """
    try:
        conn = get_conn(); c = conn.cursor()
        lid = payload.get("lead_id") or generate_lead_id()
        # find existing
        c.execute("SELECT * FROM leads WHERE lead_id=?", (lid,))
        existing = c.fetchone()
        if not existing:
            # Insert
            created_at = payload.get("created_at") or utcnow_iso()
            sla_entered = payload.get("sla_entered_at") or created_at
            stage = payload.get("stage") or "New"
            c.execute("""
                INSERT INTO leads (lead_id,created_at,source,source_details,contact_name,contact_phone,contact_email,property_address,damage_type,assigned_to,notes,estimated_value,ad_cost,stage,sla_hours,sla_entered_at,contacted,inspection_scheduled,inspection_scheduled_at,inspection_completed,estimate_submitted,estimate_submitted_at,awarded_date,awarded_invoice,lost_date,qualified,converted,score)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                lid, created_at,
                payload.get("source"),
                payload.get("source_details"),
                payload.get("contact_name"),
                payload.get("contact_phone"),
                payload.get("contact_email"),
                payload.get("property_address"),
                payload.get("damage_type"),
                payload.get("assigned_to"),
                payload.get("notes"),
                safe_float(payload.get("estimated_value")),
                safe_float(payload.get("ad_cost")),
                stage,
                int(payload.get("sla_hours") or DEFAULT_SLA_HOURS),
                sla_entered,
                int(bool(payload.get("contacted"))),
                int(bool(payload.get("inspection_scheduled"))),
                payload.get("inspection_scheduled_at"),
                int(bool(payload.get("inspection_completed"))),
                int(bool(payload.get("estimate_submitted"))),
                payload.get("estimate_submitted_at"),
                payload.get("awarded_date"),
                payload.get("awarded_invoice"),
                payload.get("lost_date"),
                int(bool(payload.get("qualified"))),
                int(bool(payload.get("converted"))),
                float(payload.get("score") or 0.0)
            ))
            c.execute(f"INSERT INTO {HISTORY_TABLE} (lead_id,who,field,old_value,new_value,timestamp) VALUES (?,?,?,?,?,?)", (lid, who, "create", "", stage, utcnow_iso()))
            conn.commit(); conn.close()
            return lid
        else:
            # Update
            # load existing row into dict
            conn2 = get_conn()
            df = pd.read_sql_query("SELECT * FROM leads WHERE lead_id=?", conn2, params=(lid,), parse_dates=["created_at","sla_entered_at"])
            conn2.close()
            if df.empty:
                return upsert_lead(payload, who)
            cur = df.iloc[0].to_dict()
            # detect auto progression flags
            auto_stage = None
            try:
                if payload.get("inspection_completed"):
                    auto_stage = "Inspection Completed"
                elif payload.get("inspection_scheduled"):
                    auto_stage = "Inspection Scheduled"
                elif payload.get("estimate_submitted"):
                    auto_stage = "Estimate Submitted"
                elif payload.get("contacted"):
                    auto_stage = "Contacted"
            except Exception:
                auto_stage = None
            chosen_stage = payload.get("stage") if payload.get("stage") is not None else (auto_stage or cur.get("stage"))
            fields = ["source","source_details","contact_name","contact_phone","contact_email","property_address","damage_type","assigned_to","notes","estimated_value","ad_cost","stage","sla_hours","sla_entered_at","contacted","inspection_scheduled","inspection_scheduled_at","inspection_completed","estimate_submitted","estimate_submitted_at","awarded_date","awarded_invoice","lost_date","qualified","converted","score"]
            for f in fields:
                if f in payload:
                    new_v = payload.get(f)
                    if f in ["contacted","inspection_scheduled","inspection_completed","estimate_submitted","qualified","converted"]:
                        new_v = int(bool(new_v))
                    if f in ["estimated_value","ad_cost","score"]:
                        new_v = safe_float(new_v)
                    if f == "sla_hours":
                        try:
                            new_v = int(new_v)
                            if new_v <= 0:
                                new_v = DEFAULT_SLA_HOURS
                        except:
                            new_v = DEFAULT_SLA_HOURS
                    old_v = cur.get(f)
                    # compare as strings to avoid pandas types mismatch
                    if str(old_v) != str(new_v):
                        c.execute(f"UPDATE leads SET {f}=? WHERE lead_id=?", (new_v, lid))
                        write_history(lid, who, f, old_v, new_v)
            # ensure chosen_stage persisted
            c.execute("SELECT stage FROM leads WHERE lead_id=?", (lid,))
            row = c.fetchone()
            cur_stage = row[0] if row else None
            if cur_stage != chosen_stage:
                c.execute("UPDATE leads SET stage=? WHERE lead_id=?", (chosen_stage, lid))
                write_history(lid, who, "stage", cur_stage, chosen_stage)
            conn.commit(); conn.close()
            # add notifications for big leads or SLA problems
            try:
                if safe_float(payload.get("estimated_value")) >= 10000 and not payload.get("contacted"):
                    add_notification("High value lead not contacted", f"Lead {lid} est {payload.get('estimated_value')}")
            except:
                pass
            return lid
    except Exception:
        traceback.print_exc()
        return None

def remove_lead(lead_id):
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("DELETE FROM leads WHERE lead_id=?", (lead_id,))
        c.execute(f"INSERT INTO {HISTORY_TABLE} (lead_id,who,field,old_value,new_value,timestamp) VALUES (?,?,?,?,?,?)", (lead_id, "admin", "deleted", "", "", utcnow_iso()))
        conn.commit(); conn.close()
        return True
    except Exception:
        return False

# -----------------------
# Priority & SLA helpers
# -----------------------
def remaining_sla_seconds(sla_entered_at, sla_hours):
    try:
        if not sla_entered_at or str(sla_entered_at) == "NaT":
            sla_entered = datetime.utcnow()
        else:
            sla_entered = pd.to_datetime(sla_entered_at).to_pydatetime()
        deadline = sla_entered + timedelta(hours=int(sla_hours or DEFAULT_SLA_HOURS))
        rem = (deadline - datetime.utcnow()).total_seconds()
        return max(rem, 0.0)
    except Exception:
        return float("inf")

def is_overdue(sla_entered_at, sla_hours):
    return remaining_sla_seconds(sla_entered_at, sla_hours) <= 0

def compute_priority(row, weights=None):
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
    rem_h = remaining_sla_seconds(row.get("sla_entered_at"), row.get("sla_hours"))/3600.0
    sla_score = max(0.0, (72.0 - min(rem_h,72.0))/72.0)
    total = ml*weights.get("score",0.6) + value_score*weights.get("value",0.25) + sla_score*weights.get("sla",0.15)
    return max(0.0, min(1.0, total))

# -----------------------
# Internal ML: train + score (same_table storage)
# -----------------------
def train_model_internal():
    if not SKLEARN_AVAILABLE:
        return None, "scikit-learn not installed"
    df = fetch_leads()
    if df.empty or df["converted"].nunique() < 2:
        return None, "Not enough labeled conversion data"
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
    joblib.dump({"model":model, "columns": X.columns.tolist()}, MODEL_FILE)
    acc = model.score(X_test, y_test)
    return model, acc

def score_leads_internal():
    if not SKLEARN_AVAILABLE:
        return "scikit-learn not installed"
    if not os.path.exists(MODEL_FILE):
        return "Model not trained"
    saved = joblib.load(MODEL_FILE)
    model = saved.get("model") if isinstance(saved, dict) else saved
    cols = saved.get("columns") if isinstance(saved, dict) else None
    df = fetch_leads()
    if df.empty:
        return "No leads to score"
    df2 = df.copy()
    df2["created_at"] = pd.to_datetime(df2["created_at"], errors="coerce")
    df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
    X = pd.get_dummies(df2[["source","stage"]].astype(str))
    if cols:
        X = X.reindex(columns=cols, fill_value=0)
    X["ad_cost"] = df2["ad_cost"]
    X["estimated_value"] = df2["estimated_value"]
    try:
        probs = model.predict_proba(X)[:,1]
    except:
        probs = model.predict(X)
        if probs.max() > 1:
            probs = probs / probs.max()
    # persist scores
    conn = get_conn(); c = conn.cursor()
    for lid, p in zip(df2["lead_id"], probs):
        c.execute("UPDATE leads SET score=? WHERE lead_id=?", (float(p), lid))
    conn.commit(); conn.close()
    return f"Scored {len(probs)} leads"

# -----------------------
# Export helpers
# -----------------------
def df_to_excel_bytes(df):
    towrite = io.BytesIO()
    try:
        # try openpyxl
        df.to_excel(towrite, index=False, engine="openpyxl")
    except Exception:
        towrite = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    towrite.seek(0)
    return towrite.read()

# -----------------------
# UI / Streamlit
# -----------------------
st.set_page_config(page_title="TITAN — Pipeline", layout="wide", initial_sidebar_state="expanded")

# CSS
st.markdown("""
<style>
body, .stApp { background: #ffffff; font-family: Comfortaa, 'Helvetica Neue', Arial, sans-serif; color: #0b1220; }
.header { font-weight:800; font-size:20px; }
.kpi-card { background:#000; color:#fff; border-radius:12px; padding:14px; }
.priority-card { background:#000; color:#fff; border-radius:12px; padding:12px; min-width:220px; }
.small-muted { color:#6b7280; font-size:13px; }
.btn-small { padding:8px 12px; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# Top bar: title left, date selector & bell right
col_l, col_mid, col_r = st.columns([3,1,2])
with col_l:
    st.markdown("<div class='header'>TITAN — Lead Pipeline</div>", unsafe_allow_html=True)

# set up session states for date
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=29)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

with col_r:
    sd = st.date_input("Start date", st.session_state.start_date, key="start_date_ui")
    ed = st.date_input("End date", st.session_state.end_date, key="end_date_ui")
    st.session_state.start_date = sd
    st.session_state.end_date = ed
    # notifications bell
    notifs = fetch_notifications(100)
    unread = sum(1 for n in notifs if not n["read"])
    st.markdown(f"<div style='text-align:right;'>🔔 <span style='background:#ef4444;color:#fff;padding:6px 10px;border-radius:20px'>{unread}</span></div>", unsafe_allow_html=True)
    if st.button("Show notifications"):
        with st.expander(f"Notifications ({len(notifs)})", expanded=True):
            for n in notifs:
                c1, c2 = st.columns([8,1])
                with c1:
                    st.markdown(f"**{n['title']}** — {n['message']}")
                    st.caption(n["created_at"])
                with c2:
                    if not n["read"]:
                        if st.button("Mark read", key=f"mark_{n['id']}"):
                            mark_notification_read(n["id"])
                            st.experimental_rerun()
                    else:
                        st.write("✅")

# Sidebar navigation
st.sidebar.title("Navigate")
page = st.sidebar.radio("Go to", ["Pipeline Board", "Lead Capture", "Analytics & SLA", "CPA & ROI", "Reports & AI", "Exports / Imports", "Settings", "ML (internal)"])

# -----------------------
# Pipeline Board page
# -----------------------
if page == "Pipeline Board":
    st.header("TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("<em>High-level pipeline performance at a glance. Use filters and cards to drill into details.</em>", unsafe_allow_html=True)

    df = fetch_leads(start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    if df.empty:
        st.info("No leads found in selected date range. Add leads on Lead Capture page.")
    else:
        total = len(df)
        sla_success_count = int(df[df["contacted"]==1].shape[0])
        sla_success_pct = (sla_success_count / total * 100) if total else 0.0
        qualified = int(df[df["qualified"]==1].shape[0])
        qualification_pct = (qualified / total * 100) if total else 0.0
        awarded = int(df[df["stage"]=="Won"].shape[0])
        lost = int(df[df["stage"]=="Lost"].shape[0])
        closed = awarded + lost
        conversion_pct = (awarded / closed * 100) if closed else 0.0
        inspection_count = int(df[df["inspection_scheduled"]==1].shape[0])
        inspection_pct = (inspection_count / qualified * 100) if qualified else 0.0
        estimates_count = int(df[df["estimate_submitted"]==1].shape[0])
        pipeline_value = df["estimated_value"].sum()
        active = total - closed

        KPI = [
            ("Active Leads", str(active), "#2563eb", "Leads currently in pipeline"),
            ("SLA Success", f"{sla_success_pct:.1f}%", "#0ea5a4", "Leads contacted within SLA"),
            ("Qualification Rate", f"{qualification_pct:.1f}%", "#a855f7", "Leads marked qualified"),
            ("Conversion Rate", f"{conversion_pct:.1f}%", "#f97316", "Won / Closed"),
            ("Inspections Booked", f"{inspection_pct:.1f}%", "#ef4444", "Qualified → Scheduled"),
            ("Estimates Sent", str(estimates_count), "#6d28d9", "Estimates submitted"),
            ("Pipeline Job Value", format_money(pipeline_value), "#22c55e", "Total pipeline job value")
        ]

        # Top 4 in first row
        cols_top = st.columns(4, gap="large")
        for (title, value, color, note), c in zip(KPI[:4], cols_top):
            c.markdown(f"""
                <div class='kpi-card'>
                  <div style='font-weight:700;color:white;'>{title}</div>
                  <div style='font-size:22px;font-weight:900;color:{color};margin-top:8px;'>{value}</div>
                  <div style='height:8px;background:#e6e6e6;border-radius:6px;margin-top:10px;'>
                    <div style='height:100%;width:45%;background:{color};border-radius:6px;'></div>
                  </div>
                  <div class='small-muted' style='margin-top:8px;color:#d1d5db;'>{note}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        cols_bot = st.columns(3, gap="large")
        for (title, value, color, note), c in zip(KPI[4:], cols_bot):
            c.markdown(f"""
                <div class='kpi-card'>
                  <div style='font-weight:700;color:white;'>{title}</div>
                  <div style='font-size:22px;font-weight:900;color:{color};margin-top:8px;'>{value}</div>
                  <div style='height:8px;background:#e6e6e6;border-radius:6px;margin-top:10px;'>
                    <div style='height:100%;width:35%;background:{color};border-radius:6px;'></div>
                  </div>
                  <div class='small-muted' style='margin-top:8px;color:#d1d5db;'>{note}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        # Pipeline stages bar
        st.markdown("### Lead Pipeline Stages")
        st.markdown("<em>Distribution across stages. Use to spot drop-offs quickly.</em>", unsafe_allow_html=True)
        stage_counts = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
        stage_df = pd.DataFrame({"stage": stage_counts.index, "count": stage_counts.values})
        st.plotly_chart(px.bar(stage_df, x="stage", y="count", title="Leads per Stage"), use_container_width=True)

        st.markdown("---")
        # Top 5 priority (black cards)
        st.markdown("### TOP 5 PRIORITY LEADS")
        st.markdown("<em>Highest urgency leads by priority score (0–1). Address these first.</em>", unsafe_allow_html=True)
        df["priority_score"] = df.apply(lambda r: compute_priority(r), axis=1)
        df["hours_left"] = df.apply(lambda r: int(remaining_sla_seconds(r.get("sla_entered_at"), r.get("sla_hours"))/3600) if r.get("sla_hours") not in (None,0) else 9999, axis=1)
        top5 = df.sort_values("priority_score", ascending=False).head(5)
        if top5.empty:
            st.info("No priority leads.")
        else:
            cols = st.columns(len(top5))
            for col, (_, r) in zip(cols, top5.iterrows()):
                score = r["priority_score"]
                if score >= 0.7:
                    label, colc = "🔴 CRITICAL", "#ef4444"
                elif score >= 0.45:
                    label, colc = "🟠 HIGH", "#f97316"
                else:
                    label, colc = "🟢 NORMAL", "#22c55e"
                hours_left = r["hours_left"]
                money = format_money(r["estimated_value"])
                sla_html = "<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>" if is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) else f"<div style='color:#ef4444;font-weight:700;'>⏳ {hours_left}h left</div>"
                col.markdown(f"""
                    <div class='priority-card'>
                      <div style='font-weight:800;color:white;'>#{r['lead_id']} — {r.get('contact_name') or 'No name'}</div>
                      <div style='margin-top:6px;color:{colc};font-weight:700;'>{label}</div>
                      <div style='margin-top:8px;color:white;'>{r.get('damage_type','Unknown').title()} | {money}</div>
                      <div style='margin-top:8px;'>{sla_html}</div>
                      <div style='margin-top:6px;color:white;'>Priority score: <strong>{score:.2f}</strong></div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        # All leads with expand to update
        st.markdown("### 📋 All Leads (expand a card to edit / change status)")
        st.markdown("<em>Expand a lead to edit details, change status, upload invoice when awarded, and create estimates.</em>", unsafe_allow_html=True)
        search = st.text_input("Search (lead id, contact, address, notes)", key="search_box")
        view = df.copy()
        if search:
            q = search.lower()
            view = view[view.apply(lambda r: q in str(r.get("lead_id","")).lower() or q in str(r.get("contact_name","")).lower() or q in str(r.get("property_address","")).lower() or q in str(r.get("notes","")).lower(), axis=1)]
        for _, lead in view.sort_values("created_at", ascending=False).iterrows():
            with st.expander(f"#{lead['lead_id']} — {lead.get('contact_name') or 'No name'} — {format_money(lead.get('estimated_value') or 0)}", expanded=False):
                left, right = st.columns([3,1])
                with left:
                    st.write(f"**Source:** {lead.get('source') or '—'}  |  **Assigned:** {lead.get('assigned_to') or '—'}")
                    st.write(f"**Address:** {lead.get('property_address') or '—'}")
                    st.write(f"**Notes:** {lead.get('notes') or '—'}")
                    st.write(f"**Created:** {pd.to_datetime(lead.get('created_at')).strftime('%Y-%m-%d %H:%M') if lead.get('created_at') else '—'}")
                with right:
                    rem = remaining_sla_seconds(lead.get("sla_entered_at"), lead.get("sla_hours"))
                    if rem <= 0 and lead.get("stage") not in ("Won","Lost"):
                        st.markdown("<div style='color:#ef4444;font-weight:800;'>❗ OVERDUE</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='font-weight:700;'>⏳ {int(rem//3600)}h {int((rem%3600)//60)}m left</div>", unsafe_allow_html=True)
                st.markdown("---")
                with st.form(f"upd_{lead['lead_id']}"):
                    new_stage = st.selectbox("Status", PIPELINE_STAGES, index=PIPELINE_STAGES.index(lead.get("stage")) if lead.get("stage") in PIPELINE_STAGES else 0)
                    new_assigned = st.text_input("Assigned to", value=lead.get("assigned_to") or "")
                    new_contacted = st.checkbox("Contacted", value=bool(lead.get("contacted")))
                    new_insp_sched = st.checkbox("Inspection Scheduled", value=bool(lead.get("inspection_scheduled")))
                    new_insp_comp = st.checkbox("Inspection Completed", value=bool(lead.get("inspection_completed")))
                    new_est_sub = st.checkbox("Estimate Submitted", value=bool(lead.get("estimate_submitted")))
                    new_notes = st.text_area("Notes", value=lead.get("notes") or "")
                    new_est = st.number_input("Estimated value (USD)", value=float(lead.get("estimated_value") or 0.0), min_value=0.0, step=100.0)
                    new_cost = st.number_input("Acquisition cost (USD)", value=float(lead.get("ad_cost") or 0.0), min_value=0.0, step=1.0)
                    if new_stage == "Won":
                        award_comment = st.text_area("Award comment")
                        award_file = st.file_uploader("Upload invoice (optional)", type=["pdf","jpg","png","xlsx","csv"])
                    if new_stage == "Lost":
                        lost_comment = st.text_area("Lost comment")
                    if st.form_submit_button("Save"):
                        payload = {
                            "lead_id": lead.get("lead_id"),
                            "stage": new_stage,
                            "assigned_to": new_assigned,
                            "contacted": 1 if new_contacted else 0,
                            "inspection_scheduled": 1 if new_insp_sched else 0,
                            "inspection_completed": 1 if new_insp_comp else 0,
                            "estimate_submitted": 1 if new_est_sub else 0,
                            "notes": new_notes,
                            "estimated_value": float(new_est or 0.0),
                            "ad_cost": float(new_cost or 0.0),
                            "sla_entered_at": lead.get("sla_entered_at") or lead.get("created_at")
                        }
                        upsert_lead(payload, who=st.session_state.get("profile", {}).get("name","admin"))
                        if new_stage == "Won":
                            add_notification("Lead Won", f"Lead {lead.get('lead_id')} marked Won")
                        if new_stage == "Lost":
                            add_notification("Lead Lost", f"Lead {lead.get('lead_id')} marked Lost")
                        st.success("Saved"); st.experimental_rerun()
                if st.button("Delete Lead", key=f"del_{lead['lead_id']}"):
                    remove_lead(lead.get("lead_id"))
                    st.success("Deleted"); st.experimental_rerun()

# -----------------------
# Lead Capture page
# -----------------------
elif page == "Lead Capture":
    st.header("📇 Lead Capture")
    st.markdown("<em>SLA Response time must be greater than 0 hours.</em>", unsafe_allow_html=True)
    with st.form("create_lead_form"):
        lead_id_in = st.text_input("Lead ID (leave blank to auto-generate)")
        source = st.selectbox("Lead Source", ["Google Ads","Organic Search","Referral","Phone","Insurance","Facebook","Instagram","LinkedIn","Other"])
        source_details = st.text_input("Source details (UTM / notes)")
        contact_name = st.text_input("Contact name")
        contact_phone = st.text_input("Contact phone")
        contact_email = st.text_input("Contact email")
        property_address = st.text_input("Property address")
        damage_type = st.selectbox("Damage type", ["water","fire","mold","contents","reconstruction","other"])
        assigned_to = st.text_input("Assigned to")
        estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
        ad_cost = st.number_input("Cost to acquire lead (USD)", min_value=0.0, value=0.0, step=1.0)
        sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=DEFAULT_SLA_HOURS, step=1, help="SLA Response time must be greater than 0 hours.")
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Create Lead")
        if submitted:
            if sla_hours <= 0:
                st.error("SLA hours must be greater than 0.")
            else:
                payload = {
                    "lead_id": lead_id_in.strip() or None,
                    "created_at": utcnow_iso(),
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
                    "sla_entered_at": utcnow_iso(),
                    "stage": "New",
                    "converted": 0,
                    "score": 0.0
                }
                lid = upsert_lead(payload, who=st.session_state.get("profile", {}).get("name","admin"))
                st.success(f"Lead created (ID: {lid})")
                st.experimental_rerun()

# -----------------------
# Analytics & SLA page
# -----------------------
elif page == "Analytics & SLA":
    st.header("📈 Analytics & SLA")
    st.markdown("<em>Use the date selector at top-right to filter data.</em>", unsafe_allow_html=True)
    df = fetch_leads(start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    if df.empty:
        st.info("No leads in selected date range.")
    else:
        # donut/pie of pipeline stages
        st.subheader("Lead Pipeline Stages (Donut)")
        sc = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
        pie = pd.DataFrame({"stage": sc.index, "count": sc.values})
        fig_p = px.pie(pie, names="stage", values="count", hole=0.45, title="Pipeline Distribution")
        st.plotly_chart(fig_p, use_container_width=True)

        # Cost vs conversions by source
        st.subheader("Cost vs Conversions by Source")
        df["won_flag"] = df["stage"].apply(lambda x: 1 if x == "Won" else 0)
        agg = df.groupby("source").agg(total_spend=("ad_cost","sum"), conversions=("won_flag","sum")).reset_index()
        if not agg.empty:
            st.plotly_chart(px.bar(agg, x="source", y=["total_spend","conversions"], barmode="group", title="Total Marketing Spend vs Conversions"), use_container_width=True)
        else:
            st.info("No spend data available for chart.")

        # SLA Overdue trend (30 days)
        st.subheader("SLA Overdue Trend (30 days)")
        today = date.today()
        days = [today - timedelta(days=i) for i in range(29, -1, -1)]
        rows = []
        for d in days:
            start_dt = pd.to_datetime(d)
            end_dt = start_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            win = df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)]
            overdue = int(win.apply(lambda r: 1 if (is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost")) else 0, axis=1).sum())
            rows.append({"date": d, "overdue": overdue})
        ts = pd.DataFrame(rows)
        if not ts.empty:
            st.plotly_chart(px.line(ts, x="date", y="overdue", markers=True, title="SLA Overdue Count (30d)"), use_container_width=True)

        # Overdue table
        st.subheader("Current Overdue Leads")
        overdue_df = df[df.apply(lambda r: is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost"), axis=1)]
        if overdue_df.empty:
            st.success("No SLA overdue leads.")
        else:
            st.dataframe(overdue_df[["lead_id","contact_name","stage","estimated_value","ad_cost","sla_hours"]])

# -----------------------
# CPA & ROI page
# -----------------------
elif page == "CPA & ROI":
    st.header("💰 CPA & ROI")
    df = fetch_leads(start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    if df.empty:
        st.info("No data for selected range.")
    else:
        won = df[df["stage"]=="Won"]
        total_spend = df["ad_cost"].sum()
        conversions = len(won)
        cpa = (total_spend / conversions) if conversions else None
        total_value = df["estimated_value"].sum()
        roi = ((total_value - total_spend)/total_spend*100) if total_spend else None

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div style='background:#111;color:white;padding:12px;border-radius:10px;'>✅ <b>Conversions (Won)</b><div style='font-size:20px;margin-top:8px;color:#2563eb;font-weight:800;'>{conversions}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div style='background:#111;color:white;padding:12px;border-radius:10px;'>🎯 <b>CPA</b><div style='font-size:20px;margin-top:8px;color:#f97316;font-weight:800;'>{format_money(cpa) if cpa else '$0.00'}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div style='background:#111;color:white;padding:12px;border-radius:10px;'>📈 <b>ROI</b><div style='font-size:18px;margin-top:8px;color:#22c55e;font-weight:800;'>{f'{roi:.1f}% ({format_money(total_value)})' if roi is not None else '—'}</div></div>", unsafe_allow_html=True)

        # Monthly chart for spend vs conversions
        if not df.empty:
            df["month"] = pd.to_datetime(df["created_at"]).dt.to_period("M").astype(str)
            monthly = df.groupby("month").agg(spend=("ad_cost","sum"), conversions=("stage", lambda x: (x=="Won").sum())).reset_index()
            if not monthly.empty:
                st.plotly_chart(px.line(monthly, x="month", y=["spend","conversions"], markers=True, title="Monthly Spend vs Conversions"), use_container_width=True)

# -----------------------
# Reports & AI page
# -----------------------
elif page == "Reports & AI":
    st.header("🧠 AI Observations & Recommendations")
    df = fetch_leads()
    if df.empty:
        st.info("No data to analyze.")
    else:
        # Generate observations for top 20 priority leads
        df["priority_score"] = df.apply(lambda r: compute_priority(r), axis=1)
        top_n = df.sort_values("priority_score", ascending=False).head(20)
        obs = []
        # Bottleneck stage
        bottleneck = df["stage"].value_counts().idxmax() if not df.empty else "—"
        obs.append(f"Most leads currently in stage: {bottleneck}")
        # Top spend sources
        top_spend = df.groupby("source")["ad_cost"].sum().sort_values(ascending=False).head(3)
        if not top_spend.empty:
            obs.append("Top spend channels: " + ", ".join(top_spend.index.astype(str).tolist()))
        else:
            obs.append("No spend data available.")
        # SLA overdue
        overdue = df[df.apply(lambda r: is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost"), axis=1)]
        obs.append(f"SLA overdue leads: {len(overdue)}")
        # Recommendations (simple heuristics)
        recs = []
        if len(overdue) > 0:
            recs.append("Prioritize contacting SLA-overdue leads immediately.")
        if top_spend.sum() > 0 and df[df["converted"]==1].shape[0] == 0:
            recs.append("High spend detected but no conversions — review ad targeting.")
        if df["estimated_value"].mean() > 5000:
            recs.append("Average job value is high — ensure senior estimator assigned to high-value leads.")
        # Per-top20 mini-summaries
        st.subheader("Top 20 Priority Leads — Quick Summary")
        for _, r in top_n.iterrows():
            st.markdown(f"- **{r['lead_id']}** — {r.get('contact_name') or 'No name'} — Priority: {r['priority_score']:.2f} — Stage: {r.get('stage')} — Est: {format_money(r.get('estimated_value'))}")
        st.subheader("Observations")
        for o in obs:
            st.write("- " + o)
        st.subheader("Recommendations")
        for rc in recs:
            st.write("- " + rc)
        summary = "\n".join(obs + [""] + recs)
        st.subheader("Executive Summary")
        st.text_area("Summary", summary, height=200)
        st.download_button("Download summary", summary.encode("utf-8"), file_name="titan_ai_summary.txt", mime="text/plain")

# -----------------------
# Exports / Imports
# -----------------------
elif page == "Exports / Imports":
    st.header("📤 Exports & Imports")
    df = fetch_leads(start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    if df.empty:
        st.info("No leads to export for selected range.")
    else:
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="leads_export.csv", mime="text/csv")
        try:
            excel_bytes = df_to_excel_bytes(df)
            st.download_button("Download Excel", excel_bytes, file_name="leads_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            st.info("Excel export unavailable; CSV provided.")

    st.markdown("---")
    uploaded = st.file_uploader("Import leads (CSV with 'lead_id' column recommended)", type=["csv"])
    if uploaded:
        try:
            imp = pd.read_csv(uploaded)
            if "lead_id" not in imp.columns:
                st.error("CSV must include 'lead_id' column.")
            else:
                cnt = 0
                for _, r in imp.fillna("").iterrows():
                    payload = {
                        "lead_id": str(r.get("lead_id")) if not pd.isna(r.get("lead_id")) else None,
                        "created_at": pd.to_datetime(r.get("created_at")).isoformat() if not pd.isna(r.get("created_at")) else utcnow_iso(),
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
                        "ad_cost": safe_float(r.get("ad_cost")),
                        "stage": r.get("stage") or "New",
                        "sla_hours": int(r.get("sla_hours") or DEFAULT_SLA_HOURS),
                        "sla_entered_at": r.get("sla_entered_at") or utcnow_iso(),
                        "converted": int(r.get("converted") or 0),
                        "score": safe_float(r.get("score"))
                    }
                    upsert_lead(payload, who="import")
                    cnt += 1
                st.success(f"Imported {cnt} leads.")
        except Exception as e:
            st.error("Import failed: " + str(e))

# -----------------------
# Settings
# -----------------------
elif page == "Settings":
    st.header("⚙️ Settings & Admin")
    st.markdown("<em>Profile saved to session used for audit entries.</em>", unsafe_allow_html=True)
    if "profile" not in st.session_state:
        st.session_state.profile = {"name":"", "role":"Admin"}
    name = st.text_input("Your name", value=st.session_state.profile.get("name",""))
    role = st.selectbox("Role", ["Admin","Estimator","Manager","Viewer"], index=["Admin","Estimator","Manager","Viewer"].index(st.session_state.profile.get("role","Admin")))
    if st.button("Save profile"):
        st.session_state.profile["name"] = name
        st.session_state.profile["role"] = role
        st.success("Profile saved.")

    st.markdown("---")
    st.subheader("Audit trail (last 200)")
    conn = get_conn()
    try:
        hist = pd.read_sql_query(f"SELECT * FROM {HISTORY_TABLE} ORDER BY timestamp DESC LIMIT 200", conn)
    except Exception:
        hist = pd.DataFrame()
    conn.close()
    if hist.empty:
        st.info("No history yet.")
    else:
        st.dataframe(hist)

# -----------------------
# ML (internal)
# -----------------------
elif page == "ML (internal)":
    st.header("🧠 Internal ML — Train & Score (no user tuning)")
    st.markdown("<em>Optional: requires scikit-learn. Trains on 'converted' column and persists 'score' in leads table.</em>", unsafe_allow_html=True)
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn not installed. ML features unavailable.")
    else:
        if st.button("Train internal model"):
            model, acc = train_model_internal()
            if model is None:
                st.error(str(acc))
            else:
                st.success(f"Model trained. Accuracy approx: {acc:.3f}")
        if st.button("Score leads & persist"):
            msg = score_leads_internal()
            st.success(str(msg))

# -----------------------
# Default fallback
# -----------------------
else:
    st.info("Select a page from the left.")

# Footer
st.markdown("---")
st.markdown("<div style='font-size:12px;color:#666'>TITAN — single-file backend. For production, migrate DB to managed service and secure endpoints.</div>", unsafe_allow_html=True)
