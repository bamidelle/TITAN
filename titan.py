# titan_pipeline_app.py
"""
TITAN — Single-file Streamlit backend
Features:
- SQLite persistence (sqlite3)
- Pipeline Board (KPIs in 2 rows), Top 5 Priority Leads (black cards)
- Analytics & SLA (donut + SLA trend + Cost vs Conversions)
- CPA & ROI page
- Lead Capture (auto Lead ID, SLA validation > 0)
- Exports (CSV + Excel fallback)
- Internal ML training & scoring (no user tuning)
- Notifications bell (unread count) and dismissable notifications
- Audit trail (lead_history)
- Admin settings (profile saved in session)
- Search & quick filters, date picker top-right
- Auto-progression (Option A) but manual override allowed
- Stores data forever (option A)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import plotly.express as px
from datetime import datetime, timedelta, date, timezone
import io, os, math, base64, traceback

# Optional ML imports (app will continue if these are missing)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ------------------------------
# CONFIG
# ------------------------------
DB_FILE = "titan_leads.db"
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
MODEL_FILE = "titan_titan_model.joblib"
NOTIF_TABLE = "notifications"

# ------------------------------
# DB initialization (no param binding in CREATE)
# ------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
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
    c.execute(f"""
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

# ------------------------------
# DB helpers
# ------------------------------
def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def now_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def read_leads(start_date=None, end_date=None):
    conn = get_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM leads ORDER BY created_at DESC", conn, parse_dates=["created_at","sla_entered_at","inspection_scheduled_at","estimate_submitted_at","awarded_date","lost_date"])
    except Exception:
        df = pd.DataFrame()
    conn.close()
    if df.empty:
        # ensure columns exist to avoid KeyError
        cols = ["id","lead_id","created_at","source","source_details","contact_name","contact_phone","contact_email","property_address","damage_type","assigned_to","notes","estimated_value","ad_cost","stage","sla_hours","sla_entered_at","contacted","inspection_scheduled","inspection_scheduled_at","inspection_completed","estimate_submitted","estimate_submitted_at","awarded_date","awarded_invoice","lost_date","qualified","converted","score"]
        return pd.DataFrame(columns=cols)
    # normalize
    df["estimated_value"] = pd.to_numeric(df["estimated_value"], errors="coerce").fillna(0.0)
    df["ad_cost"] = pd.to_numeric(df["ad_cost"], errors="coerce").fillna(0.0)
    for b in ["contacted","inspection_scheduled","inspection_completed","estimate_submitted","qualified","converted"]:
        if b in df.columns:
            df[b] = df[b].fillna(0).astype(int)
    if start_date:
        df = df[df["created_at"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["created_at"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    return df.reset_index(drop=True)

def write_history(lead_id, who, field, old, new):
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("INSERT INTO lead_history (lead_id,who,field,old_value,new_value,timestamp) VALUES (?,?,?,?,?,?)", (lead_id, who, field, str(old), str(new), now_iso()))
        conn.commit(); conn.close()
    except Exception:
        pass

def add_notification(title, message):
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute(f"INSERT INTO {NOTIF_TABLE} (title,message,created_at,read) VALUES (?,?,?,0)", (title, message, now_iso()))
        conn.commit(); conn.close()
    except Exception:
        pass

def fetch_notifications(limit=50):
    conn = get_conn(); c = conn.cursor()
    c.execute(f"SELECT id,title,message,created_at,read FROM {NOTIF_TABLE} ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = c.fetchall()
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

# ------------------------------
# Lead create/update/delete
# ------------------------------
def generate_lead_id():
    return f"L{int(datetime.utcnow().timestamp())}"

def save_lead(payload, who="admin"):
    """
    Upsert a lead. payload dict may contain lead_id (optional). Returns lead_id.
    Option A: auto-move stage forward when flags set, but manual 'stage' in payload overrides auto-progression.
    """
    try:
        conn = get_conn(); c = conn.cursor()
        lid = payload.get("lead_id") or generate_lead_id()
        # check existing
        c.execute("SELECT * FROM leads WHERE lead_id=?", (lid,))
        existing = c.fetchone()
        # helper to insert or update
        if not existing:
            # insert
            created_at = payload.get("created_at") or now_iso()
            sla_entered_at = payload.get("sla_entered_at") or created_at
            stage = payload.get("stage") or "New"
            c.execute("""
                INSERT INTO leads (lead_id, created_at, source, source_details, contact_name, contact_phone, contact_email,
                                   property_address, damage_type, assigned_to, notes, estimated_value, ad_cost, stage, sla_hours, sla_entered_at,
                                   contacted, inspection_scheduled, inspection_scheduled_at, inspection_completed, estimate_submitted, estimate_submitted_at, awarded_date, awarded_invoice, lost_date, qualified, converted, score)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                lid,
                created_at,
                payload.get("source"),
                payload.get("source_details"),
                payload.get("contact_name"),
                payload.get("contact_phone"),
                payload.get("contact_email"),
                payload.get("property_address"),
                payload.get("damage_type"),
                payload.get("assigned_to"),
                payload.get("notes"),
                float(payload.get("estimated_value") or 0.0),
                float(payload.get("ad_cost") or 0.0),
                stage,
                int(payload.get("sla_hours") or DEFAULT_SLA_HOURS),
                sla_entered_at,
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
            c.execute("INSERT INTO lead_history (lead_id,who,field,old_value,new_value,timestamp) VALUES (?,?,?,?,?,?)", (lid, who, "create", "", stage, now_iso()))
            conn.commit(); conn.close()
            return lid
        else:
            # update existing
            # read existing as dict
            conn2 = sqlite3.connect(DB_FILE, check_same_thread=False)
            df = pd.read_sql_query("SELECT * FROM leads WHERE lead_id=?", conn2, params=(lid,), parse_dates=["created_at","sla_entered_at"])
            conn2.close()
            if df.empty:
                # fallback to insert
                return save_lead(payload, who=who)
            cur = df.iloc[0].to_dict()
            # auto-progression detection
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
            except:
                auto_stage = None
            # manual override: payload 'stage' takes precedence
            chosen_stage = payload.get("stage") if payload.get("stage") is not None else (auto_stage or cur.get("stage"))
            # list of fields to update
            up_fields = ["source","source_details","contact_name","contact_phone","contact_email","property_address","damage_type","assigned_to","notes","estimated_value","ad_cost","stage","sla_hours","sla_entered_at","contacted","inspection_scheduled","inspection_scheduled_at","inspection_completed","estimate_submitted","estimate_submitted_at","awarded_date","awarded_invoice","lost_date","qualified","converted","score"]
            for f in up_fields:
                if f in payload:
                    new_val = payload.get(f)
                    if f in ["contacted","inspection_scheduled","inspection_completed","estimate_submitted","qualified","converted"]:
                        new_val = int(bool(new_val))
                    if f in ["estimated_value","ad_cost","score"]:
                        try:
                            new_val = float(new_val or 0.0)
                        except:
                            new_val = 0.0
                    if f == "sla_hours":
                        try:
                            new_val = int(new_val)
                            if new_val <= 0:
                                new_val = DEFAULT_SLA_HOURS
                        except:
                            new_val = DEFAULT_SLA_HOURS
                    old_val = cur.get(f)
                    if str(old_val) != str(new_val):
                        c.execute(f"UPDATE leads SET {f}=? WHERE lead_id=?", (new_val, lid))
                        write_history(lid, who, f, old_val, new_val)
            # ensure chosen_stage applied
            c.execute("SELECT stage FROM leads WHERE lead_id=?", (lid,))
            row_stage = c.fetchone()[0]
            if row_stage != chosen_stage:
                c.execute("UPDATE leads SET stage=? WHERE lead_id=?", (chosen_stage, lid))
                write_history(lid, who, "stage", row_stage, chosen_stage)
            conn.commit(); conn.close()
            # notifications for high-value or SLA issues
            try:
                if float(payload.get("estimated_value") or 0.0) >= 10000 and not payload.get("contacted"):
                    add_notification("High value lead not contacted", f"Lead {lid} value {payload.get('estimated_value')}")
            except:
                pass
            return lid
    except Exception as e:
        traceback.print_exc()
        return None

def delete_lead(lead_id):
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("DELETE FROM leads WHERE lead_id=?", (lead_id,))
        c.execute("INSERT INTO lead_history (lead_id,who,field,old_value,new_value,timestamp) VALUES (?,?,?,?,?,?)", (lead_id, "admin", "deleted", "", "", now_iso()))
        conn.commit(); conn.close()
        return True
    except Exception:
        return False

# ------------------------------
# Priority / SLA helpers
# ------------------------------
def remaining_sla_seconds(sla_entered_at, sla_hours):
    try:
        if sla_entered_at is None or str(sla_entered_at) == "NaT":
            sla_entered = datetime.utcnow()
        else:
            sla_entered = pd.to_datetime(sla_entered_at).to_pydatetime()
        deadline = sla_entered + timedelta(hours=int(sla_hours or DEFAULT_SLA_HOURS))
        remain = (deadline - datetime.utcnow()).total_seconds()
        return max(remain, 0.0)
    except Exception:
        return float("inf")

def is_overdue_sla(sla_entered_at, sla_hours):
    return remaining_sla_seconds(sla_entered_at, sla_hours) <= 0

def compute_priority_score(row, weights=None):
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
    value_score = min(1.0, val / max(1.0, float(weights.get("value_baseline", 5000.0))))
    rem_h = remaining_sla_seconds(row.get("sla_entered_at"), row.get("sla_hours")) / 3600.0
    sla_score = max(0.0, (72.0 - min(rem_h,72.0)) / 72.0)
    total = ml * weights.get("score",0.6) + value_score * weights.get("value",0.25) + sla_score * weights.get("sla",0.15)
    return max(0.0, min(1.0, total))

# ------------------------------
# ML internal (train + score)
# ------------------------------
def train_internal_model():
    if not SKLEARN_AVAILABLE:
        return None, "scikit-learn not available"
    df = read_leads()
    if df.empty or df["converted"].nunique() < 2:
        return None, "Not enough labeled data to train"
    df2 = df.copy()
    df2["created_at"] = pd.to_datetime(df2["created_at"], errors="coerce")
    df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
    X = pd.get_dummies(df2[["source","stage"]].astype(str))
    X["ad_cost"] = df2["ad_cost"]
    X["estimated_value"] = df2["estimated_value"]
    X["age_days"] = df2["age_days"]
    y = df2["converted"].fillna(0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    # save columns for safe reindexing
    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_FILE)
    acc = model.score(X_test, y_test)
    return model, acc

def score_and_persist():
    if not SKLEARN_AVAILABLE:
        return "scikit-learn not available"
    if not os.path.exists(MODEL_FILE):
        return "Model not trained yet"
    saved = joblib.load(MODEL_FILE)
    model = saved.get("model") if isinstance(saved, dict) else saved
    cols = saved.get("columns") if isinstance(saved, dict) else None
    df = read_leads()
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
    # persist
    conn = get_conn(); c = conn.cursor()
    for lid, p in zip(df2["lead_id"], probs):
        c.execute("UPDATE leads SET score=? WHERE lead_id=?", (float(p), lid))
    conn.commit(); conn.close()
    return f"Scored {len(probs)} leads"

# ------------------------------
# Export helpers
# ------------------------------
def df_to_excel_bytes(df):
    towrite = io.BytesIO()
    try:
        # try openpyxl
        df.to_excel(towrite, index=False, engine="openpyxl")
    except Exception:
        # fallback to csv bytes
        towrite = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    towrite.seek(0)
    return towrite.read()

# ------------------------------
# UI setup
# ------------------------------
st.set_page_config(page_title="TITAN Pipeline", layout="wide")
# minimal CSS with Comfortaa-like font fallback
st.markdown("""
    <style>
    body, .stApp { background: #ffffff; font-family: 'Comfortaa', 'Helvetica', Arial, sans-serif; color: #0b1220; }
    .kpi-card { background:#000; color:#fff; border-radius:10px; padding:12px; }
    .priority-card { background:#000; color:#fff; border-radius:10px; padding:12px; min-width:220px; }
    .small-muted { color:#6b7280; font-size:13px; }
    </style>
""", unsafe_allow_html=True)

# top bar: title left, date selectors + bell right
left_col, _, right_col = st.columns([3,1,2])
with left_col:
    st.markdown("<h2 style='margin:8px 0;'>TITAN — Lead Pipeline</h2>", unsafe_allow_html=True)
# initialize session dates
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=29)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()
with right_col:
    sd = st.date_input("Start date", st.session_state.start_date, key="sd")
    ed = st.date_input("End date", st.session_state.end_date, key="ed")
    st.session_state.start_date = sd
    st.session_state.end_date = ed
    # notifications bell
    notifs = fetch_notifications(100)
    unread = sum(1 for n in notifs if not n["read"])
    st.markdown(f"<div style='text-align:right;'>🔔 <span style='background:#ef4444;color:white;padding:5px 10px;border-radius:20px'>{unread}</span></div>", unsafe_allow_html=True)
    if st.button("Show notifications"):
        with st.expander(f"Notifications ({len(notifs)})", expanded=True):
            for n in notifs:
                cols = st.columns([9,1])
                with cols[0]:
                    st.markdown(f"**{n['title']}** — {n['message']}")
                    st.caption(n["created_at"])
                with cols[1]:
                    if not n["read"]:
                        if st.button("Mark read", key=f"mark_{n['id']}"):
                            mark_notification_read(n["id"])
                            st.experimental_rerun()
                    else:
                        st.write("✅")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Pipeline Board", "Lead Capture", "Analytics & SLA", "CPA & ROI", "Reports", "Exports / Imports", "Settings", "ML (internal)"])

# ------------------------------
# Page: Pipeline Board
# ------------------------------
if page == "Pipeline Board":
    st.header("TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("<em>High-level pipeline performance at a glance.</em>", unsafe_allow_html=True)

    df = read_leads(start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    if df.empty:
        st.info("No leads yet. Add a lead on Lead Capture.")
    else:
        total_leads = len(df)
        sla_success_count = int(df[df["contacted"]==1].shape[0])
        sla_success_pct = (sla_success_count/total_leads*100) if total_leads else 0.0
        qualified_leads = int(df[df["qualified"]==1].shape[0])
        qualification_pct = (qualified_leads/total_leads*100) if total_leads else 0.0
        awarded_count = int(df[df["stage"]=="Won"].shape[0])
        lost_count = int(df[df["stage"]=="Lost"].shape[0])
        closed = awarded_count + lost_count
        conversion_rate = (awarded_count/closed*100) if closed else 0.0
        inspection_scheduled_count = int(df[df["inspection_scheduled"]==1].shape[0])
        inspection_pct = (inspection_scheduled_count/qualified_leads*100) if qualified_leads else 0.0
        estimate_sent_count = int(df[df["estimate_submitted"]==1].shape[0])
        pipeline_job_value = df["estimated_value"].sum()
        active_leads = total_leads - closed

        KPI_ITEMS = [
            ("Active Leads", str(active_leads), "#2563eb", "Leads currently in pipeline"),
            ("SLA Success", f"{sla_success_pct:.1f}%", "#0ea5a4", "Leads contacted within SLA"),
            ("Qualification Rate", f"{qualification_pct:.1f}%", "#a855f7", "Leads marked qualified"),
            ("Conversion Rate", f"{conversion_rate:.1f}%", "#f97316", "Won / Closed"),
            ("Inspections Booked", f"{inspection_pct:.1f}%", "#ef4444", "Qualified → Scheduled"),
            ("Estimates Sent", str(estimate_sent_count), "#6d28d9", "Estimates submitted"),
            ("Pipeline Job Value", format_money(pipeline_job_value), "#22c55e", "Total pipeline job value")
        ]

        # render top 4 in first row, remaining 3 in second row (spacing)
        cols_top = st.columns(4, gap="large")
        for (title, value, color, note), c in zip(KPI_ITEMS[:4], cols_top):
            pct = 50  # placeholder progress
            c.markdown(f"""
                <div style="background:#000;padding:12px;border-radius:10px;">
                    <div style="color:#fff;font-weight:700;">{title}</div>
                    <div style="font-weight:900;font-size:22px;color:{color};">{value}</div>
                    <div style="height:8px;background:#e6e6e6;border-radius:6px;margin-top:10px;">
                        <div style="height:100%;width:{pct}%;background:{color};border-radius:6px;"></div>
                    </div>
                    <div class="small-muted" style="margin-top:8px;color:#d1d5db;">{note}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        cols_bottom = st.columns(3, gap="large")
        for (title, value, color, note), c in zip(KPI_ITEMS[4:], cols_bottom):
            pct = 35
            c.markdown(f"""
                <div style="background:#000;padding:12px;border-radius:10px;">
                    <div style="color:#fff;font-weight:700;">{title}</div>
                    <div style="font-weight:900;font-size:22px;color:{color};">{value}</div>
                    <div style="height:8px;background:#e6e6e6;border-radius:6px;margin-top:10px;">
                        <div style="height:100%;width:{pct}%;background:{color};border-radius:6px;"></div>
                    </div>
                    <div class="small-muted" style="margin-top:8px;color:#d1d5db;">{note}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Pipeline Stages (bar)
        st.markdown("### Lead Pipeline Stages")
        st.markdown("<em>Distribution of leads across pipeline stages. Use this to spot stage drop-offs quickly.</em>", unsafe_allow_html=True)
        stage_counts = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
        stage_df = pd.DataFrame({"stage": stage_counts.index, "count": stage_counts.values})
        st.plotly_chart(px.bar(stage_df, x="stage", y="count", title="Leads per Stage"), use_container_width=True)

        st.markdown("---")
        # TOP 5 priority
        st.markdown("### TOP 5 PRIORITY LEADS")
        st.markdown("<em>Highest urgency leads by priority score (0–1). Address these first.</em>", unsafe_allow_html=True)
        df["priority_score"] = df.apply(lambda r: compute_priority_score(r), axis=1)
        df["hours_left"] = df.apply(lambda r: int(remaining_sla_seconds(r.get("sla_entered_at"), r.get("sla_hours"))/3600) if r.get("sla_hours") not in (None,0) else 9999, axis=1)
        pr = df.sort_values("priority_score", ascending=False).head(5)
        if pr.empty:
            st.info("No priority leads.")
        else:
            cols = st.columns(len(pr))
            for col, (_, r) in zip(cols, pr.iterrows()):
                sc = r["priority_score"]
                if sc >= 0.7:
                    label, color = "🔴 CRITICAL", "#ef4444"
                elif sc >= 0.45:
                    label, color = "🟠 HIGH", "#f97316"
                else:
                    label, color = "🟢 NORMAL", "#22c55e"
                hours_left = r["hours_left"]
                money = format_money(r["estimated_value"])
                overdue_html = "<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>" if is_overdue_sla(r.get("sla_entered_at"), r.get("sla_hours")) else f"<div style='color:#ef4444;font-weight:700;'>⏳ {hours_left}h left</div>"
                col.markdown(f"""
                    <div class="priority-card">
                        <div style="font-weight:800;color:white;">#{r['lead_id']} — {r.get('contact_name') or 'No name'}</div>
                        <div style="margin-top:6px;color:{color};font-weight:700;">{label}</div>
                        <div style="margin-top:8px;color:white;">{r.get('damage_type','Unknown').title()} | {money}</div>
                        <div style="margin-top:8px;">{overdue_html}</div>
                        <div style="margin-top:6px;color:white;">Priority: <strong>{sc:.2f}</strong></div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        # All leads with expand to edit
        st.markdown("### 📋 All Leads (expand to edit / change status)")
        st.markdown("<em>Expand a lead to edit details, change status, upload invoice when awarded, and create estimates.</em>", unsafe_allow_html=True)
        search_q = st.text_input("Search (id, contact, address, notes)", key="search_q")
        view_df = df.copy()
        if search_q:
            q = search_q.lower()
            view_df = view_df[view_df.apply(lambda r: q in str(r.get("lead_id","")).lower() or q in str(r.get("contact_name","")).lower() or q in str(r.get("property_address","")).lower() or q in str(r.get("notes","")).lower(), axis=1)]
        for _, lead in view_df.sort_values("created_at", ascending=False).iterrows():
            with st.expander(f"#{lead['lead_id']} — {lead.get('contact_name') or 'No name'} — {format_money(lead.get('estimated_value') or 0)}", expanded=False):
                left, right = st.columns([3,1])
                with left:
                    st.write(f"**Source:** {lead.get('source') or '—'}")
                    st.write(f"**Address:** {lead.get('property_address') or '—'}")
                    st.write(f"**Notes:** {lead.get('notes') or '—'}")
                    st.write(f"**Created:** {pd.to_datetime(lead.get('created_at')).strftime('%Y-%m-%d %H:%M') if lead.get('created_at') else '—'}")
                with right:
                    rem = remaining_sla_seconds(lead.get("sla_entered_at"), lead.get("sla_hours"))
                    if rem <= 0 and lead.get("stage") not in ("Won","Lost"):
                        st.error("OVERDUE")
                    else:
                        st.info(f"{int(rem//3600)}h {(int((rem%3600)//60))}m left")
                st.markdown("---")
                with st.form(f"update_{lead['lead_id']}", clear_on_submit=False):
                    ns = st.selectbox("Status", PIPELINE_STAGES, index=PIPELINE_STAGES.index(lead.get("stage")) if lead.get("stage") in PIPELINE_STAGES else 0)
                    assigned = st.text_input("Assigned to", value=lead.get("assigned_to") or "")
                    contacted = st.checkbox("Contacted", value=bool(lead.get("contacted")))
                    insp_sched = st.checkbox("Inspection Scheduled", value=bool(lead.get("inspection_scheduled")))
                    insp_comp = st.checkbox("Inspection Completed", value=bool(lead.get("inspection_completed")))
                    est_sub = st.checkbox("Estimate Submitted", value=bool(lead.get("estimate_submitted")))
                    new_val = st.number_input("Estimated value (USD)", value=float(lead.get("estimated_value") or 0.0), min_value=0.0, step=100.0)
                    new_cost = st.number_input("Acquisition cost (USD)", value=float(lead.get("ad_cost") or 0.0), min_value=0.0, step=1.0)
                    if ns == "Won":
                        award_note = st.text_area("Award comment")
                        inv = st.file_uploader("Upload invoice (optional)", type=["pdf","jpg","png","xlsx","csv"])
                    elif ns == "Lost":
                        lost_note = st.text_area("Lost comment")
                    if st.form_submit_button("Save"):
                        payload = {
                            "lead_id": lead.get("lead_id"),
                            "stage": ns,
                            "assigned_to": assigned,
                            "contacted": 1 if contacted else 0,
                            "inspection_scheduled": 1 if insp_sched else 0,
                            "inspection_completed": 1 if insp_comp else 0,
                            "estimate_submitted": 1 if est_sub else 0,
                            "notes": lead.get("notes"),
                            "estimated_value": float(new_val or 0.0),
                            "ad_cost": float(new_cost or 0.0),
                            "sla_entered_at": lead.get("sla_entered_at") or lead.get("created_at")
                        }
                        save_lead(payload, who=st.session_state.get("profile", {}).get("name","admin"))
                        st.success("Saved"); st.experimental_rerun()
                if st.button("Delete lead", key=f"del_{lead['lead_id']}"):
                    delete_lead(lead.get("lead_id"))
                    st.success("Deleted"); st.experimental_rerun()

# ------------------------------
# Page: Lead Capture
# ------------------------------
elif page == "Lead Capture":
    st.header("📇 Lead Capture")
    st.markdown("<em>Create new leads quickly — SLA Response time must be greater than 0 hours.</em>", unsafe_allow_html=True)
    with st.form("lead_create"):
        lead_id_input = st.text_input("Lead ID (leave empty to auto-generate)")
        source = st.selectbox("Source", ["Google Ads","Organic Search","Referral","Phone","Insurance","Facebook","Instagram","LinkedIn","Other"])
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
                st.error("SLA must be greater than 0 hours.")
            else:
                payload = {
                    "lead_id": lead_id_input.strip() or None,
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
                new_id = save_lead(payload, who=st.session_state.get("profile", {}).get("name","admin"))
                st.success(f"Lead created (ID: {new_id})")
                st.experimental_rerun()

# ------------------------------
# Page: Analytics & SLA
# ------------------------------
elif page == "Analytics & SLA":
    st.header("📈 Analytics & SLA")
    st.markdown("<em>Cost vs Conversions, Pipeline donut, SLA trend and overdue table.</em>", unsafe_allow_html=True)
    df = read_leads(start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    if df.empty:
        st.info("No leads in selected range.")
    else:
        # donut/pie of pipeline stages
        st.subheader("Lead Pipeline Stages (Donut)")
        sc = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
        pie_df = pd.DataFrame({"stage": sc.index, "count": sc.values})
        fig_pie = px.pie(pie_df, names="stage", values="count", hole=0.45, title="Pipeline Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

        # Cost vs conversions by source
        st.subheader("Cost vs Conversions")
        df["won"] = df["stage"].apply(lambda s: 1 if s == "Won" else 0)
        agg = df.groupby("source").agg(total_spend=("ad_cost","sum"), conversions=("won","sum")).reset_index()
        if not agg.empty:
            st.plotly_chart(px.bar(agg, x="source", y=["total_spend","conversions"], barmode="group", title="Total Marketing Spend vs Conversions"), use_container_width=True)
        else:
            st.info("No spend data available.")

        # SLA trend last 30 days
        st.subheader("SLA Overdue Trend (30 days)")
        today = date.today()
        days = [today - timedelta(days=i) for i in range(29, -1, -1)]
        rows = []
        for d in days:
            start_dt = pd.to_datetime(d)
            end_dt = start_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            window = df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)]
            overdue_count = int(window.apply(lambda r: 1 if (is_overdue_sla(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost")) else 0, axis=1).sum())
            rows.append({"date": d, "overdue": overdue_count})
        ts = pd.DataFrame(rows)
        if not ts.empty:
            st.plotly_chart(px.line(ts, x="date", y="overdue", markers=True, title="SLA Overdue Count"), use_container_width=True)

        # overdue table
        st.subheader("Current Overdue Leads")
        overdue_df = df[df.apply(lambda r: is_overdue_sla(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost"), axis=1)]
        if overdue_df.empty:
            st.success("No SLA overdue leads.")
        else:
            st.dataframe(overdue_df[["lead_id","contact_name","stage","estimated_value","ad_cost","sla_hours"]])

# ------------------------------
# Page: CPA & ROI
# ------------------------------
elif page == "CPA & ROI":
    st.header("💰 CPA & ROI")
    df = read_leads(start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    if df.empty:
        st.info("No data.")
    else:
        wins = df[df["stage"]=="Won"]
        total_spend = df["ad_cost"].sum()
        conversions = len(wins)
        cpa = (total_spend / conversions) if conversions else None
        total_value = df["estimated_value"].sum()
        roi = ((total_value - total_spend) / total_spend * 100) if total_spend else None

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div style='background:#111;color:white;padding:12px;border-radius:10px;'>✅ <b>Conversions (Won)</b><div style='font-size:20px;margin-top:8px;color:#2563eb;font-weight:800;'>{conversions}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div style='background:#111;color:white;padding:12px;border-radius:10px;'>🎯 <b>CPA</b><div style='font-size:20px;margin-top:8px;color:#f97316;font-weight:800;'>{format_money(cpa) if cpa else '$0.00'}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div style='background:#111;color:white;padding:12px;border-radius:10px;'>📈 <b>ROI</b><div style='font-size:18px;margin-top:8px;color:#22c55e;font-weight:800;'>{f'{roi:.1f}% ({format_money(total_value)})' if roi is not None else '—'}</div></div>", unsafe_allow_html=True)

        # chart monthly spend vs conversions
        if not df.empty:
            df["month"] = pd.to_datetime(df["created_at"]).dt.to_period("M").astype(str)
            monthly = df.groupby("month").agg(spend=("ad_cost","sum"), conversions=("stage", lambda x: (x=="Won").sum())).reset_index()
            if not monthly.empty:
                st.plotly_chart(px.line(monthly, x="month", y=["spend","conversions"], markers=True, title="Monthly Spend vs Conversions"), use_container_width=True)

# ------------------------------
# Page: Reports (AI observation & summary)
# ------------------------------
elif page == "Reports":
    st.header("🧾 AI Observation & Report Summary")
    df = read_leads()
    if df.empty:
        st.info("No leads to analyze.")
    else:
        insights = []
        bottleneck = df["stage"].value_counts().idxmax() if not df.empty else "—"
        insights.append(f"Most leads are in stage: {bottleneck}")
        top_spend = df.groupby("source")["ad_cost"].sum().sort_values(ascending=False).head(3)
        insights.append(f"Top spend channels: {', '.join(top_spend.index.astype(str))}" if not top_spend.empty else "No spend data")
        overdue = df[df.apply(lambda r: is_overdue_sla(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost"), axis=1)]
        insights.append(f"SLA overdue leads: {len(overdue)}")
        total_spend = df["ad_cost"].sum()
        total_value = df["estimated_value"].sum()
        if total_spend:
            roi = ((total_value - total_spend)/total_spend*100)
            insights.append(f"Current ROI estimate: {roi:.1f}%")
        else:
            insights.append("No spend to calculate ROI")

        st.subheader("Observations")
        for i in insights:
            st.write("- " + i)

        summary = "\n".join(insights)
        st.subheader("Executive Summary")
        st.text_area("Summary", summary, height=200)
        st.download_button("Download summary", summary.encode("utf-8"), file_name="titan_summary.txt", mime="text/plain")

# ------------------------------
# Page: Exports / Imports
# ------------------------------
elif page == "Exports / Imports":
    st.header("📤 Exports & Imports")
    df = read_leads(start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    if df.empty:
        st.info("No leads for the selected range.")
    else:
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="leads_export.csv", mime="text/csv")
        try:
            excel_bytes = df_to_excel_bytes(df)
            st.download_button("Download Excel", excel_bytes, file_name="leads_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            st.info("Excel export failed; CSV available.")

    st.markdown("---")
    uploaded = st.file_uploader("Import CSV (must include lead_id)", type=["csv"])
    if uploaded:
        try:
            imp = pd.read_csv(uploaded)
            if "lead_id" not in imp.columns:
                st.error("CSV must contain 'lead_id' column.")
            else:
                cnt = 0
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
                        "estimated_value": safe_float := (lambda x: float(x) if not pd.isna(x) else 0.0)(r.get("estimated_value")) if True else 0.0,
                        "ad_cost": float(r.get("ad_cost") or 0.0),
                        "stage": r.get("stage") or "New",
                        "sla_hours": int(r.get("sla_hours") or DEFAULT_SLA_HOURS),
                        "sla_entered_at": r.get("sla_entered_at") or now_iso(),
                        "converted": int(r.get("converted") or 0),
                        "score": float(r.get("score") or 0.0)
                    }
                    save_lead(payload, who="import")
                    cnt += 1
                st.success(f"Imported {cnt} leads.")
        except Exception as e:
            st.error("Import failed: " + str(e))

# ------------------------------
# Page: Settings
# ------------------------------
elif page == "Settings":
    st.header("⚙️ Settings & Admin")
    st.markdown("<em>Set your profile and view audit trail.</em>", unsafe_allow_html=True)
    if "profile" not in st.session_state:
        st.session_state.profile = {"name":"", "role":"Admin"}
    name = st.text_input("Your name", value=st.session_state.profile.get("name",""))
    role = st.selectbox("Role", ["Admin","Estimator","Manager","Viewer"], index=["Admin","Estimator","Manager","Viewer"].index(st.session_state.profile.get("role","Admin")))
    if st.button("Save profile"):
        st.session_state.profile["name"] = name
        st.session_state.profile["role"] = role
        st.success("Profile saved to session (used for audit entries).")
    st.markdown("---")
    st.subheader("Audit trail (last 200 changes)")
    conn = get_conn()
    try:
        hist = pd.read_sql_query("SELECT * FROM lead_history ORDER BY timestamp DESC LIMIT 200", conn)
    except Exception:
        hist = pd.DataFrame()
    conn.close()
    if hist.empty:
        st.info("No history yet.")
    else:
        st.dataframe(hist)

# ------------------------------
# Page: ML (internal)
# ------------------------------
elif page == "ML (internal)":
    st.header("🧠 Internal ML — Lead scoring (No user tuning)")
    st.markdown("<em>Train an internal model and persist scores back to leads.</em>", unsafe_allow_html=True)
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn not installed. Install it to use ML features.")
    else:
        if st.button("Train internal model"):
            model, acc_or_msg = train_internal_model()
            if model is None:
                st.error(str(acc_or_msg))
            else:
                st.success(f"Model trained. Approx accuracy: {acc_or_msg:.3f}" if isinstance(acc_or_msg, float) else f"{acc_or_msg}")
        if st.button("Score leads and persist"):
            res = score_and_persist()
            if res:
                st.success(res)
            else:
                st.error("Scoring failed or model missing.")

# ------------------------------
# Default / safety
# ------------------------------
else:
    st.info("Select a page from the left navigation.")

# bottom footer
st.markdown("---")
st.markdown("<div style='font-size:12px;color:#666'>TITAN — single-file backend. Use SQLite for now. For production move DB to managed DB and secure access.</div>", unsafe_allow_html=True)
