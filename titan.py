# titan_full_app.py
# TITAN — Single-file Streamlit backend for Restoration Pipeline
# Uses sqlite3 + pandas. Pages: Pipeline, Lead Capture, Analytics & SLA, Exports, Settings, ML (internal)
# Author: ChatGPT (assistant) — delivered as requested.

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta, date
import io
import os
import joblib
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import math
import typing

# -----------------------
# Config
# -----------------------
DB_FILE = "titan_leads.db"
MODEL_FILE = "titan_model.joblib"
DEFAULT_SLA_HOURS = 24
PIPELINE_STAGES = [
    "New", "Contacted", "Inspection Scheduled", "Inspection Completed",
    "Estimate Submitted", "Qualified", "Won", "Lost"
]

# -----------------------
# Initialize DB (sqlite3)
# -----------------------
def get_conn():
    # connect to local sqlite DB file
    # use check_same_thread False for Streamlit multi-thread safety
    return sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES, check_same_thread=False)

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
        sla_hours INTEGER DEFAULT 24,
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
        score REAL
    );
    """)
    # history/audit table
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
    conn.commit()
    conn.close()

init_db()

# -----------------------
# Utility functions
# -----------------------
def _to_df(conn_res) -> pd.DataFrame:
    try:
        df = pd.DataFrame(conn_res)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=10)
def fetch_all_leads(start_date: typing.Optional[date] = None, end_date: typing.Optional[date] = None) -> pd.DataFrame:
    conn = get_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM leads ORDER BY created_at DESC", conn, parse_dates=["created_at","sla_entered_at","inspection_scheduled_at","estimate_submitted_at","awarded_date","lost_date"])
    except Exception:
        df = pd.DataFrame()
    conn.close()
    if df.empty:
        # ensure consistent columns
        cols = ["id","lead_id","created_at","source","source_details","contact_name","contact_phone","contact_email",
                "property_address","damage_type","assigned_to","notes","estimated_value","stage","sla_hours","sla_entered_at",
                "contacted","inspection_scheduled","inspection_scheduled_at","inspection_completed","estimate_submitted",
                "estimate_submitted_at","awarded_date","awarded_invoice","lost_date","qualified","ad_cost","converted","score"]
        return pd.DataFrame(columns=cols)
    # filter by provided date range
    if start_date:
        df = df[df["created_at"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["created_at"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    # ensure types
    for col in ["estimated_value","ad_cost","score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ["converted","qualified","contacted","inspection_scheduled","inspection_completed","estimate_submitted"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return df.reset_index(drop=True)

def upsert_lead(payload: dict, who: str = "admin") -> str:
    """
    Insert or update lead by lead_id. Returns lead_id.
    Payload keys should be columns names such as lead_id, source, contact_name, estimated_value, sla_hours, ad_cost, etc.
    """
    conn = get_conn()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    # defaults
    payload = payload.copy()
    if "sla_hours" not in payload or payload.get("sla_hours") in (None, ""):
        payload["sla_hours"] = DEFAULT_SLA_HOURS
    if "sla_entered_at" not in payload or not payload.get("sla_entered_at"):
        payload["sla_entered_at"] = now
    lid = payload.get("lead_id")
    if not lid:
        # generate a deterministic id
        lid = f"L{int(datetime.utcnow().timestamp())}"
        payload["lead_id"] = lid
    # check exists
    c.execute("SELECT * FROM leads WHERE lead_id = ?", (lid,))
    row = c.fetchone()
    if not row:
        # insert all expected fields (safe mapping)
        c.execute("""
            INSERT INTO leads(
                lead_id, created_at, source, source_details, contact_name, contact_phone, contact_email,
                property_address, damage_type, assigned_to, notes, estimated_value, stage, sla_hours, sla_entered_at,
                contacted, inspection_scheduled, inspection_scheduled_at, inspection_completed, estimate_submitted,
                estimate_submitted_at, awarded_date, awarded_invoice, lost_date, qualified, ad_cost, converted, score
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            payload.get("lead_id"),
            payload.get("created_at", now),
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
            payload.get("stage") or "New",
            int(payload.get("sla_hours") or DEFAULT_SLA_HOURS),
            payload.get("sla_entered_at"),
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
            float(payload.get("ad_cost") or 0.0),
            int(bool(payload.get("converted"))),
            float(payload.get("score") or 0.0)
        ))
        c.execute("INSERT INTO lead_history(lead_id, who, field, old_value, new_value, timestamp) VALUES(?,?,?,?,?,?)",
                  (lid, who, "create", "", payload.get("stage") or "New", now))
    else:
        # update only fields present in payload
        allowed = ["source","source_details","contact_name","contact_phone","contact_email","property_address","damage_type",
                   "assigned_to","notes","estimated_value","stage","sla_hours","sla_entered_at","contacted","inspection_scheduled",
                   "inspection_scheduled_at","inspection_completed","estimate_submitted","estimate_submitted_at","awarded_date",
                   "awarded_invoice","lost_date","qualified","ad_cost","converted","score"]
        for key in allowed:
            if key in payload:
                c.execute(f"SELECT {key} FROM leads WHERE lead_id = ?", (lid,))
                try:
                    old = c.fetchone()[0]
                except Exception:
                    old = None
                new = payload[key]
                # compare as strings to record meaningful change
                if str(old) != str(new):
                    c.execute(f"UPDATE leads SET {key} = ? WHERE lead_id = ?", (new, lid))
                    c.execute("INSERT INTO lead_history(lead_id, who, field, old_value, new_value, timestamp) VALUES(?,?,?,?,?,?)",
                              (lid, who, key, str(old), str(new), now))
    conn.commit()
    conn.close()
    return lid

def delete_lead(lead_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM leads WHERE lead_id = ?", (lead_id,))
    c.execute("INSERT INTO lead_history(lead_id, who, field, old_value, new_value, timestamp) VALUES(?,?,?,?,?,?)",
              (lead_id, "admin", "delete", "", "", datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

# -----------------------
# SLA and priority helpers
# -----------------------
def remaining_sla_seconds(sla_entered_iso, sla_hours):
    try:
        if not sla_entered_iso:
            sla_entered = datetime.utcnow()
        else:
            sla_entered = pd.to_datetime(sla_entered_iso).to_pydatetime()
        deadline = sla_entered + timedelta(hours=int(sla_hours or DEFAULT_SLA_HOURS))
        diff = deadline - datetime.utcnow()
        return max(diff.total_seconds(), 0.0)
    except Exception:
        return float("inf")

def is_overdue(sla_entered_iso, sla_hours):
    rem = remaining_sla_seconds(sla_entered_iso, sla_hours)
    return rem <= 0

def compute_priority_score(row):
    # Compose from ML score (score 0..1), estimated value, SLA urgency
    try:
        ml = float(row.get("score") or 0.0)
        if ml > 1.0:
            ml = ml / 100.0
    except:
        ml = 0.0
    try:
        val = float(row.get("estimated_value") or 0.0)
    except:
        val = 0.0
    val_norm = min(1.0, val / 5000.0)
    try:
        sla_hours = int(row.get("sla_hours") or DEFAULT_SLA_HOURS)
        rem_h = remaining_sla_seconds(row.get("sla_entered_at"), sla_hours) / 3600.0
        sla_score = max(0.0, (72.0 - min(rem_h, 72.0)) / 72.0)
    except:
        sla_score = 0.0
    score = 0.6 * ml + 0.25 * val_norm + 0.15 * sla_score
    return max(0.0, min(1.0, score))

# -----------------------
# ML (simple internal)
# -----------------------
def train_internal_model():
    df = fetch_all_leads()
    if df.empty or df["converted"].nunique() < 2:
        return None, "Not enough labeled conversion data to train model."
    df2 = df.copy()
    df2["created_at"] = pd.to_datetime(df2["created_at"], errors="coerce")
    df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
    X = df2[["estimated_value", "ad_cost"]].fillna(0)
    y = df2["converted"].fillna(0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    acc = model.score(X_test, y_test)
    return model, f"Trained. approx accuracy: {acc:.3f}"

def load_internal_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None

def score_dataframe_with_model(df: pd.DataFrame, model):
    if model is None or df.empty:
        df["score"] = 0.0
        return df
    df2 = df.copy()
    df2["estimated_value"] = pd.to_numeric(df2["estimated_value"].fillna(0))
    df2["ad_cost"] = pd.to_numeric(df2["ad_cost"].fillna(0))
    X = df2[["estimated_value", "ad_cost"]]
    try:
        preds = model.predict_proba(X)[:, 1]
    except Exception:
        preds = model.predict(X)
        # normalize if binary
        if preds.max() > 1:
            preds = preds / preds.max()
    df["score"] = preds
    return df

# -----------------------
# UI helpers
# -----------------------
def format_money(x):
    try:
        return f"${float(x):,.2f}"
    except:
        return "$0.00"

# -----------------------
# Streamlit UI config
# -----------------------
st.set_page_config(page_title="TITAN — Pipeline", layout="wide")
# small CSS
st.markdown("""
    <style>
      body {font-family: 'Segoe UI', Roboto, Helvetica, Arial; }
      .kpi-card { background:#0b1220; color:white; border-radius:10px; padding:12px; min-height:88px; }
      .kpi-title { font-size:12px; color: #d1d5db; margin-bottom:6px; }
      .kpi-value { font-size:20px; font-weight:800; margin-bottom:6px; }
      .priority-card { background:#000; color:#fff; padding:12px; border-radius:12px; min-width:200px; }
      .small-muted { color:#6b7280; font-size:12px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Top bar: title (left), date range + bell (right)
# -----------------------
top_left, _spacer, top_right = st.columns([3,1,2])
with top_left:
    st.markdown("<h2 style='margin:6px 0'>TITAN — Lead Pipeline</h2>", unsafe_allow_html=True)

# date range inputs & bell on top-right
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=29)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

with top_right:
    sd = st.date_input("Start date", st.session_state.start_date, key="top_sd")
    ed = st.date_input("End date", st.session_state.end_date, key="top_ed")
    # store
    st.session_state.start_date = sd
    st.session_state.end_date = ed

# bell with red count
_leads_for_top = fetch_all_leads(st.session_state.start_date, st.session_state.end_date)
def compute_overdue_count(df):
    if df.empty:
        return 0
    return int(df.apply(lambda r: 1 if (is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost")) else 0, axis=1).sum())

overdue_count = compute_overdue_count(_leads_for_top)
with top_right:
    st.markdown(f"<div style='text-align:right; font-size:16px'>🔔 <span style='background:#dc2626;color:white;padding:4px 8px;border-radius:12px'>{overdue_count}</span></div>", unsafe_allow_html=True)

# -----------------------
# Sidebar navigation
# -----------------------
page = st.sidebar.radio("Navigate", ["Pipeline", "Lead Capture", "Analytics & SLA", "Exports", "Settings", "ML (internal)"])

# -----------------------
# Pages
# -----------------------
if page == "Pipeline":
    st.markdown("## TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("<em>High-level pipeline performance at a glance. Use search and filters to drill into details.</em>", unsafe_allow_html=True)

    df = fetch_all_leads(st.session_state.start_date, st.session_state.end_date)

    # (optionally) apply model scoring
    model = load_internal_model()
    if model is not None and not df.empty:
        try:
            df = score_dataframe_with_model(df.copy(), model)
        except Exception:
            pass

    total_leads = len(df)
    qualified_leads = int(df[df["qualified"] == 1].shape[0]) if not df.empty else 0
    sla_success_count = int(df[df["contacted"] == 1].shape[0]) if not df.empty else 0
    sla_success_pct = (sla_success_count / total_leads * 100) if total_leads else 0.0
    qualification_pct = (qualified_leads / total_leads * 100) if total_leads else 0.0
    awarded_count = int(df[df["stage"] == "Won"].shape[0]) if not df.empty else 0
    lost_count = int(df[df["stage"] == "Lost"].shape[0]) if not df.empty else 0
    closed = awarded_count + lost_count
    conversion_rate = (awarded_count / closed * 100) if closed else 0.0
    inspection_count = int(df[df["inspection_scheduled"] == 1].shape[0]) if not df.empty else 0
    inspection_pct = (inspection_count / qualified_leads * 100) if qualified_leads else 0.0
    estimate_sent_count = int(df[df["estimate_submitted"] == 1].shape[0]) if not df.empty else 0
    pipeline_job_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
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

    # top row 4 cards
    cols_top = st.columns(4)
    for (title, value, color, note), col in zip(KPI_ITEMS[:4], cols_top):
        pct = 50
        col.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-title'>{title}</div>
              <div class='kpi-value' style='color:{color}; font-size:20px; font-weight:800'>{value}</div>
              <div style='height:8px; background:#e6e6e6; border-radius:6px; margin-top:8px;'>
                <div style='width:{pct}%; height:100%; background:{color}; border-radius:6px;'></div>
              </div>
              <div class='small-muted'>{note}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    cols_bottom = st.columns(3)
    for (title, value, color, note), col in zip(KPI_ITEMS[4:], cols_bottom):
        pct = 45
        col.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-title'>{title}</div>
              <div class='kpi-value' style='color:{color}; font-size:20px; font-weight:800'>{value}</div>
              <div style='height:8px; background:#e6e6e6; border-radius:6px; margin-top:8px;'>
                <div style='width:{pct}%; height:100%; background:{color}; border-radius:6px;'></div>
              </div>
              <div class='small-muted'>{note}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Top 5 priority leads
    st.markdown("### TOP 5 PRIORITY LEADS")
    st.markdown("<em>Highest urgency leads by priority score (0–1). Address these first.</em>", unsafe_allow_html=True)
    if df.empty:
        st.info("No leads yet.")
    else:
        df["priority_score"] = df.apply(lambda r: compute_priority_score(r), axis=1)
        df["hours_left"] = df.apply(lambda r: int(remaining_sla_seconds(r.get("sla_entered_at"), r.get("sla_hours"))/3600.0) if not pd.isna(r.get("sla_hours")) else 9999, axis=1)
        top5 = df.sort_values("priority_score", ascending=False).head(5)
        cols = st.columns(min(5, len(top5)))
        for col, (_, r) in zip(cols, top5.iterrows()):
            score = r.get("priority_score", 0.0)
            if score >= 0.7:
                label = "CRITICAL"
                label_color = "#ef4444"
            elif score >= 0.45:
                label = "HIGH"
                label_color = "#f97316"
            else:
                label = "NORMAL"
                label_color = "#22c55e"
            hours_left = int(r.get("hours_left", 0))
            money = format_money(r.get("estimated_value", 0.0))
            col.markdown(f"""
                <div class='priority-card'>
                  <div style='font-weight:800; font-size:14px;'>#{r.get('lead_id')} — {r.get('contact_name') or 'No name'}</div>
                  <div style='margin-top:6px; color:{label_color}; font-weight:700'>{label}</div>
                  <div style='margin-top:8px; color:#dc2626; font-weight:700'>⏳ {hours_left}h left</div>
                  <div style='margin-top:8px; color:#22c55e; font-weight:800'>{money}</div>
                  <div style='margin-top:8px; color:#fff;'>Priority: <strong>{score:.2f}</strong></div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### All Leads (search & edit)")
    q = st.text_input("Search by lead id, contact, address or notes")
    view_df = df.copy()
    if q:
        ql = q.lower()
        view_df = view_df[view_df.apply(lambda r: ql in str(r.get("lead_id","")).lower() or ql in str(r.get("contact_name","")).lower() or ql in str(r.get("property_address","")).lower() or ql in str(r.get("notes","")).lower(), axis=1)]
    if view_df.empty:
        st.info("No leads to display.")
    else:
        for _, row in view_df.sort_values("created_at", ascending=False).iterrows():
            exp_label = f"#{row.get('lead_id')} — {row.get('contact_name') or 'No name'} — {row.get('stage')}"
            with st.expander(exp_label):
                left, right = st.columns([3,1])
                with left:
                    st.write(f"**Source:** {row.get('source') or '—'}  |  **Assigned:** {row.get('assigned_to') or '—'}")
                    st.write(f"**Address:** {row.get('property_address') or '—'}")
                    st.write(f"**Contact:** {row.get('contact_name') or ''} / {row.get('contact_phone') or ''} / {row.get('contact_email') or ''}")
                    st.write(f"**Notes:** {row.get('notes') or '—'}")
                with right:
                    overdue_flag = is_overdue(row.get("sla_entered_at"), row.get("sla_hours"))
                    if overdue_flag and row.get("stage") not in ("Won","Lost"):
                        st.markdown("<div style='color:#dc2626;font-weight:700;'>❗ OVERDUE</div>", unsafe_allow_html=True)
                    else:
                        rem = remaining_sla_seconds(row.get("sla_entered_at"), row.get("sla_hours"))
                        hrs = int(rem/3600) if rem != float("inf") else None
                        if hrs is not None:
                            st.markdown(f"<div class='small-muted'>⏳ {hrs}h left</div>", unsafe_allow_html=True)
                # inline edit form
                with st.form(f"up_{row.get('lead_id')}", clear_on_submit=False):
                    new_stage = st.selectbox("Status", PIPELINE_STAGES, index=PIPELINE_STAGES.index(row.get("stage")) if row.get("stage") in PIPELINE_STAGES else 0)
                    new_assigned = st.text_input("Assigned to", value=row.get("assigned_to") or "")
                    new_est = st.number_input("Estimated value (USD)", value=float(row.get("estimated_value") or 0.0), min_value=0.0, step=100.0)
                    new_cost = st.number_input("Cost to acquire lead (USD)", value=float(row.get("ad_cost") or 0.0), min_value=0.0, step=1.0)
                    new_notes = st.text_area("Notes", value=row.get("notes") or "")
                    saveb = st.form_submit_button("Save changes")
                    if saveb:
                        try:
                            upsert_lead({
                                "lead_id": row.get("lead_id"),
                                "stage": new_stage,
                                "assigned_to": new_assigned or None,
                                "estimated_value": float(new_est or 0.0),
                                "ad_cost": float(new_cost or 0.0),
                                "notes": new_notes
                            }, who="admin")
                            st.success("Lead updated")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error("Failed to update lead: " + str(e))

# Lead Capture page
elif page == "Lead Capture":
    st.markdown("## Lead Capture")
    st.markdown("<em>SLA Response time must be greater than 0 hours.</em>", unsafe_allow_html=True)
    with st.form("lead_form", clear_on_submit=True):
        lead_id = st.text_input("Lead ID (leave blank to auto-generate)")
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
        submitted = st.form_submit_button("Create / Update Lead")
        if submitted:
            if sla_hours <= 0:
                st.error("SLA must be greater than 0 hours.")
            else:
                payload = {
                    "lead_id": lead_id.strip() if lead_id else None,
                    "created_at": datetime.utcnow().isoformat(),
                    "source": source,
                    "source_details": source_details,
                    "contact_name": contact_name,
                    "contact_phone": contact_phone,
                    "contact_email": contact_email,
                    "property_address": property_address,
                    "damage_type": damage_type,
                    "assigned_to": assigned_to,
                    "estimated_value": float(estimated_value or 0.0),
                    "ad_cost": float(ad_cost or 0.0),
                    "sla_hours": int(sla_hours),
                    "sla_entered_at": datetime.utcnow().isoformat(),
                    "notes": notes
                }
                lid = upsert_lead(payload, who="admin")
                st.success(f"Lead {lid} saved.")
                st.experimental_rerun()

# Analytics & SLA
elif page == "Analytics & SLA":
    st.markdown("## Analytics & SLA")
    st.markdown("<em>Cost vs Conversions + SLA overdue trends and table.</em>", unsafe_allow_html=True)
    df = fetch_all_leads(st.session_state.start_date, st.session_state.end_date)
    if df.empty:
        st.info("No leads to analyze for selected date range.")
    else:
        agg = df.copy()
        agg["won"] = agg["stage"].apply(lambda s: 1 if s == "Won" else 0)
        agg_src = agg.groupby("source").agg(total_spend=("ad_cost","sum"), conversions=("won","sum")).reset_index()
        if not agg_src.empty:
            fig = px.bar(agg_src, x="source", y=["total_spend","conversions"], barmode="group", title="Total Marketing Spend vs Conversions by Source")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("SLA Overdue (last 30 days)")
        today = date.today()
        days = [today - timedelta(days=i) for i in range(29, -1, -1)]
        ts = []
        for d in days:
            start_dt = pd.to_datetime(d)
            end_dt = pd.to_datetime(d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df_day = df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)]
            overdue_cnt = int(df_day.apply(lambda r: 1 if (is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost")) else 0, axis=1).sum()) if not df_day.empty else 0
            ts.append({"date": d, "overdue": overdue_cnt})
        ts_df = pd.DataFrame(ts)
        fig2 = px.line(ts_df, x="date", y="overdue", markers=True, title="SLA Overdue Count (30d)")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Current Overdue Leads")
        overdue_df = df[df.apply(lambda r: is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost"), axis=1)]
        if overdue_df.empty:
            st.info("No overdue leads currently.")
        else:
            st.dataframe(overdue_df[["lead_id","contact_name","stage","estimated_value","ad_cost","sla_hours"]])

# Exports
elif page == "Exports":
    st.markdown("## Exports & Imports")
    st.markdown("<em>Export CSV for portability. Import CSV to upsert by lead_id.</em>", unsafe_allow_html=True)
    df = fetch_all_leads(st.session_state.start_date, st.session_state.end_date)
    if df.empty:
        st.info("No leads for selected range.")
    else:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads (CSV)", csv_bytes, file_name="leads_export.csv", mime="text/csv")
    st.markdown("---")
    uploaded = st.file_uploader("Import CSV (must include lead_id column)", type=["csv"])
    if uploaded:
        try:
            imp = pd.read_csv(uploaded)
            if "lead_id" not in imp.columns:
                st.error("CSV must contain lead_id column.")
            else:
                count = 0
                for _, r in imp.iterrows():
                    payload = {
                        "lead_id": str(r.get("lead_id") or ""),
                        "created_at": pd.to_datetime(r.get("created_at")).isoformat() if not pd.isna(r.get("created_at")) else datetime.utcnow().isoformat(),
                        "source": r.get("source"),
                        "contact_name": r.get("contact_name"),
                        "contact_phone": r.get("contact_phone"),
                        "contact_email": r.get("contact_email"),
                        "property_address": r.get("property_address"),
                        "damage_type": r.get("damage_type"),
                        "assigned_to": r.get("assigned_to"),
                        "notes": r.get("notes"),
                        "estimated_value": float(r.get("estimated_value") or 0.0),
                        "ad_cost": float(r.get("ad_cost") or 0.0),
                        "sla_hours": int(r.get("sla_hours") or DEFAULT_SLA_HOURS),
                        "stage": r.get("stage") or "New",
                        "converted": int(r.get("converted") or 0)
                    }
                    upsert_lead(payload, who="import")
                    count += 1
                st.success(f"Imported/upserted {count} rows.")
        except Exception as e:
            st.error("Import failed: " + str(e))

# Settings
elif page == "Settings":
    st.markdown("## Settings & User (internal)")
    st.markdown("<em>Simple profile and role stored in session for audit/tracking.</em>", unsafe_allow_html=True)
    if "profile" not in st.session_state:
        st.session_state.profile = {"name": "", "role": "Admin"}
    name = st.text_input("Your name", value=st.session_state.profile.get("name",""))
    role = st.selectbox("Your role", ["Admin","Estimator","Manager","Viewer"], index=["Admin","Estimator","Manager","Viewer"].index(st.session_state.profile.get("role","Admin")))
    if st.button("Save profile"):
        st.session_state.profile["name"] = name
        st.session_state.profile["role"] = role
        st.success("Profile saved to session.")

    st.markdown("---")
    st.subheader("Audit trail (recent 50 events)")
    conn = get_conn()
    hist = pd.read_sql_query("SELECT * FROM lead_history ORDER BY timestamp DESC LIMIT 50", conn)
    conn.close()
    if hist.empty:
        st.info("No history recorded yet.")
    else:
        st.dataframe(hist)

# ML (internal)
elif page == "ML (internal)":
    st.markdown("## ML — Internal Lead Scoring")
    st.markdown("<em>Small internal model that predicts conversion from estimated value & ad_cost. No user-tunable params.</em>", unsafe_allow_html=True)
    df = fetch_all_leads()
    st.write(f"Total rows in DB: {len(df)}")
    model_exists = os.path.exists(MODEL_FILE)
    if model_exists:
        st.success("Internal ML model available.")
        if st.button("Score leads and persist scores"):
            model = load_internal_model()
            if model is None:
                st.error("Failed to load model.")
            else:
                scored = score_dataframe_with_model(df.copy(), model)
                conn = get_conn(); c = conn.cursor()
                for _, r in scored.iterrows():
                    try:
                        c.execute("UPDATE leads SET score = ? WHERE lead_id = ?", (float(r.get("score") or 0.0), r.get("lead_id")))
                    except:
                        continue
                conn.commit(); conn.close()
                st.success("Scores saved back to DB.")
    else:
        st.info("No trained model found yet. Train with the button below when you have labeled data.")
        if st.button("Train internal model"):
            model, msg = train_internal_model()
            if model is None:
                st.error(msg)
            else:
                st.success("Model trained and saved (internal).")

# Footer
st.markdown("---")
st.markdown("<div style='font-size:12px;color:#444'>TITAN — Single-file backend. Uses SQLite for persistence. Exports CSV to avoid openpyxl issues.</div>", unsafe_allow_html=True)
