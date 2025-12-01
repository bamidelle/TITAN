# titan_singlefile.py
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta, date
import io
import joblib
import plotly.express as px
import os
import math

# -------------------------
# CONFIG
# -------------------------
DB_FILE = "titan_leads.db"
MODEL_FILE = "titan_model.joblib"
PIPELINE_STAGES = ["New", "Contacted", "Inspection Scheduled", "Inspection Completed",
                   "Estimate Submitted", "Qualified", "Won", "Lost"]
DEFAULT_SLA_HOURS = 24

# -------------------------
# DB helpers (sqlite3)
# -------------------------
def get_conn():
    # ensure DB file path exists (in current working dir)
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()
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
        inspection_scheduled INTEGER DEFAULT 0,
        inspection_completed INTEGER DEFAULT 0,
        estimate_submitted INTEGER DEFAULT 0,
        awarded_date TEXT,
        lost_date TEXT,
        qualified INTEGER DEFAULT 0,
        ad_cost REAL DEFAULT 0,
        converted INTEGER DEFAULT 0,
        score REAL
    );
    """, (DEFAULT_SLA_HOURS,))
    # audit/history table
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

# -------------------------
# Utility functions
# -------------------------
def fetch_all_leads(start_date=None, end_date=None):
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM leads ORDER BY created_at DESC", conn, parse_dates=["created_at","sla_entered_at","awarded_date","lost_date"])
    conn.close()
    if df.empty:
        # return empty with expected columns
        cols = ["id","lead_id","created_at","source","source_details","contact_name","contact_phone","contact_email",
                "property_address","damage_type","assigned_to","notes","estimated_value","stage","sla_hours","sla_entered_at",
                "inspection_scheduled","inspection_completed","estimate_submitted","awarded_date","lost_date","qualified","ad_cost","converted","score"]
        return pd.DataFrame(columns=cols)
    # optionally filter
    if start_date:
        df = df[df["created_at"] >= pd.to_datetime(start_date)]
    if end_date:
        # include full day
        df = df[df["created_at"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    return df.reset_index(drop=True)

def upsert_lead(payload: dict, who="admin"):
    conn = get_conn()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    # normalize defaults
    payload = payload.copy()
    payload.setdefault("sla_hours", DEFAULT_SLA_HOURS)
    payload.setdefault("sla_entered_at", now)
    payload.setdefault("estimated_value", 0.0)
    payload.setdefault("ad_cost", 0.0)
    payload.setdefault("stage", "New")
    payload.setdefault("contact_name", "")
    payload.setdefault("notes", "")
    lid = payload.get("lead_id")
    if not lid:
        # generate a simple id
        lid = f"L{int(datetime.utcnow().timestamp())}"
        payload["lead_id"] = lid
    # Insert or update
    c.execute("SELECT * FROM leads WHERE lead_id = ?", (lid,))
    exists = c.fetchone()
    if not exists:
        c.execute("""
            INSERT INTO leads(lead_id, created_at, source, source_details, contact_name, contact_phone, contact_email,
                              property_address, damage_type, assigned_to, notes, estimated_value, stage, sla_hours, sla_entered_at,
                              inspection_scheduled, inspection_completed, estimate_submitted, awarded_date, lost_date, qualified, ad_cost, converted, score)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            payload["lead_id"], payload.get("created_at", now), payload.get("source"), payload.get("source_details"),
            payload.get("contact_name"), payload.get("contact_phone"), payload.get("contact_email"),
            payload.get("property_address"), payload.get("damage_type"), payload.get("assigned_to"),
            payload.get("notes"), float(payload.get("estimated_value") or 0.0), payload.get("stage"),
            int(payload.get("sla_hours") or DEFAULT_SLA_HOURS), payload.get("sla_entered_at"),
            int(bool(payload.get("inspection_scheduled"))), int(bool(payload.get("inspection_completed"))),
            int(bool(payload.get("estimate_submitted"))), payload.get("awarded_date"), payload.get("lost_date"),
            int(bool(payload.get("qualified"))), float(payload.get("ad_cost") or 0.0), int(bool(payload.get("converted"))),
            payload.get("score")
        ))
        c.execute("INSERT INTO lead_history(lead_id, who, field, old_value, new_value, timestamp) VALUES(?,?,?,?,?,?)",
                  (lid, who, "create", "", payload.get("stage"), now))
    else:
        # update only allowed fields in payload
        fields = ["source","source_details","contact_name","contact_phone","contact_email","property_address",
                  "damage_type","assigned_to","notes","estimated_value","stage","sla_hours","sla_entered_at",
                  "inspection_scheduled","inspection_completed","estimate_submitted","awarded_date","lost_date","qualified","ad_cost","converted","score"]
        for f in fields:
            if f in payload:
                # record history
                c.execute(f"SELECT {f} FROM leads WHERE lead_id = ?", (lid,))
                old = c.fetchone()[0]
                new = payload[f]
                if new is None:
                    new = old
                if str(old) != str(new):
                    c.execute(f"UPDATE leads SET {f} = ? WHERE lead_id = ?", (new, lid))
                    c.execute("INSERT INTO lead_history(lead_id, who, field, old_value, new_value, timestamp) VALUES(?,?,?,?,?,?)",
                              (lid, who, f, str(old), str(new), now))
    conn.commit()
    conn.close()
    return lid

def delete_lead(lead_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM leads WHERE lead_id = ?", (lead_id,))
    c.execute("INSERT INTO lead_history(lead_id, who, field, old_value, new_value, timestamp) VALUES(?,?,?,?,?,?)",
              (lead_id, "admin", "delete", "", "", datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

# -------------------------
# SLA & Priority helpers
# -------------------------
def remaining_sla_seconds(sla_entered_iso, sla_hours):
    try:
        if sla_entered_iso is None or pd.isna(sla_entered_iso):
            sla_entered = datetime.utcnow()
        else:
            sla_entered = pd.to_datetime(sla_entered_iso).to_pydatetime()
        deadline = sla_entered + timedelta(hours=int(sla_hours or DEFAULT_SLA_HOURS))
        remaining = deadline - datetime.utcnow()
        return max(0.0, remaining.total_seconds())
    except Exception:
        return float("inf")

def is_overdue(sla_entered_iso, sla_hours):
    return remaining_sla_seconds(sla_entered_iso, sla_hours) <= 0

def compute_priority_score(row):
    # simple composition: normalized estimated_value, inverse remaining SLA, converted flag, score
    try:
        value = float(row.get("estimated_value") or 0.0)
    except:
        value = 0.0
    value_score = min(1.0, value / 5000.0)
    rem_hours = remaining_sla_seconds(row.get("sla_entered_at"), row.get("sla_hours")) / 3600.0
    sla_score = max(0.0, (72.0 - min(rem_hours,72.0)) / 72.0)
    ml_score = float(row.get("score") or 0.0)
    total = 0.6 * ml_score + 0.25 * value_score + 0.15 * sla_score
    # ml_score might be on 0..1 scale or 0..100; normalize if >1
    if total > 1.0:
        total = total / 100.0 if total > 10 else 1.0
    return max(0.0, min(1.0, total))

# -------------------------
# ML helpers (very small internal)
# -------------------------
def train_internal_model():
    df = fetch_all_leads()
    if df.empty or df["converted"].nunique() < 2:
        return None, "Not enough labelled conversion data to train"
    # basic features: estimated_value, ad_cost
    X = df[["estimated_value","ad_cost"]].fillna(0)
    y = df["converted"].fillna(0).astype(int)
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(X, y)
    joblib.dump(m, MODEL_FILE)
    return m, "trained"

def load_internal_model():
    if os.path.exists(MODEL_FILE):
        try:
            m = joblib.load(MODEL_FILE)
            return m
        except Exception:
            return None
    return None

# -------------------------
# UI helpers
# -------------------------
def format_money(v):
    try:
        return f"${float(v):,.2f}"
    except:
        return "$0.00"

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="TITAN — Pipeline", layout="wide", initial_sidebar_state="collapsed")

# top bar: left title, right date pickers and bell
col_left, col_center, col_right = st.columns([3,1,2])
with col_left:
    st.markdown("<h2 style='margin:4px 0'>TITAN — Pipeline</h2>", unsafe_allow_html=True)

# date range + alert bell on top-right
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=29)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()
with col_right:
    sd = st.date_input("Start date", value=st.session_state.start_date, key="top_start")
    ed = st.date_input("End date", value=st.session_state.end_date, key="top_end")
    # store back into session_state to avoid rerun surprises
    st.session_state.start_date = sd
    st.session_state.end_date = ed

# bell with red count
df_all = fetch_all_leads(st.session_state.start_date, st.session_state.end_date)
overdue_count = int(df_all.apply(lambda r: 1 if (is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost")) else 0, axis=1).sum()) if not df_all.empty else 0
bell_html = f"<div style='text-align:right; font-size:18px;'>🔔 <span style='background:#dc2626; color:white; padding:4px 8px; border-radius:12px'>{overdue_count}</span></div>"
with col_right:
    st.markdown(bell_html, unsafe_allow_html=True)

# Navigation (single pipeline page + others)
nav = st.sidebar.radio("Go to", ["Pipeline","Lead Capture","Analytics & SLA","Exports","Settings","Train ML"])

# -------------------------
# Pipeline page (replaces Dashboard)
# -------------------------
if nav == "Pipeline":
    st.markdown("## TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("<em>High-level pipeline performance at a glance. Use filters and cards to drill into details.</em>", unsafe_allow_html=True)

    df = fetch_all_leads(st.session_state.start_date, st.session_state.end_date)

    total_leads = len(df)
    sla_success_count = int(df[df["converted"]==1].shape[0]) if not df.empty else 0
    sla_success_pct = (sla_success_count / total_leads * 100) if total_leads else 0.0
    qualified_leads = int(df[df["qualified"]==1].shape[0]) if not df.empty else 0
    qualification_pct = (qualified_leads / total_leads * 100) if total_leads else 0.0
    awarded_count = int(df[df["stage"]=="Won"].shape[0]) if not df.empty else 0
    lost_count = int(df[df["stage"]=="Lost"].shape[0]) if not df.empty else 0
    closed = awarded_count + lost_count
    conversion_rate = (awarded_count / closed * 100) if closed else 0.0
    inspection_count = int(df[df["inspection_scheduled"]==1].shape[0]) if not df.empty else 0
    inspection_pct = (inspection_count / qualified_leads * 100) if qualified_leads else 0.0
    estimate_sent_count = int(df[df["estimate_submitted"]==1].shape[0]) if not df.empty else 0
    pipeline_job_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
    active_leads = total_leads - (awarded_count + lost_count)

    KPI_ITEMS = [
        ("Active Leads", f"{active_leads}", "#111827", "Leads currently in pipeline"),
        ("SLA Success", f"{sla_success_pct:.1f}%", "#0ea5a4", "Leads contacted within SLA"),
        ("Qualification Rate", f"{qualification_pct:.1f}%", "#a855f7", "Leads marked qualified"),
        ("Conversion Rate", f"{conversion_rate:.1f}%", "#f97316", "Won / Closed"),
        ("Inspections Booked", f"{inspection_pct:.1f}%", "#ef4444", "Qualified → Scheduled"),
        ("Estimates Sent", f"{estimate_sent_count}", "#6d28d9", "Estimates submitted"),
        ("Pipeline Job Value", f"{format_money(pipeline_job_value)}", "#22c55e", "Total pipeline job value")
    ]

    # Top row (4) and bottom row (3) with spacing
    cols_top = st.columns(4)
    for (title, value, color, note), c in zip(KPI_ITEMS[:4], cols_top):
        # simple progress simulation
        try:
            pct = min(100, max(5, int((float(hash(title) % 80) + 20))))
        except:
            pct = 40
        c.markdown(f"""
            <div style='background:#000; color:white; padding:12px; border-radius:10px; text-align:left;'>
              <div style='font-size:12px; opacity:0.9'>{title}</div>
              <div style='font-size:22px; font-weight:800; color:{color};'>{value}</div>
              <div style='height:8px; background:#e6e6e6; border-radius:6px; margin-top:8px;'>
                <div style='width:{pct}%; height:100%; background:{color}; border-radius:6px;'></div>
              </div>
              <div style='font-size:12px; color:#9ca3af; margin-top:6px;'>{note}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    cols_bottom = st.columns(3)
    for (title, value, color, note), c in zip(KPI_ITEMS[4:], cols_bottom):
        try:
            pct = min(100, max(5, int((float(hash(title) % 80) + 20))))
        except:
            pct = 40
        c.markdown(f"""
            <div style='background:#000; color:white; padding:12px; border-radius:10px; text-align:left;'>
              <div style='font-size:12px; opacity:0.9'>{title}</div>
              <div style='font-size:22px; font-weight:800; color:{color};'>{value}</div>
              <div style='height:8px; background:#e6e6e6; border-radius:6px; margin-top:8px;'>
                <div style='width:{pct}%; height:100%; background:{color}; border-radius:6px;'></div>
              </div>
              <div style='font-size:12px; color:#9ca3af; margin-top:6px;'>{note}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # TOP 5 PRIORITY LEADS
    st.markdown("### TOP 5 PRIORITY LEADS")
    st.markdown("<em>Highest urgency leads by priority score. Address these first.</em>", unsafe_allow_html=True)

    if df.empty:
        st.info("No leads yet.")
    else:
        # compute priority score and hours left
        df["priority_score"] = df.apply(lambda r: compute_priority_score(r), axis=1)
        df["hours_left"] = df.apply(lambda r: math.floor(remaining_sla_seconds(r.get("sla_entered_at"), r.get("sla_hours"))/3600.0) if not pd.isna(r.get("sla_hours")) else 9999, axis=1)
        top5 = df.sort_values("priority_score", ascending=False).head(5)
        cols = st.columns(min(5, len(top5)))
        for col, (_, r) in zip(cols, top5.iterrows()):
            score = r.get("priority_score", 0.0)
            if score >= 0.7:
                label = "CRITICAL"
                color = "#ef4444"
            elif score >= 0.45:
                label = "HIGH"
                color = "#f97316"
            else:
                label = "NORMAL"
                color = "#22c55e"
            hours_left = int(r.get("hours_left", 0))
            money = format_money(r.get("estimated_value", 0.0))
            col.markdown(f"""
                <div style='background:#000; padding:12px; border-radius:12px; color:white;'>
                  <div style='font-weight:800; font-size:14px;'>#{r.get("lead_id")} — {r.get("contact_name") or "No name"}</div>
                  <div style='margin-top:6px; color:{color}; font-weight:700'>{label}</div>
                  <div style='margin-top:8px; color:#dc2626; font-weight:700'>⏳ {hours_left}h left</div>
                  <div style='margin-top:8px; color:#22c55e; font-weight:800'>{money}</div>
                  <div style='margin-top:8px; color:#fff;'>Priority: <strong>{score:.2f}</strong></div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # All Leads — searchable
    st.markdown("### All Leads (expand to edit)")
    st.markdown("<em>Search by lead id, contact name, address, or notes.</em>", unsafe_allow_html=True)
    q = st.text_input("Search")
    view_df = df.copy()
    if q:
        qlow = q.lower()
        view_df = view_df[view_df.apply(lambda r: qlow in str(r.get("lead_id","")).lower() or qlow in str(r.get("contact_name","")).lower() or qlow in str(r.get("property_address","")).lower() or qlow in str(r.get("notes","")).lower(), axis=1)]
    if view_df.empty:
        st.info("No leads match.")
    else:
        for _, row in view_df.sort_values("created_at", ascending=False).iterrows():
            lead_label = f"#{row['lead_id']} — {row.get('contact_name') or 'No name'} — {row.get('stage')}"
            with st.expander(lead_label):
                c1, c2 = st.columns([3,1])
                with c1:
                    st.write(f"**Source:** {row.get('source')}  |  **Assigned:** {row.get('assigned_to') or '—'}")
                    st.write(f"**Address:** {row.get('property_address') or '—'}")
                    st.write(f"**Notes:** {row.get('notes') or '—'}")
                    st.write(f"**Created:** {row.get('created_at')}")
                with c2:
                    overdue_flag = is_overdue(row.get("sla_entered_at"), row.get("sla_hours"))
                    if overdue_flag and row.get("stage") not in ("Won","Lost"):
                        st.markdown("<div style='color:#dc2626;font-weight:700;'>❗ OVERDUE</div>", unsafe_allow_html=True)
                    else:
                        rem = remaining_sla_seconds(row.get("sla_entered_at"), row.get("sla_hours"))
                        hrs = int(rem/3600) if rem != float("inf") else None
                        if hrs is None:
                            st.markdown("—")
                        else:
                            st.markdown(f"<div style='color:#111827;font-weight:700;'>⏳ {hrs}h left</div>", unsafe_allow_html=True)

                # inline edit form
                with st.form(f"update_{row['lead_id']}", clear_on_submit=False):
                    new_stage = st.selectbox("Status", PIPELINE_STAGES, index=PIPELINE_STAGES.index(row.get("stage")) if row.get("stage") in PIPELINE_STAGES else 0)
                    new_assigned = st.text_input("Assigned to", value=row.get("assigned_to") or "")
                    new_est = st.number_input("Job Value Estimate (USD)", value=float(row.get("estimated_value") or 0.0), min_value=0.0, step=100.0)
                    new_cost = st.number_input("Cost to acquire lead (USD)", value=float(row.get("ad_cost") or 0.0), min_value=0.0, step=1.0)
                    new_notes = st.text_area("Notes", value=row.get("notes") or "")
                    submitted = st.form_submit_button("Save changes")
                    if submitted:
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
                            st.error("Failed to save: " + str(e))

# -------------------------
# Lead Capture page
# -------------------------
elif nav == "Lead Capture":
    st.header("📇 Lead Capture")
    st.markdown("<em>Create or update a lead. SLA Response time must be greater than 0 hours.</em>", unsafe_allow_html=True)
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
                lid = lead_id.strip() if lead_id else None
                payload = {
                    "lead_id": lid,
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
                new_id = upsert_lead(payload, who="admin")
                st.success(f"Lead {new_id} saved.")
                st.experimental_rerun()

# -------------------------
# Analytics & SLA page
# -------------------------
elif nav == "Analytics & SLA":
    st.header("📈 Analytics & SLA")
    st.markdown("<em>Cost vs Conversions and SLA trends (select date range at top-right).</em>", unsafe_allow_html=True)
    df = fetch_all_leads(st.session_state.start_date, st.session_state.end_date)
    if df.empty:
        st.info("No leads in selected date range.")
    else:
        # Cost vs Conversions: aggregate by source
        agg = df.copy()
        agg["won"] = agg["stage"].apply(lambda s: 1 if s == "Won" else 0)
        agg_src = agg.groupby("source").agg(total_spend=("ad_cost","sum"), conversions=("won","sum")).reset_index()
        if not agg_src.empty:
            fig = px.bar(agg_src, x="source", y=["total_spend","conversions"], barmode="group", title="Total Marketing Spend vs Conversions by Source")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        # SLA Overdue trend (last 30 days)
        st.subheader("SLA Overdue (last 30 days)")
        today = date.today()
        days = [today - timedelta(days=i) for i in range(29, -1, -1)]
        ts_rows = []
        for d in days:
            start_dt = pd.to_datetime(d)
            end_dt = pd.to_datetime(d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df_day = df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)]
            overdue_count = int(df_day.apply(lambda r: 1 if (is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost")) else 0, axis=1).sum()) if not df_day.empty else 0
            ts_rows.append({"date": d, "overdue": overdue_count})
        ts_df = pd.DataFrame(ts_rows)
        fig2 = px.line(ts_df, x="date", y="overdue", markers=True, title="SLA Overdue Count (30d)")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Current Overdue Leads")
        overdue_df = df[df.apply(lambda r: is_overdue(r.get("sla_entered_at"), r.get("sla_hours")) and r.get("stage") not in ("Won","Lost"), axis=1)]
        if overdue_df.empty:
            st.info("No overdue leads currently.")
        else:
            st.dataframe(overdue_df[["lead_id","contact_name","stage","estimated_value","ad_cost","sla_hours"]])

# -------------------------
# Exports
# -------------------------
elif nav == "Exports":
    st.header("📤 Export data")
    df = fetch_all_leads(st.session_state.start_date, st.session_state.end_date)
    if df.empty:
        st.info("No leads to export for selected range.")
    else:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads (CSV)", csv, file_name="leads_export.csv", mime="text/csv")
    st.markdown("---")
    st.subheader("Import CSV (upsert by lead_id)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            imp_df = pd.read_csv(uploaded)
            count=0
            for _, row in imp_df.iterrows():
                payload = {
                    "lead_id": str(row.get("lead_id") or ""),
                    "created_at": pd.to_datetime(row.get("created_at")).isoformat() if not pd.isna(row.get("created_at")) else datetime.utcnow().isoformat(),
                    "source": row.get("source"),
                    "contact_name": row.get("contact_name"),
                    "contact_phone": row.get("contact_phone"),
                    "contact_email": row.get("contact_email"),
                    "property_address": row.get("property_address"),
                    "damage_type": row.get("damage_type"),
                    "assigned_to": row.get("assigned_to"),
                    "notes": row.get("notes"),
                    "estimated_value": float(row.get("estimated_value") or 0.0),
                    "ad_cost": float(row.get("ad_cost") or 0.0),
                    "sla_hours": int(row.get("sla_hours") or DEFAULT_SLA_HOURS),
                    "stage": row.get("stage") or "New",
                    "converted": int(row.get("converted") or 0)
                }
                upsert_lead(payload, who="import")
                count += 1
            st.success(f"Imported/upserted {count} rows.")
        except Exception as e:
            st.error("Import failed: " + str(e))

# -------------------------
# Settings
# -------------------------
elif nav == "Settings":
    st.header("⚙️ Settings & Users")
    st.markdown("<em>Simple internal user: set your name and role for records.</em>", unsafe_allow_html=True)
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_role" not in st.session_state:
        st.session_state.user_role = "Admin"
    st.session_state.user_name = st.text_input("Your name", value=st.session_state.user_name)
    st.session_state.user_role = st.selectbox("Role", ["Admin","Editor","Viewer"], index=["Admin","Editor","Viewer"].index(st.session_state.user_role) if st.session_state.user_role in ["Admin","Editor","Viewer"] else 0)
    if st.button("Save profile"):
        st.success("Profile saved in session.")

# -------------------------
# Train ML (small)
# -------------------------
elif nav == "Train ML":
    st.header("🧠 Train internal ML model")
    st.markdown("<em>Trains small RandomForest on estimated_value & ad_cost to predict conversions (internal use).</em>", unsafe_allow_html=True)
    df = fetch_all_leads()
    st.write(f"Total rows: {len(df)}")
    if st.button("Train now"):
        m, msg = train_internal_model()
        if m is None:
            st.error(msg)
        else:
            st.success("Model trained and saved (internal)")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("<div style='font-size:12px;color:#444'>TITAN - Single-file backend (SQLite). Exports CSV to avoid openpyxl issues.</div>", unsafe_allow_html=True)
