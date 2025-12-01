# titan_full_app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta, date
import io, os, math, subprocess, sys
import joblib
import plotly.express as px

# -----------------------
# Config
# -----------------------
DB_FILE = "titan_leads.db"
MODEL_FILE = "titan_model.joblib"
PIPELINE_STAGES = ["New","Contacted","Qualified","Estimate Sent","Inspection Scheduled","Inspection Completed","Won","Lost"]

# -----------------------
# Initialize SQLite
# -----------------------
def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS leads(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id TEXT UNIQUE,
        created_at TEXT,
        source TEXT,
        stage TEXT,
        estimated_value REAL,
        ad_cost REAL,
        converted INTEGER,
        notes TEXT,
        sla_hours INTEGER,
        sla_entered_at TEXT,
        score REAL
    );
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------
# ML scoring
# -----------------------
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

def train_model():
    df = fetch_leads()
    if df["converted"].nunique() < 2:
        return "Not enough labeled converted data to train."
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["age_days"] = (datetime.utcnow() - df["created_at"]).dt.days
    X = pd.get_dummies(df[["source","stage"]])
    X["ad_cost"] = df["ad_cost"]
    X["estimated_value"] = df["estimated_value"]
    X["age_days"] = df["age_days"]
    y = df["converted"]
    Xt,Xe,yt,ye = train_test_split(X,y,test_size=.2,random_state=42)
    model = RandomForestClassifier(120,random_state=42)
    model.fit(Xt,yt)
    joblib.dump(model,MODEL_FILE)
    acc = model.score(Xe,ye)
    return f"Model trained. Accuracy ~{acc:.3f}"

def predict_score(df):
    model = load_model()
    if not model:
        df["score"] = 0.5
        return df
    X = pd.get_dummies(df[["source","stage"]])
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    X["estimated_value"] = df["estimated_value"]
    X["ad_cost"] = df["ad_cost"]
    scores = model.predict_proba(X)[:,1]
    df["score"] = scores
    return df

# -----------------------
# Fetch leads
# -----------------------
def fetch_leads():
    conn = get_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM leads ORDER BY created_at DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=["lead_id","created_at","source","stage","estimated_value","ad_cost","converted","notes","sla_hours","sla_entered_at","score"])
    df["estimated_value"] = pd.to_numeric(df["estimated_value"], errors="coerce").fillna(0)
    df["ad_cost"] = pd.to_numeric(df["ad_cost"], errors="coerce").fillna(0)
    df["converted"] = df["converted"].fillna(0).astype(int)
    df["sla_hours"] = df["sla_hours"].fillna(24).astype(int)
    df["score"] = df["score"].fillna(0.5)
    if load_model():
        df = predict_score(df)
    return df.reset_index(drop=True)

# -----------------------
# UI Top-right bar
# -----------------------
st.set_page_config(layout="wide")

if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=30)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

top_l,_,top_r = st.columns([3,1,2])
with top_l:
    st.markdown("<h2 style='color:#0ea5e9'>Titanic — Lead Pipeline System</h2>", unsafe_allow_html=True)
with top_r:
    sd = st.date_input("Start Date", st.session_state.start_date, key="sd")
    ed = st.date_input("End Date", st.session_state.end_date, key="ed")
    st.session_state.start_date = sd
    st.session_state.end_date = ed

df_top = fetch_leads()
def is_overdue(ts, hours):
    try:
        entered = pd.to_datetime(ts).to_pydatetime()
        return datetime.utcnow() > entered + timedelta(hours=hours)
    except:
        return False

count_not = df_top[df_top.apply(lambda r: is_overdue(r["sla_entered_at"], r["sla_hours"]) and r["stage"] not in ["Won","Lost"], axis=1)].shape[0]
with top_r:
    st.markdown(f"<div style='text-align:right'>🔔 <span style='background:red;color:white;padding:4px 9px;border-radius:13px'>{count_not}</span></div>", unsafe_allow_html=True)

# -----------------------
# Insert/Update/Delete
# -----------------------
def save_lead(row, who="admin"):
    conn = get_conn(); c = conn.cursor()
    iso = row["sla_entered_at"]
    c.execute("SELECT stage FROM leads WHERE lead_id=?", (row["lead_id"],))
    ex = c.fetchone()
    if not ex:
        c.execute("""INSERT OR IGNORE INTO leads(lead_id,created_at,source,stage,estimated_value,ad_cost,converted,notes,sla_hours,sla_entered_at,score)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                     (row["lead_id"], row["created_at"], row["source"], row["stage"], row["estimated_value"], row["ad_cost"], row["converted"], row["notes"], row["sla_hours"], iso, row["score"]))
    else:
        if ex[0] != row["stage"]:
            c.execute("UPDATE leads SET stage=? WHERE lead_id=?", (row["stage"], row["lead_id"]))
    conn.commit(); conn.close()

def remove_lead(lid):
    conn = get_conn(); c = conn.cursor()
    c.execute("DELETE FROM leads WHERE lead_id=?", (lid,))
    conn.commit(); conn.close()

# -----------------------
# Page 1 — Pipeline KPI
# -----------------------
def page_pipeline():
    st.markdown("## Total Lead Pipeline KPI")
    df = fetch_leads()
    total = len(df)
    qual = int(df[df["stage"].isin(["Qualified","Estimate Sent","Won"])].shape[0])
    won = int(df[df["stage"]=="Won"].shape[0])
    lost = int(df[df["stage"]=="Lost"].shape[0])
    closed = won + lost
    conv = (won/closed*100) if closed else 0
    spend = df["ad_cost"].sum()
    cpa = (spend/won) if won else 0
    roi = ((df["estimated_value"].sum() - spend) / spend * 100) if spend else 0
    active = total - closed

    row1 = st.columns(4, gap="large")
    row2 = st.columns(4, gap="large")

    KPI = [
        ("Active Leads",active,"leads still open"),
        ("Conversion Rate",f"{conv:.1f}%","Won vs Closed"),
        ("Total Spend",f"${spend:,.2f}","marketing spend"),
        ("Average CPA",f"${cpa:,.2f}","Cost per won lead"),
        ("ROI",f"{roi:.1f}%","profit/spend"),
        ("Qualified Leads",qual,"ready for estimate"),
        ("Won Leads",won,"converted"),
        ("Lost Leads",lost,"not converted"),
    ]

    for i,(t,v,note) in enumerate(KPI):
        target = (v/total*100) if isinstance(v,(int,float)) and total else 40
        card = f"""
        <div class='kpi-card'>
            <div class='kpi-title'>{t}</div>
            <div class='kpi-value'>{v}</div>
            <div style='height:7px;border-radius:7px;background:#333'><div style='height:100%;width:{target}%;border-radius:7px;background:#0ea5e9'></div></div>
            <div class='small-muted'>{note}</div>
        </div>
        """
        if i < 4:
            row1[i].markdown(card, unsafe_allow_html=True)
        else:
            row2[i-4].markdown(card, unsafe_allow_html=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True) # spacing between rows

    # TOP PRIORITY LEADS
    st.markdown("### 🚨 Top 5 Priority Leads")
    df["priority_score"] = df.apply(lambda r: (r["score"]*0.7 + (r["estimated_value"]/5000)*0.2 + (1 if is_overdue(r["sla_entered_at"],r["sla_hours"]) else 0)*0.1), axis=1)
    top5 = df.sort_values("priority_score", ascending=False).head(5)
    if top5.empty:
        st.info("No priority leads yet.")
    else:
        cols = st.columns(len(top5))
        for col,(_,r) in zip(cols,top5.iterrows()):
            ps = r["priority_score"]
            if ps >= 0.7:
                urg = "CRITICAL"; uc="#ef4444"
            elif ps>=0.4:
                urg="HIGH"; uc="#f97316"
            else:
                urg="NORMAL"; uc="#22c55e"
            rem = remaining_sla_seconds(r["sla_entered_at"],r["sla_hours"])/3600
            col.markdown(f"""
            <div class='priority-card'>
                <div style='font-size:15px;font-weight:700'>#{r["lead_id"] or r["id"]}</div>
                <div style='color:white'>{r["contact_name"] or "Unknown Contact"}</div>
                <div style='color:{uc};font-weight:800;font-size:13px;'>Urgency: {urg}</div>
                <div style='color:#dc2626;font-weight:700'>⏳ {math.floor(rem)}h left</div>
                <div style='color:#22c55e;font-weight:800'>{format_money(r["estimated_value"])}</div>
                <div style='color:white'>Score: {r["score"]:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # PIPELINE FUNNEL CHART
    st.subheader("Pipeline Funnel")
    funnel = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
    st.bar_chart(funnel, use_container_width=True)

    st.markdown("---")
    st.subheader("Lead Table")
    q = st.text_input("Search leads","Search by ID, Name, Notes or Value")
    if q:
        df = df[df.apply(lambda r: q.lower() in str(r.values).lower(), axis=1)]
    if df.empty:
        st.info("No leads found.")
    else:
        for _,r in df.iterrows():
            with st.expander(f"#{r['lead_id']} — {r['stage']}"):
                colA,colB = st.columns([3,1])
                with colA:
                    st.write("**Source**:",r["source"])
                    st.write("**Address**:",r["property_address"])
                    st.write("**Contact**:",r["contact_name"],r["contact_phone"],r["contact_email"])
                    st.write("**Value**:",format_money(r["estimated_value"]))
                    st.write("**Cost**:",format_money(r["ad_cost"]))
                    st.write("**SLA Hours**:",r["sla_hours"])
                    st.write("**Notes**:",r["notes"] or "—")
                with colB:
                    if is_overdue(r["sla_entered_at"],r["sla_hours"]) and r["stage"] not in ["Won","Lost"]:
                        st.error("OVERDUE")
                    else:
                        st.info(f"{math.floor(remaining_sla_seconds(r['sla_entered_at'],r['sla_hours'])/3600)}h left")
                with st.form(f"form_{r['lead_id']}"):
                    ns = st.selectbox("Pipeline Stage",PIPELINE_STAGES, index=PIPELINE_STAGES.index(r["stage"]) if r["stage"] in PIPELINE_STAGES else 0)
                    nv = st.number_input("Estimated value",value=float(r["estimated_value"] or 0))
                    nc = st.number_input("Acquisition cost",value=float(r["ad_cost"] or 0))
                    note = st.text_area("Notes",value=r["notes"])
                    sb = st.form_submit_button("Save")
                    if sb:
                        save_lead({"lead_id":r["lead_id"],"stage":ns,"estimated_value":nv,"ad_cost":nc,"notes":note})
                        st.success("Saved ✅"); st.experimental_rerun()
                if st.button("Delete Lead",key=f"del_{r['lead_id']}",type="secondary"):
                    remove_lead(r["lead_id"]); st.success("Deleted ✅"); st.experimental_rerun()


# -----------------------
# Page 2 — Lead Capture
# -----------------------
def page_capture():
    st.markdown("## Lead Capture")
    with st.form("lead_input",clear_on_submit=True):
        lid = st.text_input("Lead ID (optional)")
        src = st.selectbox("Source",PIPELINE_STAGES)
        stg = st.selectbox("Stage",PIPELINE_STAGES)
        nm = st.text_input("Contact Name")
        ph = st.text_input("Contact Phone")
        em = st.text_input("Contact Email")
        add = st.text_input("Property Address")
        dt = st.selectbox("Damage Type",["water","mold","fire","contents","reconstruction","other"])
        val = st.number_input("Estimated Value",min_value=0.0,value=0.0,step=50.0)
        cost = st.number_input("Acquisition Cost",min_value=0.0,value=0.0,step=1.0)
        sla = st.number_input("SLA Hours",min_value=1,value=DEFAULT_SLA_HOURS)
        notes = st.text_area("Notes")
        sb = st.form_submit_button("Submit")
        if sb:
            if sla <=0:
                st.error("SLA must be > 0")
            else:
                save_lead({"lead_id": lid or f"L{int(datetime.utcnow().timestamp())}", "created_at":datetime.utcnow().isoformat(),"source":src,"stage":stg,
                           "contact_name":nm,"contact_phone":ph,"contact_email":em,"property_address":add,"damage_type":dt,"estimated_value":val,"ad_cost":cost,
                           "sla_hours":sla,"notes":notes,"converted":1 if stg=="Won" else 0,"score":0.5})
                st.success("Lead saved ✅"); st.experimental_rerun()

# -----------------------
# Page 3 — Analytics & SLA
# -----------------------
def page_analytics():
    st.markdown("## Analytics & SLA")
    df = fetch_leads()
    if df.empty:
        st.info("No analytics data available")
        return

    # COST VS CONVERSION
    agg = df.copy()
    agg["won"] = agg["stage"].apply(lambda s: 1 if s == "Won" else 0)
    agg_src = agg.groupby("source").agg(spend=("ad_cost","sum"), wins=("won","sum")).reset_index()

    fig = px.bar(agg_src, x="source", y=["spend","wins"], title="Cost vs Conversions", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # SLA OVER 30 DAYS
    st.markdown("---")
    st.subheader("SLA Overdue Trend (30 Days)")

    today = date.today()
    last30 = []
    for i in range(30):
        d = today - timedelta(days=i)
        last30.append(d)

    results = []
    for d in reversed(last30):
        day_df = df[(df["created_at"]>=pd.to_datetime(d)) & (df["created_at"]<pd.to_datetime(d)+pd.Timedelta(days=1))]
        overdue = int(day_df[day_df.apply(lambda r: is_overdue(r["sla_entered_at"],r["sla_hours"]) and r["stage"] not in ["Won","Lost"], axis=1)].shape[0])
        results.append({"date":d,"overdue":overdue})

    trend = pd.DataFrame(results)
    fig2 = px.line(trend, x="date", y="overdue", title="SLA Overdue Count", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    # OVERDUE TABLE
    st.markdown("---")
    st.subheader("Currently Overdue Leads")
    odf = df[df.apply(lambda r: is_overdue(r["sla_entered_at"],r["sla_hours"]) and r["stage"] not in ["Won","Lost"], axis=1)]
    if odf.empty:
        st.success("No overdue leads ✅")
    else:
        st.dataframe(odf[["lead_id","contact_name","stage","estimated_value","ad_cost","sla_hours"]])


# -----------------------
# Page 4 — CPA
# -----------------------
def page_cpa():
    st.markdown("## Cost Per Acquisition (CPA)")
    df = fetch_leads()
    spend = df["ad_cost"].sum()
    wins = df[df["stage"]=="Won"].shape[0]
    cpa = (spend/wins) if wins else 0
    st.markdown(f"<div class='priority-card'><div class='kpe'>${cpa:,.2f}</div>Average CPA</div>",unsafe_allow_html=True)

# -----------------------
# Page 5 — ROI & ROI Page
# -----------------------
def page_roi():
    st.markdown("## Return on Investment (ROI)")
    df = fetch_leads()
    spend = df["ad_cost"].sum()
    value = df["estimated_value"].sum()
    roi = ((value-spend)/spend*100) if spend else 0
    st.markdown(f"<div class='priority-card'><div class='kpe'>{roi:.1f}%</div>ROI Rate</div>",unsafe_allow_html=True)

# -----------------------
# Seed generator
# -----------------------
def seed_data():
    df = generate_mock(200)
    conn = get_conn(); c=conn.cursor()
    for _,r in df.iterrows():
        try:
            c.execute("INSERT OR IGNORE INTO leads(lead_id,created_at,source,stage,estimated_value,ad_cost,converted,notes,sla_hours,sla_entered_at,score) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      (r["lead_id"],r["created_at"],r["source"],r["stage"],r["estimated_value"],r["ad_cost"],r["converted"],r["notes"],r["sla_hours"],r["sla_entered_at"],r["score"]))
        except: pass
    conn.commit(); conn.close()

def generate_mock(n=100):
    rng=np.random.default_rng(32)
    df=pd.DataFrame({
        "lead_id":[f"M{10000+i}" for i in range(n)],
        "created_at":[(datetime.utcnow()-timedelta(days=int(x))).isoformat() for x in rng.integers(0,95,n)],
        "source":rng.choice(["Google Ads","Facebook","Referral","Partner","Organic"],n),
        "stage":rng.choice(PIPELINE_STAGES,n,p=[.1,.2,.2,.15,.15,.2]),
        "estimated_value":rng.normal(3000,2200,n).clip(150,22000),
        "ad_cost":rng.normal(60,40,n).clip(0,800),
        "converted":rng.integers(0,2,n),
        "notes":[""]*n,
        "sla_hours":rng.integers(1,72,n),
        "sla_entered_at":[(datetime.utcnow()-timedelta(hours=int(h))).isoformat() for h in rng.integers(0,48,n)],
        "score":rng.random(n)
    })
    return df


# -----------------------
# Export CSV/Excel fallback
# -----------------------
def export_excel(df):
    towrite=io.BytesIO()
    try:
        df.to_excel(towrite,index=False,engine="openpyxl")
    except ModuleNotFoundError:
        subprocess.run([sys.executable,"-m","pip","install","openpyxl"],check=False)
        df.to_excel(towrite,index=False,engine="openpyxl")
    towrite.seek(0)
    b64=base64.b64encode(towrite.read()).decode()
    return f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"


def page_exports():
    st.markdown("## Export Leads")
    df = fetch_leads()
    if df.empty:
        st.info("No leads available.")
    else:
        st.download_button("Download CSV",df.to_csv(index=False).encode(),"leads.csv","text/csv")
        link = export_excel(df)
        st.markdown(f"<a href='{link}' download='leads.xlsx'>Download Excel</a>",unsafe_allow_html=True)


# -----------------------
# AI Observation Page + Report Summary
# -----------------------
def page_reports():
    st.markdown("## AI Observation & Recommendation")
    df = fetch_leads()
    obs = []
    if df.empty:
        st.info("No leads for reporting")
        return
    # bottlenecks
    bottleneck = df["stage"].value_counts().idxmax()
    obs.append(f"- Most leads are currently stuck at '{bottleneck}' stage, consider follow-ups or optimization.")
    # cost spenders
    high_cost = df[df["ad_cost"]>df["ad_cost"].quantile(0.8)]
    if not high_cost.empty:
        obs.append(f"- {len(high_cost)} leads had high acquisition cost — optimize bidding or campaigns.")
    # SLA breaches
    overdue_df = df[df.apply(lambda r: is_overdue(r["sla_entered_at"],r["sla_hours"]) and r["stage"] not in ["Won","Lost"],axis=1)]
    if not overdue_df.empty:
        obs.append(f"- {len(overdue_df)} active leads breached SLA — follow up urgently.")
    # ROI suggestions
    total_spend = df["ad_cost"].sum()
    total_value = df["estimated_value"].sum()
    if total_spend:
        roi = ((total_value-total_spend)/total_spend*100)
        if roi < 20:
            obs.append("- ROI is low (<20%), consider reducing marketing cost or increasing qualification.")
        else:
            obs.append("- ROI is healthy, scale up winning channels!")
    # summary
    st.markdown("### Report Summary")
    for o in obs:
        st.write(o)
    st.markdown("---")
    st.subheader("Download executive report")
    txt = "\n".join(obs)
    st.download_button("Download report",txt.encode(),"lead_report.txt","text/plain")

# -----------------------
# Email/SMS SLA alert placeholder
# -----------------------
def send_alert_placeholder(lid,hrs):
    # This is placeholder only
    st.toast(f"ALERT: Lead #{lid} has only {hrs}h left to respond!")


# -----------------------
# Navigation routing
# -----------------------
if page == "Pipeline":
    page_pipeline()
elif page == "Lead Capture":
    page_capture()
elif page == "Analytics & SLA":
    page_analytics()
elif page == "CPA":
    page_cpa()
elif page == "ROI":
    page_roi()
elif page == "Exports":
    page_exports()
elif page == "Settings":
    st.markdown("## User & Admin Roles")
    name = st.text_input("User Name")
    role = st.selectbox("Role",["Admin","Estimator","Manager","Viewer"])
    if st.button("Save"):
        st.session_state.profile={"name":name,"role":role}
        st.success("Saved ✅"); st.experimental_rerun()
elif page == "Reports":
    page_reports()
elif page == "ML":
    if st.button("Train Model"):
        st.success(train_model())
    st.write("ML scoring active internally.")
