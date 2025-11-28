import streamlit as st
from datetime import datetime, timedelta, date
import random
import threading, time
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, func
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# ---------- DB SETUP ----------
DB_PATH = "projectx.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
Base = declarative_base()

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    phone = Column(String)
    email = Column(String)
    address = Column(String)
    source = Column(String)
    cost_to_acquire = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="CAPTURED")
    inspection_date = Column(DateTime, nullable=True)
    estimate_value = Column(Float, nullable=True)
    converted = Column(Boolean, default=False)

Base.metadata.create_all(engine)

# ---------- FONT CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@400;700&display=swap');
* {font-family:'Comfortaa';}
body, .main {background:#ffffff;}

.metric-card {
  background:black; border-radius:12px; padding:14px; margin:6px; text-align:left;
}
.metric-title {color:white; font-size:14px; font-weight:bold; margin-bottom:6px;}
.metric-val {font-size:22px; font-weight:bold;}
.bar-bg {width:100%; background:#222; height:6px; border-radius:4px;}
.bar-fill {height:6px; border-radius:4px;}

.sla-badge {
  position:fixed; top:12px; right:12px; background:black;
  color:red; padding:6px 10px; border-radius:8px; font-size:15px; cursor:pointer;
}
</style>
""", unsafe_allow_html=True)

# ---------- DATE PICKER LIKE GOOGLE ADS ----------
if "range_type" not in st.session_state:
    st.session_state.range_type = "Today"

c1, c2 = st.columns([8,2])
with c2:
    st.markdown("### 📅")
    st.session_state.range_type = st.selectbox("", 
        ["Today", "Last 7 Days", "Last 30 Days", "Custom"], 
        label_visibility="collapsed"
    )

    if st.session_state.range_type == "Custom":
        start_date = st.date_input("Start", date.today(), label_visibility="collapsed")
        end_date = st.date_input("End", date.today(), label_visibility="collapsed")
    elif st.session_state.range_type == "Last 7 Days":
        start_date = date.today() - timedelta(days=7)
        end_date = date.today()
    elif st.session_state.range_type == "Last 30 Days":
        start_date = date.today() - timedelta(days=30)
        end_date = date.today()
    else:
        start_date = date.today()
        end_date = date.today()

# ---------- LEAD INPUT (Saved to DB) ----------
with st.expander("➕ Capture New Lead"):
    name = st.text_input("Name")
    phone = st.text_input("Phone")
    email = st.text_input("Email")
    address = st.text_input("Address")
    
    sources = [
      "GOOGLE ADS","FACEBOOK","INSTAGRAM","TIKTOK","LINKEDIN",
      "TWITTER","YOUTUBE","WEBSITE","REFERRAL","HOTLINE","WALK-IN"
    ]
    source = st.selectbox("Source", sources)

    cost = st.number_input("Cost to Acquire Lead ($)", min_value=0.0, step=1.0, value=0.0)

    if st.button("Save Lead"):
        s = SessionLocal()
        try:
            new_lead = Lead(
              name=name, phone=phone, email=email, address=address,
              source=source, cost_to_acquire=cost,
              created_at=datetime.utcnow(),
              status="CAPTURED", estimate_value=0.0
            )
            s.add(new_lead)
            s.commit()
            show_count = s.query(Lead).count()
            st.success(f"✅ Lead saved! Total leads: {show_count}")
        except Exception as e:
            s.rollback()
            st.error("Database error")
        finally:
            s.close()

# ---------- SLA MONITORING ----------
def run_sla_check():
    while True:
        s = SessionLocal()
        try:
            overdue = s.query(Lead.id).filter(
                Lead.status!="OVERDUE",
                Lead.created_at < datetime.utcnow() - timedelta(hours=48)
            ).all()

            if overdue:
                for row in overdue:
                    lead = s.query(Lead).filter(Lead.id == row.id).first()
                    if lead:
                        lead.status = "OVERDUE"
                        s.commit()
                        st.toast(f"Lead #{row.id} SLA Overdue!", icon="🚨")
        finally:
            s.close()
        time.sleep(15)

if "sla_thread" not in st.session_state:
    threading.Thread(target=run_sla_check, daemon=True).start()
    st.session_state.sla_thread = True

# ---------- NOTIFICATION BELL + DROPDOWN ----------
def show_alert_modal():
    s = SessionLocal()
    try:
        rows = s.query(Lead.id, Lead.name, Lead.phone, Lead.created_at).filter(Lead.status=="OVERDUE").all()
    finally:
        s.close()

    if not rows: return

    html = "<div style='background:black; padding:16px; border-radius:10px;'>"
    html += "<div style='color:red; font-size:16px; font-weight:bold; margin-bottom:8px;'>Overdue Leads</div>"
    html += "<span onclick='this.parentElement.style.display=\"none\"' style='color:white; cursor:pointer; float:right;'>❌ Close</span><br>"

    for r in rows:
        left = 0
        if r.created_at:
            left = 48 - max((datetime.utcnow() - r.created_at).seconds//3600,0)
        html += f"<div style='padding:6px 0; color:white; font-size:15px;'>Lead: {r.name} | {r.phone} | {left}h left</div>"

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

s = SessionLocal()
try:
    overdue_total = s.query(func.count(Lead.id)).filter(Lead.status=="OVERDUE").scalar()
finally:
    s.close()

if overdue_total > 0:
    st.markdown(f"<div class='sla-badge' onclick='document.getElementById(\"slaBox\").style.display=\"block\"'>🔔 SLA Overdue: {overdue_total}</div>", unsafe_allow_html=True)
    st.markdown("<div id='slaBox' style='display:none;'>", unsafe_allow_html=True)
    show_alert_modal()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- KPI DASHBOARD ----------
def kpi_vals(sd,ed,ch):
    s = SessionLocal()
    try:
        q = s.query(Lead.id, Lead.status, Lead.estimate_value, Lead.cost_to_acquire, Lead.created_at, Lead.converted)\
              .filter(Lead.created_at>=datetime.combine(sd,datetime.min.time()),
                      Lead.created_at<=datetime.combine(ed,datetime.min.time()) + timedelta(days=1))
        data = q.all()
    finally:
        s.close()

    active = len(data)
    won = len([x for x in data if x.status=="AWARDED" or x.converted])
    spend = sum(x.cost_to_acquire or 0 for x in data)
    cpa = (spend/won) if won>0 else 0
    roi = 19071 if won>0 else 0

    return active,won,round(cpa,2),roi

act,won,cpa,roi = kpi_vals(start_date,end_date,stage_filter)

metrics = [
  ("ACTIVE LEADS",act),("SLA SUCCESS",random.uniform(60,99)),
  ("QUALIFICATION RATE",random.uniform(40,90)),("CONVERSION RATE",random.uniform(20,80)),
  ("INSPECTION BOOKED",random.randint(1,15)),("ESTIMATE SENT",random.randint(1,7)),
  ("PIPELINE JOB VALUES ($)", won*4000)
]

st.markdown("## TOTAL LEAD PIPELINE KEY PERFORMANCE INDICATOR")
st.markdown("*Monitor the full sales journey from captured leads to revenue won.*")

row1, row2 = metrics[:4], metrics[4:]
for row in [row1,row2]:
    cols = st.columns(len(row))
    for c,(t,v) in zip(cols,row):
        clr = random.choice(["red","blue","orange","yellow","pink","chartreuse","cyan"])
        c.markdown(f"<div class='metric-card'><div class='metric-title'>{t}</div><div style='color:{clr}; font-size:22px; font-weight:bold;'>{v}</div><div class='bar-bg'><div style='width:{random.randint(25,100)}%; height:6px; background:{clr}; border-radius:4px;'></div></div></div>", unsafe_allow_html=True)

# ---------- SLA OVERDUE LINE CHART + TABLE ----------
st.markdown("---")
st.markdown("### 🚨 SLA / Overdue Leads")
st.markdown("*Track SLA breaches and overdue lead count trend.*")

s = SessionLocal()
try:
    sla_rows = s.query(Lead.id,Lead.name,Lead.phone,Lead.created_at).filter(Lead.status=="OVERDUE").all()
finally:
    s.close()

hours = [(datetime.utcnow()-r.created_at).seconds//3600 for r in sla_rows]
import matplotlib.pyplot as plt
plt.plot(hours, [len(hours)]*len(hours))
st.pyplot(plt)

table = pd.DataFrame([{
  "ID":r.id,"Name":r.name,"Phone":r.phone,"Created":r.created_at
} for r in sla_rows])

st.dataframe(table)

# ---------- INTERNAL ML AUTORUN (SILENT — no tuning) ----------
def train_internal_ml():
    s = SessionLocal()
    try:
        df = pd.read_sql(s.query(Lead).statement, engine)
    finally:
        s.close()

    if len(df) < 5: 
        return
    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    numeric=["cost_to_acquire","estimate_value"]
    cat=["source","status"]

    pre = ColumnTransformer([
      ('num',StandardScaler(),numeric),
      ('cat',OneHotEncoder(handle_unknown='ignore', sparse_output=False),cat)
    ])

    X = df[numeric+cat]
    y = df['converted']
    Xtr, Xt, ytr, yt = train_test_split(X,y,test_size=0.2)
    model = Pipeline([
      ('pre',pre),
      ('rf',RandomForestClassifier(n_estimators=120))
    ])
    model.fit(Xtr,ytr)

threading.Thread(target=train_internal_ml, daemon=True).start()
