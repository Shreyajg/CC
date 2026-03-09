import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import plotly.express as px
import pytz
import time
import streamlit.components.v1 as components
from gtts import gTTS
import os
# ---------------- PAGE ----------------
st.set_page_config(layout="wide")
st.markdown("""
<style>

/* main background */
.stApp {
    background-color: #0e1117;
}

/* title */
h1 {
    color: #EAEAEA;
    font-weight: 700;
}

/* section headers */
h2, h3 {
    color: #A9B1D6;
}

/* metric cards */
[data-testid="stMetric"] {
    background-color: #1c1f26;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #2b2f3a;
}

/* charts */
[data-testid="stPlotlyChart"] {
    background-color: #1c1f26;
    border-radius: 10px;
    padding: 10px;
}

/* warning boxes */
.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
st.title("WearOS Behaviour Monitoring")

with st.sidebar:
    st.header("Dashboard Controls")

    live_mode = st.toggle("Live Monitoring", value=False)

    if live_mode:
        st.caption("Auto refresh every 20 seconds")

    st.markdown("---")
    st.caption("WearOS Behaviour Monitoring")


# ---------------- FIREBASE ----------------
if not firebase_admin._apps:
    cred = credentials.Certificate("CC/ServiceKey.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------------- BASELINE STORAGE ----------------
@st.cache_data(ttl=3600)
def load_baseline_from_firestore():
    doc = db.collection("baseline_profile").document("user_baseline").get()
    if doc.exists:
        return pd.Series(doc.to_dict()["values"])
    return None

def save_baseline_to_firestore(series):
    db.collection("baseline_profile").document("user_baseline").set(
        {"values": series.to_dict()}
    )

saved_baseline = load_baseline_from_firestore()

# ---------------- SHOW SAVED BASELINE ----------------
if saved_baseline is not None:
    st.subheader("Saved Personal Baseline")

    baseline_df = pd.DataFrame(saved_baseline, columns=["Baseline Value"])
    st.dataframe(baseline_df)

    st.success("Baseline loaded from Firebase — monitoring mode active")
else:
    st.warning("No baseline saved yet — system will learn one.")


# ---------------- LOAD TODAY DATA (LOW READS) ----------------
@st.cache_data(ttl=60)
def load_sensor_data():
    ist = pytz.timezone("Asia/Kolkata")
    start = pd.Timestamp.now(tz=ist).normalize().tz_convert("UTC")
    docs = (
        db.collection("sensor_samples")
        .limit(1200)                 # 🔥 read cap
        .stream()
    )
    df = pd.DataFrame([d.to_dict() for d in docs])
    if df.empty: return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(ist)
    df = df.sort_values("timestamp")
    df["accelMag_smooth"] = df["accelMag"].rolling(5,min_periods=1).mean()
    return df

@st.cache_data(ttl=60)
def load_audio_data():
    ist = pytz.timezone("Asia/Kolkata")
    start = pd.Timestamp.now(tz=ist).normalize().tz_convert("UTC")
    docs = (
        db.collection("audio_samples")
        .limit(600)                  # 🔥 read cap
        .stream()
    )
    df = pd.DataFrame([d.to_dict() for d in docs])
    if df.empty: return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(ist)
    return df.sort_values("timestamp")

sensor_df = load_sensor_data()
audio_df = load_audio_data()

# ---------------- VISUALISE SENSORS ----------------
st.subheader("📈 Motion & Physiology")

col1, col2 = st.columns(2)

with col1:
    if not sensor_df.empty:
        fig = px.line(sensor_df, x="timestamp", y="accelMag_smooth",
                      title="Movement Intensity")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if "heartRate" in sensor_df:
        fig = px.line(sensor_df, x="timestamp", y="heartRate",
                      title="Heart Rate")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

if {"gyroX","gyroY","gyroZ"}.issubset(sensor_df.columns):
    fig = px.line(sensor_df,x="timestamp",y=["gyroX","gyroY","gyroZ"],
                  title="Rotation Activity")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig,use_container_width=True)

# ---------------- VISUALISE AUDIO ----------------


audio_cols=["audio_energy","audio_silence_ratio","audio_zcr",
            "speech_ratio","energy_variance","high_freq_ratio"]

st.subheader("🎤 Audio Behaviour")

cols = st.columns(2)

i = 0
for col in audio_cols:
    if col in audio_df.columns:
        fig = px.line(audio_df, x="timestamp", y=col, title=col)
        fig.update_layout(template="plotly_dark")

        cols[i%2].plotly_chart(fig, use_container_width=True)
        i += 1

# ---------------- MERGE STREAMS ----------------
merged=None
if not sensor_df.empty:
    merged=sensor_df.copy()
    if not audio_df.empty:
        merged=pd.merge_asof(audio_df.sort_values("timestamp"),
                             sensor_df.sort_values("timestamp"),
                             on="timestamp",direction="nearest",
                             tolerance=pd.Timedelta("10s"))

# ---------------- BEHAVIOUR WINDOWS ----------------
def compute_windows(df):
    if df is None or df.empty: return pd.DataFrame()
    df=df.set_index("timestamp")
    win=pd.DataFrame()
    win["movement_mean"]=df["accelMag_smooth"].resample("5min").mean()
    win["movement_std"]=df["accelMag_smooth"].resample("5min").std()
    win["movement_burst"]=df["accelMag_smooth"].resample("5min").apply(
        lambda s:(s>s.mean()+2*s.std()).sum() if len(s)>1 else 0)
    win["hr_mean"]=df["heartRate"].resample("5min").mean()
    # --- ACCEL FEATURES ---
    win["accel_mean"] = df["accelMag_smooth"].resample("5min").mean()
    win["accel_std"] = df["accelMag_smooth"].resample("5min").std()

    win["accel_peak_count"] = df["accelMag_smooth"].resample("5min").apply(
        lambda s: (s > s.mean() + 2*s.std()).sum() if len(s) > 1 else 0
    )

    # --- GYRO FEATURE (rotation intensity) ---
    if "gyroZ" in df: win["rotation_intensity"] = df["gyroZ"].abs().resample("5min").mean()
    if "audio_energy" in df: win["audio_energy"]=df["audio_energy"].resample("5min").mean()
    if "speech_ratio" in df: win["speech_ratio"]=df["speech_ratio"].resample("5min").mean()
    if "energy_variance" in df: win["audio_var"]=df["energy_variance"].resample("5min").mean()
    if "light" in df: win["light_mean"]=df["light"].resample("5min").mean()
    return win.ffill().bfill().reset_index()

behavior_df=compute_windows(merged)

# ---------------- BASELINE + ALS ----------------
if saved_baseline is None and len(behavior_df)>=3:
    st.info("Learning baseline...")
    baseline=behavior_df.median(numeric_only=True)
    save_baseline_to_firestore(baseline)
    st.success("Baseline saved!")
else:
    baseline=saved_baseline

def rel(a,b): return max(0,(a-b)/b) if b else 0

def compute_als(row):

    def safe_rel(a, b):
        if pd.isna(a) or pd.isna(b) or b == 0:
            return 0
        return max(0, (a - b) / b)

    # ----- MOTOR -----
    motor = min(1,
          0.30 * safe_rel(row.get("movement_mean", 0),
                    baseline.get("movement_mean", 0)) +

    0.25 * safe_rel(row.get("movement_std", 0),
                    baseline.get("movement_std", 0)) +

    0.20 * safe_rel(row.get("movement_burst", 0),
                    baseline.get("movement_burst", 0)) +

    0.25 * safe_rel(row.get("rotation_intensity", 0),
                    baseline.get("rotation_intensity", 0))
)
    

    # ----- HEART -----
    heart = min(1, max(0,
        (row.get("hr_mean", 0) - baseline.get("hr_mean", 0)) / 15
    ))

    # ----- AUDIO (SAFE) -----
    audio = 0
    if all(k in row for k in ["speech_ratio", "audio_energy", "audio_var"]) and \
       all(k in baseline for k in ["speech_ratio", "audio_energy", "audio_var"]):

        audio = min(1,
            0.40 * safe_rel(row["speech_ratio"], baseline["speech_ratio"]) +
            0.30 * safe_rel(row["audio_energy"], baseline["audio_energy"]) +
            0.30 * safe_rel(row["audio_var"], baseline["audio_var"])
        )

    # ----- CONTEXT -----
    context = min(1,
        safe_rel(row.get("light_mean", 0), baseline.get("light_mean", 0))
    )

    return 0.35*motor + 0.30*audio + 0.25*heart + 0.10*context
def intervention_for_state(state):
    if state == "High Agitation":
        return {
            "guided_breathing": False,
            "play_music": True,
            "alert_caregiver": True,
            "severity": "HIGH",
            "message": "High agitation — music started and caregiver alerted."
        }

    elif state == "Agitated":
        return {
            "guided_breathing": False,
            "play_music": True,
            "alert_caregiver": False,
            "severity": "MEDIUM",
            "message": "Moderate agitation — calming music started."
        }

    elif state == "Elevated":
        return {
            "guided_breathing": True,
            "play_music": False,
            "alert_caregiver": False,
            "severity": "LOW",
            "message": "Mild elevation — guided breathing started."
        }

    else:
        return {
            "guided_breathing": False,
            "play_music": False,
            "alert_caregiver": False,
            "severity": "NONE",
            "message": "No intervention required."
        }

def guided_breathing_animation():
    st.info("🧘 Guided Breathing Started")
    
    placeholder = st.empty()

    for _ in range(2):
        placeholder.markdown("### 🫁 Inhale... 4")
        time.sleep(1)
        placeholder.markdown("### 🫁 Inhale... 3")
        time.sleep(1)
        placeholder.markdown("### 🫁 Inhale... 2")
        time.sleep(1)
        placeholder.markdown("### 🫁 Inhale... 1")
        time.sleep(1)

        placeholder.markdown("### 🌬 Exhale... 4")
        time.sleep(1)
        placeholder.markdown("### 🌬 Exhale... 3")
        time.sleep(1)
        placeholder.markdown("### 🌬 Exhale... 2")
        time.sleep(1)
        placeholder.markdown("### 🌬 Exhale... 1")
        time.sleep(1)

    placeholder.empty()


def generate_breathing_audio():
    text = """
    Inhale... 4... 3... 2... 1...
    Exhale... 4... 3... 2... 1...
    Inhale... 4... 3... 2... 1...
    Exhale... 4... 3... 2... 1...
    """

    tts = gTTS(text=text, lang="en", slow=True)
    tts.save("guided_breathing.mp3")

def show_calming_playlist():
    components.html("""
    <iframe style="border-radius:12px"
    src="https://open.spotify.com/embed/playlist/37i9dQZF1DX3Ogo9pFvBkY"
    width="100%" height="380" frameBorder="0"
    allowfullscreen=""
    allow="clipboard-write; encrypted-media; fullscreen; picture-in-picture">
    </iframe>
    """, height=400)
def show_emergency_alert():
    st.markdown("""
    <style>
    .emergency-alert {
        padding: 20px;
        background-color: #ff4b4b;
        color: white;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        border-radius: 10px;
        animation: blink 1s infinite;
    }

    @keyframes blink {
        0% {opacity: 1;}
        50% {opacity: 0.3;}
        100% {opacity: 1;}
    }
    </style>

    <div class="emergency-alert">
        🚨 HIGH AGITATION DETECTED — CAREGIVER ALERT SENT 🚨
    </div>
    """, unsafe_allow_html=True)

def show_caregiver_alert():
    st.error("🚨 HIGH AGITATION DETECTED")
    st.markdown("### 🔔 Caregiver Alert Triggered")

# ---------------- ALS COMPUTATION & DISPLAY ----------------
if saved_baseline is not None and behavior_df is not None and len(behavior_df) > 0:

    # compute ALS
    behavior_df["ALS"] = behavior_df.apply(compute_als, axis=1)

    # smoothing
    behavior_df["ALS"] = behavior_df["ALS"].rolling(3, min_periods=1).mean()

    # state label
    def agitation_state(score):
        if score < 0.0:
            return "Calm"
        elif score < 0.1:
            return "Elevated"
        elif score < 0.7:
            return "Agitated"
        else:
            return "High Agitation"

    behavior_df["State"] = behavior_df["ALS"].apply(agitation_state)

    # ---- VISUALS ----
    st.subheader("Agitation Timeline")
    fig = px.line(
    behavior_df,
    x="timestamp",
    y="ALS",
    title="Agitation Level Timeline"
)

    fig.update_layout(
        template="plotly_dark",
        yaxis_title="ALS Score",
        xaxis_title="Time"
    )

    st.plotly_chart(fig, use_container_width=True)

    latest = behavior_df.iloc[-1]

    c1, c2 = st.columns(2)
    st.subheader("🧠 Current Behaviour State")

    c1, c2, c3 = st.columns(3)

    c1.metric("Agitation Level", f"{latest['ALS']:.2f}")

    state = latest["State"]

    color = {
        "Calm":"🟢",
        "Elevated":"🟡",
        "Agitated":"🟠",
        "High Agitation":"🔴"
    }

    c2.metric("State", f"{color[state]} {state}")

    c3.metric("Timestamp", latest["timestamp"].strftime("%H:%M"))
    intervention = intervention_for_state(latest["State"])

    if latest["State"] == "Elevated":
        generate_breathing_audio()
        st.audio('guided_breathing.mp3',autoplay=True,loop=True)
        guided_breathing_animation()

    elif latest["State"] == "Agitated":
        st.warning("Moderate agitation detected")
        show_calming_playlist()

    elif latest["State"] == "High Agitation":
        show_emergency_alert()
        show_calming_playlist()

    st.subheader("Recommended Intervention")
    if intervention["severity"] == "HIGH":
        st.error(intervention["message"])
    elif intervention["severity"] == "MEDIUM":
        st.warning(intervention["message"])
    elif intervention["severity"] == "LOW":
        st.info(intervention["message"])
    else:
        st.success(intervention["message"])



