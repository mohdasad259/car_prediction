import streamlit as st
import numpy as np
import joblib
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0f14; color: #e8e6e0; }

section[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #1f2330;
}
section[data-testid="stSidebar"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.76rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
    font-weight: 500 !important;
}

.hero {
    background: linear-gradient(135deg, #1a1d27 0%, #0d0f14 100%);
    border: 1px solid #1f2330; border-radius: 16px;
    padding: 2.4rem 3rem; margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
}
.hero::after {
    content: '🚗'; position: absolute;
    right: 2.5rem; top: 50%; transform: translateY(-50%);
    font-size: 5rem; opacity: 0.07;
}
.hero h1 {
    font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800;
    color: #f0ede6; margin: 0 0 0.35rem 0; letter-spacing: -0.02em;
}
.hero p { color: #6b7280; font-size: 0.92rem; margin: 0; font-style: italic; }
.hero .accent { color: #f5a623; }

.metric-card {
    background: #13161e; border: 1px solid #1f2330;
    border-radius: 12px; padding: 1.2rem 1.4rem;
    text-align: center; transition: border-color .2s;
}
.metric-card:hover { border-color: #f5a623; }
.metric-card .lbl { font-size: 0.68rem; text-transform: uppercase; letter-spacing: .1em; color: #6b7280; margin-bottom: .4rem; }
.metric-card .val { font-family: 'Syne',sans-serif; font-size: 1.55rem; font-weight: 700; color: #f0ede6; }
.metric-card .unit { font-size: 0.78rem; color: #6b7280; margin-left: 3px; }

.result-banner {
    background: linear-gradient(90deg,#1a1208 0%,#13161e 100%);
    border: 1.5px solid #f5a623; border-radius: 16px;
    padding: 2rem 2.5rem; display: flex;
    align-items: center; justify-content: space-between;
    margin: 1.2rem 0 1.8rem 0;
}
.result-banner .price-label { font-size:.7rem; text-transform:uppercase; letter-spacing:.12em; color:#f5a623; margin-bottom:.25rem; }
.result-banner .price-value { font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800; color:#f0ede6; letter-spacing:-.03em; }
.result-banner .price-sub { color:#6b7280; font-size:.82rem; margin-top:.2rem; }
.result-banner .seg-tag { border-radius:8px; padding:.55rem 1.2rem; font-size:.88rem; font-weight:500; border:1px solid; }

.sec-title {
    font-family:'Syne',sans-serif; font-size:.68rem;
    text-transform:uppercase; letter-spacing:.14em; color:#f5a623;
    margin: 1.6rem 0 .7rem 0; border-left: 3px solid #f5a623; padding-left: 9px;
}

.info-box {
    background:#13161e; border:1px solid #1f2330;
    border-radius:10px; padding:1rem 1.2rem;
    font-size:.82rem; color:#6b7280; line-height:1.6;
}
.info-box b.h { color:#f0ede6; }

.note-chip {
    display:inline-block; background:rgba(245,166,35,.08);
    border:1px solid rgba(245,166,35,.2); border-radius:6px;
    padding:2px 8px; font-size:.7rem; color:#f5a623; margin-left:4px;
}

div.stButton > button {
    width:100%;
    background: linear-gradient(135deg,#f5a623 0%,#e8940d 100%);
    color:#0d0f14; font-family:'Syne',sans-serif;
    font-weight:700; font-size:.95rem; letter-spacing:.05em;
    border:none; border-radius:12px; padding:.82rem 2rem;
    text-transform:uppercase;
}
div.stButton > button:hover { opacity:.9; border:none; }
hr { border-color:#1f2330 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    m = joblib.load(os.path.join(base, "ridge_model.pkl"))
    s = joblib.load(os.path.join(base, "scaler.pkl"))
    return m, s

try:
    model, scaler = load_models()
    model_ok = True
except Exception as e:
    model_ok = False
    err_msg = str(e)

# ── Exact feature order from notebook ────────────────────────────────────────
# pd.get_dummies(drop_first=True) drops first alphabetical category per column:
#   carbody      → drops "convertible"
#   drivewheel   → drops "4wd"
#   enginelocation → drops "front"
#   enginetype   → drops "dohc"
#   cylindernumber → drops "eight"
FEATURE_ORDER = [
    'symboling', 'wheelbase', 'carlength', 'carwidth', 'curbweight',
    'enginesize', 'horsepower', 'citympg',
    'carbody_hardtop', 'carbody_hatchback', 'carbody_sedan', 'carbody_wagon',
    'drivewheel_fwd', 'drivewheel_rwd',
    'enginelocation_rear',
    'enginetype_dohcv', 'enginetype_l', 'enginetype_ohc',
    'enginetype_ohcf', 'enginetype_ohcv', 'enginetype_rotor',
    'cylindernumber_five', 'cylindernumber_four', 'cylindernumber_six',
    'cylindernumber_three', 'cylindernumber_twelve', 'cylindernumber_two',
]  # 27 features — matches scaler.feature_names_in_

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Car Price <span class="accent">Predictor</span></h1>
  <p>Ridge Regression · CarPrice Assignment Dataset · 27 features · Configure specs in the sidebar</p>
</div>
""", unsafe_allow_html=True)

if not model_ok:
    st.error(f"⚠️ Could not load model. Place `ridge_model.pkl` and `scaler.pkl` in the same folder.\n\n`{err_msg}`")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:#f0ede6;margin-bottom:1.2rem;'>⚙️ Car Specifications</h2>", unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>📐 Dimensions & Weight</div>", unsafe_allow_html=True)
    wheelbase  = st.slider("Wheelbase (in)",    86.6, 120.9,  98.8, 0.1)
    carlength  = st.slider("Car Length (in)",  141.1, 208.1, 174.0, 0.1)
    carwidth   = st.slider("Car Width (in)",    60.3,  72.3,  65.9, 0.1)
    curbweight = st.slider("Curb Weight (lbs)", 1488,  4066,  2555, 10)

    st.markdown("<div class='sec-title'>🔧 Engine</div>", unsafe_allow_html=True)
    enginesize = st.slider("Engine Size (cc)", 61, 326, 126, 1)
    horsepower = st.slider("Horsepower (hp)",  48, 288, 102, 1)

    enginetype = st.selectbox(
        "Engine Type",
        ["DOHC (baseline)", "DOHCV", "L (Inline)", "OHC", "OHCF", "OHCV", "Rotor"],
        index=3,
        help="Baseline (dropped dummy) = DOHC"
    )
    enginelocation = st.selectbox(
        "Engine Location",
        ["Front (baseline)", "Rear"],
        help="Baseline (dropped dummy) = Front"
    )
    cylindernumber = st.selectbox(
        "Cylinder Number",
        ["Eight (baseline)", "Five", "Four", "Six", "Three", "Twelve", "Two"],
        index=3,
        help="Baseline (dropped dummy) = Eight"
    )

    st.markdown("<div class='sec-title'>🚙 Body & Drive</div>", unsafe_allow_html=True)
    carbody = st.selectbox(
        "Car Body",
        ["Convertible (baseline)", "Hardtop", "Hatchback", "Sedan", "Wagon"],
        index=3,
        help="Baseline (dropped dummy) = Convertible"
    )
    drivewheel = st.selectbox(
        "Drive Wheel",
        ["4WD (baseline)", "FWD (Front)", "RWD (Rear)"],
        index=1,
        help="Baseline (dropped dummy) = 4WD"
    )

    st.markdown("<div class='sec-title'>⛽ Fuel & Risk</div>", unsafe_allow_html=True)
    citympg  = st.slider("City MPG", 13, 49, 26, 1)
    symboling = st.select_slider(
        "Risk Rating (Symboling)",
        options=[-3, -2, -1, 0, 1, 2, 3], value=0,
        help="-3 = safest, +3 = riskiest"
    )

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Price", use_container_width=True)

# ── Build Feature Vector ──────────────────────────────────────────────────────
def build_vector():
    # carbody (base = convertible)
    cb_ht = 1 if "Hardtop"   in carbody else 0
    cb_hb = 1 if "Hatchback" in carbody else 0
    cb_sd = 1 if "Sedan"     in carbody else 0
    cb_wg = 1 if "Wagon"     in carbody else 0

    # drivewheel (base = 4wd)
    dw_fwd = 1 if "FWD" in drivewheel else 0
    dw_rwd = 1 if "RWD" in drivewheel else 0

    # enginelocation (base = front)
    el_rear = 1 if "Rear" in enginelocation else 0

    # enginetype (base = dohc)
    et_map = {"DOHCV": (1,0,0,0,0,0), "L (Inline)": (0,1,0,0,0,0),
              "OHC":   (0,0,1,0,0,0), "OHCF":       (0,0,0,1,0,0),
              "OHCV":  (0,0,0,0,1,0), "Rotor":       (0,0,0,0,0,1)}
    et_key = enginetype.replace(" (baseline)", "").strip()
    et_dv, et_l, et_ohc, et_ohcf, et_ohcv, et_rot = et_map.get(et_key, (0,0,0,0,0,0))

    # cylindernumber (base = eight)
    cy_map = {"Five":    (1,0,0,0,0,0), "Four":   (0,1,0,0,0,0),
              "Six":     (0,0,1,0,0,0), "Three":  (0,0,0,1,0,0),
              "Twelve":  (0,0,0,0,1,0), "Two":    (0,0,0,0,0,1)}
    cy_key = cylindernumber.replace(" (baseline)", "").strip()
    cy_5, cy_4, cy_6, cy_3, cy_12, cy_2 = cy_map.get(cy_key, (0,0,0,0,0,0))

    vec = [
        symboling, wheelbase, carlength, carwidth, curbweight,
        enginesize, horsepower, citympg,
        cb_ht, cb_hb, cb_sd, cb_wg,
        dw_fwd, dw_rwd, el_rear,
        et_dv, et_l, et_ohc, et_ohcf, et_ohcv, et_rot,
        cy_5, cy_4, cy_6, cy_3, cy_12, cy_2,
    ]
    return np.array(vec, dtype=float).reshape(1, -1)

# ── Metric Tiles ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, lbl, val, unit in [
    (c1, "Engine Size", enginesize, "cc"),
    (c2, "Horsepower",  horsepower, "hp"),
    (c3, "City MPG",    citympg,    "mpg"),
    (c4, "Curb Weight", curbweight, "lbs"),
]:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='lbl'>{lbl}</div>
            <div class='val'>{val}<span class='unit'>{unit}</span></div>
        </div>""", unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    X_scaled = scaler.transform(build_vector())
    price = max(float(model.predict(X_scaled)[0]), 0)

    seg, sc = (
        ("Budget",    "#4ade80") if price < 8000 else
        ("Mid-Range", "#60a5fa") if price < 15000 else
        ("Upper-Mid", "#f5a623") if price < 25000 else
        ("Premium",   "#f87171")
    )

    def clean(s): return s.replace(" (baseline)", "")

    st.markdown(f"""
    <div class='result-banner'>
        <div>
            <div class='price-label'>Estimated Market Price</div>
            <div class='price-value'>${price:,.0f}</div>
            <div class='price-sub'>{clean(carbody)} · {clean(drivewheel)} · {clean(enginetype)} · {clean(cylindernumber)} cyl</div>
        </div>
        <div class='seg-tag' style='border-color:{sc}55;color:{sc};'>{seg} Segment</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='background:#13161e;border:1px dashed #2a2f3e;border-radius:16px;
                padding:2.5rem;text-align:center;margin:1.2rem 0 1.8rem 0;'>
        <div style='font-size:2rem;margin-bottom:.5rem;'>🔍</div>
        <div style='font-family:Syne,sans-serif;font-size:1.05rem;color:#6b7280;'>
            Adjust specs in the sidebar and hit <strong style='color:#f5a623;'>Predict Price</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Summary ───────────────────────────────────────────────────────────────────
def clean(s): return s.replace(" (baseline)", "")
def is_base(s): return "(baseline)" in s

st.markdown("<div class='sec-title'>📋 Input Summary</div>", unsafe_allow_html=True)
ca, cb = st.columns(2)
with ca:
    st.markdown(f"""
    <div class='info-box'>
    <b class='h'>Dimensions</b><br>
    Wheelbase: <b>{wheelbase}"</b> &nbsp;·&nbsp; Length: <b>{carlength}"</b> &nbsp;·&nbsp; Width: <b>{carwidth}"</b><br>
    Curb Weight: <b>{curbweight} lbs</b>
    <br><br>
    <b class='h'>Performance</b><br>
    Engine: <b>{enginesize} cc</b> &nbsp;·&nbsp; <b>{horsepower} hp</b> &nbsp;·&nbsp; <b>{citympg} city MPG</b><br>
    Risk (Symboling): <b>{symboling:+d}</b>
    </div>""", unsafe_allow_html=True)

with cb:
    def row(label, val):
        chip = "<span class='note-chip'>baseline</span>" if is_base(val) else ""
        return f"{label}: <b>{clean(val)}</b>{chip}<br>"
    st.markdown(f"""
    <div class='info-box'>
    <b class='h'>Configuration</b><br>
    {row("Body", carbody)}
    {row("Drive Wheel", drivewheel)}
    {row("Engine Type", enginetype)}
    {row("Engine Location", enginelocation)}
    {row("Cylinders", cylindernumber)}
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#2a2f3e;font-size:.72rem;padding:.4rem 0;'>
    Ridge Regression · StandardScaler · 27 features · CarPrice_Assignment.csv
    · Baselines dropped by get_dummies(drop_first=True):
    <em>convertible, 4wd, front, dohc, eight</em>
</div>
""", unsafe_allow_html=True)