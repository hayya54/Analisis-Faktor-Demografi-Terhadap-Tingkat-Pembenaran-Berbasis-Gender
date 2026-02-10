import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import plotly.express as px
import plotly.graph_objects as go
from transformers import BertTokenizer, BertModel

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Justification Violence Analytics",
    page_icon="📊",
    layout="wide"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# CUSTOM CSS (UI ENHANCEMENT)
# ======================================================
st.markdown("""
<style>
body {background-color: #0E1117;}
.block-container {padding-top: 2rem;}
h1, h2, h3 {color: #FAFAFA;}
.card {
    background: linear-gradient(135deg,#1f2933,#111827);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
}
.info-low {background:#0f5132;padding:15px;border-radius:12px;}
.info-mid {background:#664d03;padding:15px;border-radius:12px;}
.info-high {background:#842029;padding:15px;border-radius:12px;}
.caption {color:#9CA3AF;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL & DATA
# ======================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_hybrid_v3.pkl")
    scaler = joblib.load("standard_scaler_v3.pkl")
    pca = joblib.load("pca_v3.pkl")
    encoders = joblib.load("label_encoders_v3.pkl")

    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    bert.eval()

    return model, scaler, pca, encoders, tok, bert

@st.cache_data
def load_data():
    df = pd.read_csv("Violence Against Women  Girls Data.csv")
    df["Survey_Year"] = pd.to_datetime(df["Survey Year"], errors="coerce").dt.year
    return df

model, scaler, pca_model, label_encoders, tokenizer, bert_model = load_artifacts()
df = load_data()

# ======================================================
# QUESTION OPTIONS
# ======================================================
QUESTION_OPTIONS = [
    "A husband is justified in hitting or beating his wife if she burns the food",
    "A husband is justified in hitting or beating his wife if she argues with him",
    "A husband is justified in hitting or beating his wife if she goes out without telling him",
    "A husband is justified in hitting or beating his wife if she neglects the children",
    "A husband is justified in hitting or beating his wife if she refuses to have sex with him",
    "A husband is justified in hitting or beating his wife for at least one specific reason"
]

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def get_bert_embedding(text):
    enc = tokenizer([text], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = bert_model(**enc)
    return out.last_hidden_state[:,0,:].cpu().numpy()

def predict(inputs):
    cols = ['Country','Gender','Demographics Question','Demographics Response','Survey_Year']
    enc_vals = []
    for c in cols:
        v = inputs[c]
        if c != "Survey_Year": v = str(v)
        le = label_encoders[c]
        enc_vals.append(le.transform([v])[0] if v in le.classes_ else le.transform([le.classes_[0]])[0])

    X_demo = scaler.transform(np.array(enc_vals).reshape(1,-1))
    X_text = pca_model.transform(get_bert_embedding(inputs["Question"]))
    X = np.hstack((X_demo, X_text))
    return float(np.clip(model.predict(X)[0],0,100))

def interpret(val):
    if val < 30:
        return "RENDAH", "info-low", "Penolakan terhadap kekerasan sangat kuat."
    elif val < 60:
        return "SEDANG", "info-mid", "Masih terdapat toleransi pada kondisi tertentu."
    elif val < 80:
        return "TINGGI", "info-high", "Pembenaran tinggi, perlu perhatian kebijakan."
    else:
        return "SANGAT TINGGI", "info-high", "Risiko normalisasi kekerasan sangat kuat."

# ======================================================
# SIDEBAR
# ======================================================
menu = st.sidebar.radio("Menu", ["📊 Dashboard", "📈 Visualisasi", "🔮 Prediksi"])

# ======================================================
# DASHBOARD
# ======================================================
if menu == "📊 Dashboard":
    st.markdown("## 📊 Ringkasan Dataset")
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Data", len(df))
    c2.metric("Negara", df.Country.nunique())
    c3.metric("Tahun Survey", df.Survey_Year.nunique())
    st.markdown('<p class="caption">Dataset global tentang pembenaran kekerasan berbasis gender.</p>', unsafe_allow_html=True)

# ======================================================
# VISUALISASI
# ======================================================
elif menu == "📈 Visualisasi":
    st.markdown("## 📈 Visualisasi & Insight")
    fig = px.histogram(df, x="Value", nbins=40)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<p class="caption">Distribusi nilai menunjukkan variasi sikap masyarakat.</p>', unsafe_allow_html=True)

    fig = px.box(df, x="Gender", y="Value")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<p class="caption">Perbedaan distribusi berdasarkan gender.</p>', unsafe_allow_html=True)

# ======================================================
# PREDIKSI
# ======================================================
else:
    st.markdown("## 🔮 Simulasi Prediksi Tingkat Pembenaran")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1,col2 = st.columns(2)

    with col1:
        country = st.selectbox("Country", sorted(df.Country.unique()))
        gender = st.selectbox("Gender", sorted(df.Gender.unique()))
        year = st.selectbox("Survey Year", sorted(df.Survey_Year.dropna().unique()))

    with col2:
        dq = st.selectbox("Demographics Question", sorted(df["Demographics Question"].unique()))
        dr = st.selectbox("Demographics Response",
                          sorted(df[df["Demographics Question"]==dq]["Demographics Response"].unique()))

    question = st.selectbox("Question", QUESTION_OPTIONS)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Prediksi Sekarang"):
        val = predict({
            "Country":country,"Gender":gender,
            "Demographics Question":dq,
            "Demographics Response":dr,
            "Survey_Year":year,"Question":question
        })

        level, style, desc = interpret(val)

        st.markdown(f"""
        <div class="{style}">
        <h3>Hasil: {level}</h3>
        <b>{val:.2f}%</b><br>
        {desc}
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            gauge={'axis': {'range': [0,100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="caption">Prediksi dihasilkan oleh model Hybrid BERT–LightGBM.</p>', unsafe_allow_html=True)
