import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Predictive Modeling for Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #f7faf7;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1220px;
    }

    /* Top Navigation Header */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 22px;
        background: #ffffff;
        border: 1px solid #e1ece3;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(18, 55, 33, 0.04);
        margin-bottom: 22px;
    }

    .header-logo {
        font-size: 22px;
        font-weight: 800;
        color: #176b3a;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .header-badge {
        background: #eaf6ee;
        color: #1a713e;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }

    /* Prediction Result Banner */
    .prediction-card {
        background: linear-gradient(135deg, #edf8f0 0%, #ddf2e3 100%);
        border: 1.5px solid #a8dcba;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .prediction-status {
        color: #276941;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .prediction-title {
        color: #104e29;
        font-size: 40px;
        font-weight: 800;
        line-height: 1.1;
        margin: 6px 0;
    }

    .prediction-desc {
        color: #496353;
        font-size: 14px;
        line-height: 1.4;
    }

    /* Primary Predict Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(180deg, #238b50 0%, #176b3a 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px;
        font-size: 16px;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(23, 107, 58, 0.22);
        transition: all 0.2s ease-in-out;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(23, 107, 58, 0.32);
        color: #ffffff;
    }

    /* Info Callout Box */
    .tip-box {
        background: #ffffff;
        border-left: 4px solid #176b3a;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13px;
        color: #465f50;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        margin-top: 15px;
    }

    .footer {
        margin-top: 40px;
        padding: 18px;
        border-top: 1px solid #e1ece3;
        color: #6c7f73;
        font-size: 13px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- CROP METADATA ----------------
CROP_MEDIA = {
    "Rice": {
        "image": "https://images.squarespace-cdn.com/content/v1/585280329de4bb2c6d6e31ea/1590605267035-ZNTNRHI2M3X87HDH3MNS/Photo+Sep+01%2C+2+04+24+PM.jpg?format=1500w",
        "description": "Thrives in heavy, clayey soils with standing water and steady potassium balance."
    },
    "Wheat": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQm_mrrVVrHupb0lV3DrSMHT9mTJohYESKEGmnnZ7g1kw&s=10",
        "description": "Prefers well-drained, fertile loamy soil with neutral to mildly acidic pH ranges."
    },
    "Maize": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTWSlyXTwMHkB0OXQW2kISs08QFuD_-lfo-VM97tZiqNw&s=10",
        "description": "Requires deep, loose soils with balanced micro-nutrients and high organic content."
    },
    "Cotton": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ6-M--qXxIwE7v90hBXMHXGEhqBY54PpghN-xS8is37w&s=10",
        "description": "Needs fertile, high-cation-exchange soils with substantial potassium availability."
    },
    "Jute": {
        "image": "https://gojuteinternational.com/wp-content/uploads/2024/01/Jute-Cultivation-jpg.webp",
        "description": "Thrives in warm, humid alluvial river basins with high annual rainfall."
    },
    "Coffee": {
        "image": "https://c8.alamy.com/comp/KA68H1/coffee-beans-in-jute-sack-and-cup-isolated-on-white-background-KA68H1.jpg",
        "description": "Flourishes in shaded, rich volcanic loam with mildly acidic pH levels."
    },
    "Kidneybeans": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTSiAN1Qbm1l0dim15uJmw17sh5qAu8B46AAQUdAcilouojH2twY9dhuPzT&s=10",
        "description": "Requires well-aerated, light loamy soil; sensitive to standing water and over-saturation."
    },
    "Pigeonpeas": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQdLn5NKRE7QyiLAMz_vegaAvswezzdHXeAUtMOTDa1Yg&s=10",
        "description": "Drought-hardy legume that thrives in semi-arid regions with well-drained soil."
    },
    "Mothbeans": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlLu7o1PQkw1d2qgtbTTS5Lekh2WjgI3eoQDZGmSaJBg&s=10",
        "description": "Exceptionally drought-resistant, growing well in arid zones and sandy soils."
    },
    "Mungbean": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTNLM_xdF0uKI8BFJcU14f3Gv4nAd8t706sDENra5EZwA&s=10",
        "description": "Warm-season crop that improves soil fertility through nitrogen fixation."
    },
    "Blackgram": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRgK2vpErHZJCANJH1zPp3IYfrPFW8DLLWK3UMCrSI_OA&s=10",
        "description": "Performs well in heavier soils with good moisture retention during initial germination."
    },
    "Lentil": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSGZVyQMk0j3s14cQ_sTAYVMw-6LJhQBNBykBPwoSVxcA&s=10",
        "description": "Prefers cool conditions and well-drained soils; sensitive to waterlogged root beds."
    },
    "Pomegranate": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTQ-volqy3EJwx71kjGzCbwYJjozL2INg9j6k6TIF7ogA&s=10",
        "description": "Prefers semi-arid climates and can tolerate mildly alkaline or poor soil quality."
    },
    "Banana": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRO_y__vpj8mtO1ULGbxfJGLWFdbIWLjaioDHLQQkXBOA&s=10",
        "description": "Heavy consumer of soil potassium, requiring rich, humid organic beds."
    },
    "Mango": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYtYg7ho020RAqDmJ4_Rk2dqkbjVSyd0ZeROFhjw4cOA&s=10",
        "description": "Requires deep alluvial or red loamy soils with a distinct dry period for flowering."
    },
    "Grapes": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQAhHIOsJ4x8xjZ-tSOPPo_QS2OZcINoV6JOzf4pVRyzA&s=10",
        "description": "Demands well-drained gravelly or sandy loam to encourage deep root establishment."
    },
    "Watermelon": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTRI6Aa5l3EkcBVWx6NrMG3mAazMvVtK5ij3t3YZwElDQ&s=10",
        "description": "Fast-growing vine crop requiring light, sandy, nutrient-rich beds with full sun."
    },
    "Muskmelon": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWA2xJLdPQ5Jsez0JQnzgPW1DzNUqWMytQLxIZ4LeBWQ&s=10",
        "description": "Thrives best in warm climates with loose, sandy loam and moderate moisture."
    },
    "Apple": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTnG-27iHa-BvKpyD43OGvOdRL4R-XZ6LScs5McIw3w4A&s",
        "description": "Requires temperate cold chill hours and deep, well-drained acidic to neutral loams."
    },
    "Orange": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSGrSq3zODKAQvENu1XhPHw2vlt-q_0thPT7JdaDGYRJA&s=10",
        "description": "Citrus crop needing ample sunlight and permeable soil with balanced drainage."
    },
    "Papaya": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQEHe7gnCnkDgUibUaoxvx0dqNlMXQH99rPksJqN8D1Ig&s=10",
        "description": "Fast-fruiting tropical tree requiring rich, porous loams without water stagnation."
    },
    "Coconut": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR2liJjj2cALAwQFLLoNDbZV3xb69hsCp4UN_MFQPiUuw&s=10",
        "description": "Suited for coastal saline-tolerant soils, sandy beaches, and high humidity."
    },
    "Default": {
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSuAEGOqHjZ04Lgj4dYTXmNYNePL9Td7ZrlTFNiD-cB4Q&s=10",
        "description": "Nutrient profile indicates favorable conditions for general cereal and field rotations."
    }
}

# ---------------- DATA + MODEL LOGIC ----------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("soil_measures.csv")
    except FileNotFoundError:
        np.random.seed(42)
        n_samples = 400
        return pd.DataFrame({
            "N": np.random.uniform(10, 140, n_samples),
            "P": np.random.uniform(5, 100, n_samples),
            "K": np.random.uniform(10, 120, n_samples),
            "ph": np.random.uniform(4.5, 8.5, n_samples),
            "crop": np.random.choice(["Rice", "Maize", "Wheat", "Cotton"], n_samples)
        })

crops = load_data()

# 1. Prediction Model (Trained on K)
X = crops[["K"]]
y = crops["crop"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 2. Notebook Feature Benchmark (F1-score comparison)
@st.cache_data
def compute_feature_performance(data: pd.DataFrame):
    X_all = data[["N", "P", "K", "ph"]]
    y_all = data["crop"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    performance = {}
    for feature in ["N", "P", "K", "ph"]:
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train[[feature]], y_train)
        y_pred = log_reg.predict(X_test[[feature]])
        f1 = metrics.f1_score(y_test, y_pred, average="weighted")
        performance[feature] = f1

    return performance

feature_performance = compute_feature_performance(crops)
best_feature = max(feature_performance, key=feature_performance.get)

# ---------------- SESSION STATE INITIALIZATION ----------------
if "has_predicted" not in st.session_state:
    st.session_state.has_predicted = False
    st.session_state.last_prediction = None
    st.session_state.last_probs = None
    st.session_state.last_potassium = None

# ---------------- HEADER ----------------
st.markdown("""
<div class="app-header">
    <div class="header-logo">🌿 AgriPredict Intelligence</div>
    <div class="header-badge">Logistic Regression Classifier</div>
</div>
""", unsafe_allow_html=True)

# ---------------- HERO BANNER ----------------
hero_col1, hero_col2 = st.columns([1.25, 1], gap="medium")
with hero_col1:
    with st.container(border=True):
        st.subheader("🌾 Predictive Soil Modeling for Agriculture")
        st.write(
            "Input laboratory soil nutrients and chemical metrics below. The platform leverages "
            "a multi-class classification model trained on calibrated soil data to determine "
            "the optimal crop suitability for your land."
        )
        st.caption(f"Benchmark analysis confirms **{best_feature}** as the primary single predictor.")

with hero_col2:
    try:
        st.image("farmer_in_a_field.jpg", use_container_width=True)
    except Exception:
        st.image(
            "https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=800&auto=format&fit=crop&q=60",
            use_container_width=True
        )

st.write("")

# ---------------- WORKSPACE ----------------
left_col, right_col = st.columns([1, 1], gap="large")

# ================= LEFT: INPUTS =================
with left_col:
    with st.container(border=True):
        st.markdown("### 🌱 Soil Chemistry Parameters")
        st.caption("Adjust sliders or enter values to test crop response.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🟢 Nitrogen (N)**")
            nitrogen = st.slider("Nitrogen level", 0.0, 200.0, 50.0, 1.0, label_visibility="collapsed")
            st.caption(f"Current: `{nitrogen:g}` ppm")

        with c2:
            st.markdown("**🟣 Phosphorus (P)**")
            phosphorous = st.slider("Phosphorus level", 0.0, 200.0, 30.0, 1.0, label_visibility="collapsed")
            st.caption(f"Current: `{phosphorous:g}` ppm")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**🟡 Potassium (K)** *(Primary Feature)*")
            potassium = st.slider("Potassium level", 0.0, 200.0, 40.0, 1.0, label_visibility="collapsed")
            st.caption(f"Current: `{potassium:g}` ppm")

        with c4:
            st.markdown("**🔵 Soil pH**")
            ph = st.slider("Soil pH level", 0.0, 14.0, 6.5, 0.1, label_visibility="collapsed")
            st.caption(f"Current: `{ph:.1f}` pH")

        st.write("")
        predict_btn = st.button("🌱 Predict Recommended Crop")

        if predict_btn:
            input_data = pd.DataFrame({"K": [potassium]})
            raw_pred = model.predict(input_data)[0]
            probs = model.predict_proba(input_data)[0]

            st.session_state.has_predicted = True
            st.session_state.last_prediction = str(raw_pred).capitalize()
            st.session_state.last_probs = probs
            st.session_state.last_potassium = potassium

        st.markdown("""
        <div class="tip-box">
            <b>Pro Tip:</b> Even though all 4 metrics are captured for context, the active inference pipeline utilizes 
            <b>Potassium (K)</b> as identified during feature evaluation.
        </div>
        """, unsafe_allow_html=True)

# ================= RIGHT: PREDICTION & BENCHMARK =================
with right_col:
    with st.container(border=True):
        st.markdown("### 🌾 Crop Suitability Prediction")

        if st.session_state.has_predicted:
            predicted_crop = st.session_state.last_prediction
            probabilities = st.session_state.last_probs
            used_k = st.session_state.last_potassium
            confidence = float(np.max(probabilities) * 100)
            crop_meta = CROP_MEDIA.get(predicted_crop, CROP_MEDIA["Default"])

            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-status">Optimal Match Found</div>
                <div class="prediction-title">{predicted_crop}</div>
                <div class="prediction-desc">{crop_meta['description']}</div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Input Potassium", f"{used_k:g}")
            m2.metric("Confidence", f"{confidence:.1f}%")
            m3.metric("Key Predictor", best_feature.upper())

            st.write("")
            st.image(crop_meta["image"], caption=f"Representative crop: {predicted_crop}", use_container_width=True)

            classes = [str(c).capitalize() for c in model.classes_]
            prob_df = pd.DataFrame({"Crop": classes, "Score": probabilities})
            top_candidates = prob_df.sort_values(by="Score", ascending=False).head(3)

            st.markdown("**Candidate Probabilities:**")
            for _, row in top_candidates.iterrows():
                st.write(f"{row['Crop']} (`{row['Score'] * 100:.1f}%`)")
                st.progress(float(row["Score"]))

        else:
            st.info("Enter your soil values and click **Predict Recommended Crop** to generate an evaluation.")
            st.image(
                "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSuaGtZSxhLyzGt30HGLYzXk_1nLGvkKNzoWl9dy0hTSA&s=10",
                caption="Awaiting soil data submission",
                use_container_width=True
            )

    # ---------------- BENCHMARK CHART ----------------
    with st.container(border=True):
        st.markdown("### 📊 Feature Predictive Performance")
        st.caption("Weighted F1-score comparisons from individual logistic regression models.")

        chart_df = pd.DataFrame({
            "Feature": list(feature_performance.keys()),
            "F1 Score": [round(val, 4) for val in feature_performance.values()]
        }).set_index("Feature")

        st.bar_chart(chart_df, color="#7dcd9e79")
        st.info(f"**{best_feature}** yielded the highest weighted F1-score across independent single-variable splits.")

# ---------------- ABOUT THE MODEL ----------------
with st.container(border=True):
    st.markdown("### 🏛️ Model Architecture & Methodology")
    st.write(
        f"This crop recommendation system uses a **multi-class Logistic Regression** model. "
        f"During exploratory benchmarking, each available feature (`N`, `P`, `K`, `ph`) was evaluated individually. "
        f"Feature **{best_feature}** achieved the highest weighted F1-score, confirming it as the strongest individual "
        f"predictor for identifying crop classes without multi-collinearity interference."
    )

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    AgriPredict Modeling Platform &nbsp;•&nbsp; Built with Python, Streamlit &amp; Scikit-learn
</div>
""", unsafe_allow_html=True)
