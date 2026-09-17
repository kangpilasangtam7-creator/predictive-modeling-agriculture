import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(
    page_title="Predictive Modeling for Agriculture",
    page_icon="🌱",
    layout="centered"
)

# Title
st.title("🌱 Predictive Modeling for Agriculture")
st.write(
    "Enter the soil measurements below to predict the most suitable crop."
)

# Load dataset
crops = pd.read_csv("soil_measures.csv")

# Features and target
X = crops[["N", "P", "K", "ph"]]
y = crops["crop"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Input section
st.subheader("🌾 Soil Parameters")

N = st.number_input(
    "Nitrogen (N)",
    min_value=0.0,
    value=50.0
)

P = st.number_input(
    "Phosphorous (P)",
    min_value=0.0,
    value=50.0
)

K = st.number_input(
    "Potassium (K)",
    min_value=0.0,
    value=50.0
)

ph = st.number_input(
    "pH",
    min_value=0.0,
    max_value=14.0,
    value=6.5
)

# Prediction
if st.button("🌱 Predict Crop"):

    input_data = pd.DataFrame(
        [[N, P, K, ph]],
        columns=["N", "P", "K", "ph"]
    )

    prediction = model.predict(input_data)

    st.success(
        f"Recommended Crop: **{prediction[0]}**"
    )