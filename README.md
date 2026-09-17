# 🌱 Predictive Modeling for Agriculture

## Machine Learning-Based Crop Recommendation System

A machine learning project that analyzes soil measurements and recommends a suitable crop using a **Logistic Regression classification model**.

The project provides an interactive **Streamlit web application** where users can enter soil parameters such as Nitrogen (N), Phosphorus (P), Potassium (K), and soil pH. The application evaluates the soil data and generates a crop recommendation along with model confidence and feature-performance analysis.

---

##  Project Overview

Agricultural productivity is strongly influenced by soil conditions. Selecting a suitable crop based on the characteristics of the soil can help farmers make better-informed decisions.

This project explores how machine learning can be applied to soil measurements to classify suitable crop categories.

The project includes:

- Soil data analysis
- Feature evaluation using F1-score
- Logistic Regression classification
- Interactive Streamlit interface
- Crop recommendation based on the trained model
- Candidate probability visualization
- Feature predictive-performance visualization

---

##  Objectives

The main objectives of this project are:

- Analyze soil measurements relevant to crop selection.
- Evaluate the predictive performance of individual soil features.
- Build a machine learning classification model.
- Develop an easy-to-use interface for entering soil measurements.
- Provide a crop recommendation based on the trained model.
- Present model results and feature performance visually.

---

##  Soil Parameters

The application accepts four soil-related parameters:

| Parameter | Description |
|-----------|-------------|
| **Nitrogen (N)** | Nitrogen level in the soil |
| **Phosphorus (P)** | Phosphorus level in the soil |
| **Potassium (K)** | Potassium level in the soil |
| **pH** | Soil acidity/alkalinity level |

Although all four parameters are collected by the application, the **currently deployed prediction model uses Potassium (K) as its prediction feature**.

The other features are evaluated separately to compare their predictive performance.

---

##  Machine Learning Approach

### Logistic Regression

The project uses **Logistic Regression** for multi-class crop classification.

The deployed prediction pipeline is trained using Potassium:

```python
X = crops[["K"]]
y = crops["crop"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
