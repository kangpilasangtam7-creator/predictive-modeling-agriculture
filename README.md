# Predictive Modeling for Agriculture

## Project Overview

This project uses machine learning to predict a suitable crop based on soil measurements.

## Features

The model uses the following soil parameters:

- Nitrogen (N)
- Phosphorous (P)
- Potassium (K)
- pH

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- Streamlit

## Dataset

The project uses the `soil_measures.csv` dataset containing soil measurements and crop labels.

## Machine Learning

Logistic Regression is used to train a multi-class classification model and predict the most suitable crop based on the given soil parameters.

## Web Application

A Streamlit interface allows users to enter soil measurements and receive a crop prediction.

## Project Structure

```text
Predictive Modeling for Agriculture/
├── app.py
├── notebook.ipynb
├── soil_measures.csv
├── farmer_in_a_field.jpg
├── requirements.txt
└── README.md
