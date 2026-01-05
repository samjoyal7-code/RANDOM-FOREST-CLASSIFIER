import streamlit as st
import pandas as pd
import joblib

# Load saved model and encoder
model = joblib.load("random_forest_titanic.pkl")
le = joblib.load("sex_label_encoder.pkl")

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings / Spouses aboard", min_value=0, value=0)
parch = st.number_input("Parents / Children aboard", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)

# Predict button
if st.button("Predict Survival"):
    
    # Encode sex
    sex_encoded = le.transform([sex])[0]
    
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex_encoded,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare
    }])
    
    # Prediction
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("🎉 Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is likely to NOT SURVIVE")
