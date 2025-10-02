# app.py

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------ Load Saved Files ------------------
lr = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")
le_y = joblib.load("label_encoder_y.pkl")

# ------------------ Streamlit UI ------------------
st.title("Your Breast Cancer Risk Advisor")
st.write("This app predicts **breast cancer risk** based on your lifestyle, medical history, and symptoms.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
family_cancer_history_count = st.number_input("Family Cancer History (0–3)", min_value=0, max_value=3, value=0)
menopause_status = st.selectbox("Menopause Status", options=["Pre-Menopause", "Post-Menopause"])
lump = st.selectbox("Any Lump?", options=["No", "Yes"])
chest_pain = st.selectbox("Chest Pain?", options=["No", "Yes"])
swelling = st.selectbox("Swelling?", options=["No", "Yes"])
skin_changes = st.selectbox("Skin Changes?", options=["No", "Yes"])
nipple_inversion = st.selectbox("Nipple Inversion?", options=["No", "Yes"])
nipple_discharge = st.selectbox("Nipple Discharge?", options=["No", "Yes"])
lymph_swelling = st.selectbox("Lymph Node Swelling?", options=["No", "Yes"])
weight = st.number_input("Weight (kg)", min_value=30, max_value=150, value=70)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
exercise_hours = st.number_input("Exercise Hours per Week", min_value=0, max_value=20, value=3)
smoking = st.selectbox("Do you smoke?", options=["No", "Yes"])
diet = st.selectbox("Diet Quality", options=["Unhealthy", "Healthy"])

if st.button("Submit"):

    # ------------------ Encode Inputs ------------------
    menopause_status_enc = 1 if menopause_status == "Post-Menopause" else 0
    lump_enc = 1 if lump == "Yes" else 0
    chest_pain_enc = 1 if chest_pain == "Yes" else 0
    swelling_enc = 1 if swelling == "Yes" else 0
    skin_changes_enc = 1 if skin_changes == "Yes" else 0
    nipple_inversion_enc = 1 if nipple_inversion == "Yes" else 0
    nipple_discharge_enc = 1 if nipple_discharge == "Yes" else 0
    lymph_swelling_enc = 1 if lymph_swelling == "Yes" else 0
    smoking_enc = 1 if smoking == "Yes" else 0
    diet_enc = 0 if diet == "Healthy" else 1

    # ------------------ Prepare Features ------------------
    features = np.array([[age, family_cancer_history_count, menopause_status_enc, lump_enc, chest_pain_enc,
                          swelling_enc, skin_changes_enc, nipple_inversion_enc, nipple_discharge_enc, lymph_swelling_enc,
                          weight, bmi, exercise_hours, smoking_enc, diet_enc]])

    # Scale numeric features
    numeric_indices = [0, 10, 11, 12]  # Age, Weight, BMI, Exercise
    features[:, numeric_indices] = scaler.transform(features[:, numeric_indices])

    # ------------------ Predict ------------------
    prediction = lr.predict(features)[0]
    probability = lr.predict_proba(features)[0][1]
    prediction_label = 'Yes' if prediction == 1 else 'No'

    # ------------------ Display Results ------------------
    st.subheader("Prediction Result")
    if prediction_label == "Yes":
        st.error("High risk detected. Please consult a doctor immediately.")
    else:
        st.success("Low risk detected. Keep monitoring and maintaining a healthy lifestyle.")
    st.write(f"**Estimated Risk Probability:** {probability * 100:.2f}%")

    # Probability Bar
    fig, ax = plt.subplots()
    ax.bar(["Estimated Risk"], [probability * 100], color="red" if prediction_label == "Yes" else "green")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    st.pyplot(fig)

    # ------------------ Personalized Recommendations ------------------
    if bmi < 18.5:
        st.warning("You are **Underweight** → Add high-protein, calorie-rich healthy foods (nuts, seeds, eggs, dairy).")
    elif 18.5 <= bmi < 25:
        st.success("Your weight is **Normal** → Maintain a balanced diet with fruits, vegetables, whole grains.")
    else:
        st.error("You are **Overweight/Obese** → Focus on a low-fat, high-fiber diet. Avoid fried/junk foods, excess red meat, and sugary drinks.")

    if exercise_hours < 5:
        st.warning("You exercise less than 5 hours/week → Increase regular physical activity to reduce cancer risk.")
    else:
        st.success("Your exercise routine is good. Keep it up!")

    if smoking_enc == 1:
        st.error("Smoking detected → Stop smoking immediately to reduce breast cancer risk.")
    else:
        st.success("Great! You don’t smoke.")

    if diet_enc == 1:
        st.warning("Your diet is **Unhealthy** → Add more vegetables, fruits, whole grains, and lean proteins.")
    else:
        st.success("Excellent! You already follow a healthy diet !!")
