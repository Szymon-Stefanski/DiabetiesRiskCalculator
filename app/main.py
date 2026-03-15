import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np


def main():
    with open("model.pkl", "rb") as f:
        model = pkl.load(f)

    st.set_page_config(page_title="CDC Diabetes Predictor", layout="centered")

    st.title("Diabetes Risk Assistant")
    st.write("Fill out the form below to get an AI-based prediction.")

    with st.form("main_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            high_bp = st.selectbox("High Blood Pressure?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            high_chol = st.selectbox("High Cholesterol?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            chol_check = st.selectbox("Cholesterol Check in last 5 years?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            bmi = st.number_input("BMI", min_value=10, max_value=100, value=25)
            smoker = st.selectbox("Smoker?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            stroke = st.selectbox("History of Stroke?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            heart_dis = st.selectbox("Heart Disease or Attack?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            phys_act = st.selectbox("Physical Activity in last 30 days?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            fruits = st.selectbox("Eat Fruit Daily?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            veggies = st.selectbox("Eat Vegetables Daily?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

        with col2:
            hvy_alcohol = st.selectbox("Heavy Alcohol Consumption?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            healthcare = st.selectbox("Have Healthcare?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            no_doc_cost = st.selectbox("No Doctor due to Cost?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            gen_hlth = st.slider("General Health (1-5)", 1, 5, 3)
            ment_hlth = st.slider("Mental Health Bad Days (0-30)", 0, 30, 0)
            phys_hlth = st.slider("Physical Health Bad Days (0-30)", 0, 30, 0)
            diff_walk = st.selectbox("Difficulty Walking?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            age = st.slider("Age Group (1-13)", 1, 13, 7)
            education = st.slider("Education Level (1-6)", 1, 6, 4)
            income = st.slider("Income Scale (1-8)", 1, 8, 5)

        submit = st.form_submit_button("Analyze")

    if submit:
        features = np.array([[
            high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_dis, 
            phys_act, fruits, veggies, hvy_alcohol, healthcare, no_doc_cost, 
            gen_hlth, ment_hlth, phys_hlth, diff_walk, sex, age, education, income
        ]])
        
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        st.subheader("Result:")
        if prediction[0] == 1:
            st.error(f"High Risk of Diabetes ({probability:.2%})")
        else:
            st.success(f"Low Risk of Diabetes ({probability:.2%})")


if __name__ == "__main__":
    main()
