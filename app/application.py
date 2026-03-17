import streamlit as st
import pickle as pkl
import numpy as np
import plotly.graph_objects as go


age_labels = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80 or older"
}


def main():
    try:
        with open("../model/model.pkl", "rb") as f:
            model = pkl.load(f)
    except FileNotFoundError:
        st.error("Model file not found at '../model/model.pkl'.")
        return

    st.set_page_config(page_title="Diabetes Risk Assistant", layout="wide")

    st.title("Diabetes Risk Assistant")

    results_container = st.container()

    st.markdown("---")

    with st.form("main_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Medical Data")
            high_bp = st.selectbox("High Blood Pressure?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            high_chol = st.selectbox("High Cholesterol?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            chol_check = st.selectbox("Cholesterol Check in last 5 years?", [1, 0],
                                      format_func=lambda x: "Yes" if x == 1 else "No")
            bmi = st.number_input("BMI", min_value=10, max_value=100, value=25)
            stroke = st.selectbox("History of Stroke?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            heart_dis = st.selectbox("Heart Disease or Attack?", [0, 1],
                                     format_func=lambda x: "Yes" if x == 1 else "No")
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            age = st.select_slider(
                "Age",
                options=list(age_labels.keys()),
                format_func=lambda x: age_labels[x],
                value=7
            )

        with col2:
            st.subheader("Lifestyle & Socioeconomics")
            smoker = st.selectbox("Smoker?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            hvy_alcohol = st.selectbox("Heavy Alcohol Consumption?", [0, 1],
                                       format_func=lambda x: "Yes" if x == 1 else "No")
            phys_act = st.selectbox("Physical Activity in last 30 days?", [1, 0],
                                    format_func=lambda x: "Yes" if x == 1 else "No")
            fruits = st.selectbox("Eat Fruit Daily?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            veggies = st.selectbox("Eat Vegetables Daily?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            gen_hlth = st.slider("General Health (1-5)", 1, 5, 3)
            ment_hlth = st.slider("Mental Health Bad Days (0-30)", 0, 30, 0)
            phys_hlth = st.slider("Physical Health Bad Days (0-30)", 0, 30, 0)
            diff_walk = st.selectbox("Difficulty Walking?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

            healthcare = 1
            no_doc_cost = 0
            education = 4
            income = 5

        submit = st.form_submit_button("Analyze Risk")

    if submit:
        features = np.array([[
            high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_dis,
            phys_act, fruits, veggies, hvy_alcohol, healthcare, no_doc_cost,
            gen_hlth, ment_hlth, phys_hlth, diff_walk, sex, age, education, income
        ]])

        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        categories = ['BMI', 'Gen Health', 'Ment Health', 'Phys Health', 'Age', 'Income']
        user_values = [
            min(bmi / 50, 1.0),
            (6 - gen_hlth) / 5,
            (ment_hlth) / 30,
            (phys_hlth) / 30,
            age / 13,
            income / 8
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=user_values,
            theta=categories,
            fill='toself',
            name='User Profile',
            line_color='#00f2ff'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=80, r=80, t=40, b=40),
            height=400
        )

        with results_container:
            res_col, chart_col = st.columns([1, 1])
            with res_col:
                st.subheader("Analysis Results")
                if prediction[0] == 1:
                    st.error("HIGH RISK")
                    st.write(f"Probability: **{probability:.2%}**")
                else:
                    st.success("LOW RISK")
                    st.write(f"Probability: **{probability:.2%}**")

            with chart_col:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")


if __name__ == "__main__":
    main()
