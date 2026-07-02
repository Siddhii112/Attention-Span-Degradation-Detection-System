import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA
from recommendation import get_recommendation

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Attention Span Detector", layout="wide")
st.title("🎓 Attention Span Degradation Detection System")

# =========================
# LOAD MODELS
# =========================
rf = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le_target = joblib.load("models/label_encoder.pkl")

# =========================
# SIDEBAR
# =========================
page = st.sidebar.selectbox("Select Page", ["Prediction", "Analytics"])

st.sidebar.markdown("""
## 📘 About Project
This system predicts student attention using:
- Machine Learning (Random Forest)
- ARIMA Forecasting
- Behavioral Data Analysis
""")
# =========================
# 🔴 PREDICTION PAGE
# =========================
if page == "Prediction":

    st.header("📊 Predict Attention Span")

    col1, col2 = st.columns(2)

    with col1:
        time_spent = st.number_input("Time Spent (minutes)", 0, 300, 30)
        pages = st.number_input("Pages Visited", 0, 50, 5)
        video = st.number_input("Video Watched (%)", 0, 100, 50)

    with col2:
        clicks = st.number_input("Click Events", 0, 100, 10)
        assignment = st.number_input("Assignment Score", 0, 100, 70)

    if st.button("Predict"):
        st.session_state["prediction_done"] = True

    if st.session_state["prediction_done"]:

        # Prediction
        input_data = np.array([[time_spent, pages, video, clicks, assignment]])
        input_scaled = scaler.transform(input_data)

        pred = rf.predict(input_scaled)[0]
        label = le_target.inverse_transform([pred])[0]

        # Rule fix
        if time_spent < 15 and pages < 3 and assignment < 50:
            label = "Low"

        # UI
        if label == "Low":
            color, progress = "red", 30
        elif label == "Medium":
            color, progress = "orange", 60
        else:
            color, progress = "green", 90

        st.markdown(f"<h2 style='color:{color};'>Attention Level: {label}</h2>", unsafe_allow_html=True)
        st.progress(progress)

        # Recommendations
        st.subheader("💡 Recommendations")
        rec = get_recommendation(label)

        if label == "Low":
            st.error(rec)
        elif label == "Medium":
            st.warning(rec)
        else:
            st.success(rec)

        # =========================
        # 📝 FEEDBACK SYSTEM
        # =========================
        st.subheader("📝 Give Feedback")

        rating = st.slider("Rate this prediction (1-5)", 1, 5, 3)
        feedback = st.text_area("Your feedback")

        if st.button("Submit Feedback"):

            feedback_file = "feedback.csv"

            feedback_df = pd.DataFrame({
                "time_spent": [time_spent],
                "pages": [pages],
                "video": [video],
                "clicks": [clicks],
                "assignment": [assignment],
                "prediction": [label],
                "rating": [rating],
                "feedback": [feedback]
            })

            feedback_df.to_csv(
                feedback_file,
                mode='a',
                header=not os.path.exists(feedback_file),
                index=False,
                quoting=1
            )

            st.success("✅ Feedback saved successfully!")
# =========================
# 📈 ANALYTICS PAGE
# =========================
else:

    st.header("📈 Student Analytics & Forecast")

    # Load dataset
    df = pd.read_csv("student_learning_interaction_dataset.csv")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Fix timestamp
    df['timestamp'] = df['timestamp'].astype(str)
    df['timestamp'] = df['timestamp'].str.replace('.', ':', regex=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%d-%m-%Y %H:%M", errors='coerce')

    df = df.dropna(subset=['timestamp'])
    df = df.sort_values(by=["student_id", "timestamp"])

    # Select student
    student_ids = df['student_id'].unique()
    selected_student = st.selectbox("Select Student", student_ids)

    student_data = df[df['student_id'] == selected_student]

    # Time series prep
    student_data = student_data.set_index('timestamp')
    student_data = student_data[['attention_score']]
    student_data = student_data.resample('D').mean()
    student_data['attention_score'] = student_data['attention_score'].interpolate()

    attention = student_data['attention_score']

    # Historical
    st.subheader("📊 Historical Attention")
    st.line_chart(attention)

    st.info("""
    📊 **Historical Attention Explanation**

    - Shows past attention trend over time  
    - Helps identify patterns and sudden drops  
    - Useful for analyzing student behavior  
    """)
    

    # Forecast
    model = ARIMA(attention, order=(1, 0, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)

    st.subheader("🔮 Future Attention Forecast")
    st.line_chart(forecast)

    st.info("""
    🔮 **Future Forecast Explanation**

    -  Predicts future attention using ARIMA  
    - Based on past trends  
    - Helps in early intervention  

    ⚠️ Predictions may vary in real scenarios 
    """)
    

    # =========================
    # 📊 FEEDBACK ANALYTICS
    # =========================

    st.subheader("📝 User Feedback Analysis")

    feedback_file = "feedback.csv"

    if os.path.exists(feedback_file):
        try:
            feedback_df = pd.read_csv("feedback.csv")

            st.metric("⭐ Average Rating", round(feedback_df["rating"].mean(), 2))
    
            # ONLY KEEP REQUIRED COLUMNS
            feedback_table = feedback_df[["rating", "feedback"]]
            st.write("💬 Recent Feedback")
            st.dataframe(feedback_table)
           

        except:
            st.error("Error reading feedback file. Delete feedback.csv and retry.")

    else:
        st.info("No feedback available yet.")
        