import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Heart Risk Predictor",
    page_icon="❤️",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("❤️ AI-Based Heart Disease Risk Prediction")
st.write(
    "An educational AI system that predicts heart disease "
    "risk using a Random Forest classification model."
    "A Group Project Made By Priti,Khushi,Raj,Sachin"
)


# =====================================================
# SAMPLE DATASET
# =====================================================

data = {
    "Age": [
        25, 32, 45, 52, 61,
        29, 40, 55, 67, 36,
        48, 58, 23, 70, 43,
        51, 63, 31, 46, 57
    ],

    "BloodPressure": [
        110, 120, 140, 150, 160,
        115, 130, 145, 170, 125,
        135, 155, 105, 180, 128,
        148, 165, 118, 138, 152
    ],

    "Cholesterol": [
        170, 180, 230, 250, 270,
        160, 210, 240, 290, 190,
        220, 260, 155, 300, 205,
        245, 280, 175, 225, 255
    ],

    "BMI": [
        21, 23, 28, 30, 32,
        20, 25, 29, 34, 24,
        27, 31, 19, 35, 26,
        30, 33, 22, 28, 32
    ],

    "Smoking": [
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 0,
        1, 1, 0, 1, 0,
        1, 1, 0, 0, 1
    ],

    "PhysicalActivity": [
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 1,
        0, 0, 1, 0, 1,
        0, 0, 1, 1, 0
    ],

    "Risk": [
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 0,
        1, 1, 0, 1, 0,
        1, 1, 0, 0, 1
    ]
}


df = pd.DataFrame(data)


# =====================================================
# PREPARE DATA
# =====================================================

X = df.drop("Risk", axis=1)
y = df["Risk"]


# =====================================================
# TRAIN AND TEST DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# =====================================================
# CREATE RANDOM FOREST MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(X_train, y_train)


# =====================================================
# MODEL PREDICTION
# =====================================================

y_pred = model.predict(X_test)


# =====================================================
# MODEL PERFORMANCE
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# =====================================================
# PATIENT INFORMATION
# =====================================================

st.header("👤 Enter Patient Information")


col1, col2 = st.columns(2)


# -----------------------------
# LEFT COLUMN
# -----------------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40
    )

    bp = st.number_input(
        "Blood Pressure",
        min_value=80,
        max_value=220,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=400,
        value=200
    )


# -----------------------------
# RIGHT COLUMN
# -----------------------------

with col2:

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=50.0,
        value=24.0
    )

    smoking = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )

    activity = st.selectbox(
        "Regular Physical Activity",
        ["Yes", "No"]
    )


# =====================================================
# CONVERT INPUT VALUES
# =====================================================

smoking_value = 1 if smoking == "Yes" else 0

activity_value = 1 if activity == "Yes" else 0


# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button("🔍 Predict Heart Disease Risk"):

    input_data = pd.DataFrame([
        {
            "Age": age,
            "BloodPressure": bp,
            "Cholesterol": cholesterol,
            "BMI": bmi,
            "Smoking": smoking_value,
            "PhysicalActivity": activity_value
        }
    ])


    # -----------------------------
    # MAKE PREDICTION
    # -----------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # =================================================
    # PREDICTION RESULT
    # =================================================

    st.subheader("📋 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Higher Heart Disease Risk Detected"
        )

    else:

        st.success(
            "✅ Lower Heart Disease Risk Detected"
        )


    # =================================================
    # RISK PROBABILITY
    # =================================================

    st.write(
        f"Estimated Risk Probability: "
        f"**{probability * 100:.2f}%**"
    )


    # =================================================
    # RISK LEVEL
    # =================================================

    if probability < 0.35:

        risk_level = "LOW"

    elif probability < 0.65:

        risk_level = "MODERATE"

    else:

        risk_level = "HIGH"


    st.info(
        f"Risk Level: **{risk_level}**"
    )


    # =================================================
    # PROGRESS BAR
    # =================================================

    st.progress(
        float(probability)
    )


    # =================================================
    # EXPLAINABLE AI
    # =================================================

    st.subheader("🔎 Important Factors")


    importance = pd.DataFrame(
        {
            "Factor": X.columns,
            "Importance": model.feature_importances_
        }
    )


    importance = importance.sort_values(
        "Importance",
        ascending=False
    )


    st.bar_chart(
        importance.set_index("Factor")
    )


    # =================================================
    # SHOW FACTOR TABLE
    # =================================================

    st.write("Feature Importance:")

    st.dataframe(
        importance,
        use_container_width=True
    )


# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.header("📊 AI Model Performance")


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with c2:

    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )


with c3:

    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )


with c4:

    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )


# =====================================================
# DATASET PREVIEW
# =====================================================

st.header("📁 Training Dataset")


with st.expander("View Training Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )


# =====================================================
# PROJECT INFORMATION
# =====================================================

st.header("ℹ️ About This Project")

st.write(
    """
    This project uses a Random Forest Classification algorithm
    to predict heart disease risk based on age, blood pressure,
    cholesterol, BMI, smoking habits and physical activity.
    
    The system also displays risk probability, risk level,
    feature importance and model performance metrics.
    """
)


# =====================================================
# DISCLAIMER
# =====================================================
st.warning(
    "⚠️ This project is created for educational purposes only. "
    "It is not a medical diagnosis and should not replace "
    "professional medical advice."
)
