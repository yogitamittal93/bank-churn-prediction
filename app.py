import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# Only load the two working models
MODEL_FILES = {
    "Random Forest": "model1.pkl",
    "Gradient Boosting": "model3.pkl"
}


# ==========================================
# 2. LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    models = {}

    for name, filename in MODEL_FILES.items():

        file_path = BASE_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {filename}")

        models[name] = joblib.load(file_path)

    return models


@st.cache_resource
def load_correlation():

    path = BASE_DIR / "corr_matrix.joblib"

    if path.exists():
        return joblib.load(path)

    return None


try:
    models = load_models()
    corr_matrix = load_correlation()

except Exception as e:
    st.error(f"Asset loading error: {e}")
    st.stop()


# ==========================================
# 3. SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Model Configuration")

primary_model = st.sidebar.selectbox(
    "Select Primary Model",
    list(models.keys())
)

compare_models = st.sidebar.checkbox(
    "Compare Both Models",
    value=True
)

st.sidebar.divider()
st.sidebar.markdown("### Available Models")

for name in models:
    st.sidebar.success(name)


# ==========================================
# 4. HEADER
# ==========================================

st.title("🏦 Bank Customer Churn Prediction")

st.markdown(
    "Enter customer details to predict the probability "
    "of customer churn using Machine Learning."
)

tab1, tab2 = st.tabs([
    "🔮 Churn Prediction",
    "📊 Correlation Analysis"
])


# ==========================================
# 5. CUSTOMER INPUT
# ==========================================

with tab1:

    st.subheader("Customer Information")

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            credit_score = st.number_input(
                "Credit Score", 300, 850, 650
            )

            geography = st.selectbox(
                "Geography",
                ["France", "Spain", "Germany"]
            )

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            age = st.number_input(
                "Age", 18, 100, 42
            )

        with col2:

            tenure = st.slider(
                "Tenure (Years)", 0, 10, 5
            )

            balance = st.number_input(
                "Account Balance ($)",
                min_value=0.0,
                max_value=300000.0,
                value=85000.0,
                step=500.0
            )

            products = st.selectbox(
                "Number of Products",
                [1, 2, 3, 4]
            )

        with col3:

            credit_card = st.selectbox(
                "Has Credit Card?",
                [1, 0],
                format_func=lambda x: "Yes" if x else "No"
            )

            active_member = st.selectbox(
                "Active Member?",
                [1, 0],
                format_func=lambda x: "Yes" if x else "No"
            )

            salary = st.number_input(
                "Estimated Salary ($)",
                min_value=0.0,
                max_value=250000.0,
                value=100000.0,
                step=1000.0
            )

        submitted = st.form_submit_button(
            "Predict Churn",
            type="primary",
            use_container_width=True
        )


    # ==========================================
    # 6. PREDICTION
    # ==========================================

    if submitted:

        input_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": credit_card,
            "IsActiveMember": active_member,
            "EstimatedSalary": salary
        }])

        st.divider()

        results = {}
        errors = {}

        for name, model in models.items():

            try:

                prediction = int(model.predict(input_df)[0])

                probability = None

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(input_df)[0]

                    classes = list(model.classes_)

                    if 1 in classes:
                        probability = float(
                            probabilities[classes.index(1)]
                        )

                results[name] = {
                    "prediction": prediction,
                    "probability": probability
                }

            except Exception as e:

                errors[name] = str(e)


        # ==========================================
        # 7. PRIMARY MODEL RESULT
        # ==========================================

        st.subheader(f"Primary Result: {primary_model}")

        if primary_model in results:

            result = results[primary_model]

            prediction = result["prediction"]
            probability = result["probability"]

            c1, c2 = st.columns(2)

            with c1:

                if probability is not None:
                    st.metric(
                        "Churn Probability",
                        f"{probability * 100:.2f}%"
                    )
                else:
                    st.metric("Prediction", prediction)

            with c2:

                if prediction == 1:
                    st.error("⚠️ Customer is likely to CHURN")
                else:
                    st.success("✅ Customer is likely to STAY")

            if probability is not None:
                st.progress(probability)


        # ==========================================
        # 8. MODEL COMPARISON
        # ==========================================

        if compare_models:

            st.divider()
            st.subheader("Model Comparison")

            comparison = []

            for name, result in results.items():

                comparison.append({
                    "Model": name,
                    "Prediction": (
                        "Churn" if result["prediction"] == 1 else "Stay"
                    ),
                    "Churn Probability": (
                        f'{result["probability"] * 100:.2f}%'
                        if result["probability"] is not None
                        else "Unavailable"
                    )
                })

            if comparison:

                st.dataframe(
                    pd.DataFrame(comparison),
                    use_container_width=True,
                    hide_index=True
                )

                churn_votes = sum(
                    r["prediction"] == 1
                    for r in results.values()
                )

                st.metric(
                    "Models Predicting Churn",
                    f"{churn_votes}/{len(results)}"
                )

                if churn_votes > len(results) / 2:
                    st.warning("Majority prediction: Customer Churn")
                else:
                    st.success("Majority prediction: Customer Retention")


        # ==========================================
        # 9. ERRORS
        # ==========================================

        if errors:

            with st.expander("Model Errors"):

                for name, error in errors.items():
                    st.error(f"{name}: {error}")


# ==========================================
# 10. CORRELATION MATRIX
# ==========================================

with tab2:

    st.subheader("Feature Correlation Heatmap")

    if corr_matrix is not None:

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            ax=ax
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        st.pyplot(fig)
        plt.close(fig)

        if "Exited" in corr_matrix.columns:

            st.subheader("Top Features Associated with Churn")

            target_corr = (
                corr_matrix["Exited"]
                .drop("Exited")
                .sort_values(
                    key=lambda x: x.abs(),
                    ascending=False
                )
            )

            st.dataframe(
                target_corr.rename("Correlation").to_frame(),
                use_container_width=True
            )

    else:

        st.info("Correlation matrix not found.")

st.sidebar.divider()
st.sidebar.caption("Bank Churn Prediction | ML Dashboard")
