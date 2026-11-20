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
    page_title="Bank Churn Prediction Dashboard",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Naive Bayes": "model1.pkl",
    #"XGBoost": "model2.pkl",
    "Gradient Boosting": "model3.pkl",
    "SVM": "model4.pkl"
}

FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]


# ==========================================
# 2. LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    models = {}

    for name, filename in MODEL_FILES.items():

        path = BASE_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {filename}"
            )

        models[name] = joblib.load(path)

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
    st.error(f"Error loading model assets: {e}")
    st.stop()


# ==========================================
# 3. SIDEBAR CONFIGURATION
# ==========================================

st.sidebar.title("⚙️ Model Configuration")

primary_model = st.sidebar.selectbox(
    "Select Primary Model",
    list(models.keys()),
    index=1
)

compare_models = st.sidebar.checkbox(
    "Compare All Four Models",
    value=True
)

st.sidebar.divider()

st.sidebar.subheader("Loaded Models")

for name in models:
    st.sidebar.success(name)


# ==========================================
# 4. MAIN HEADER
# ==========================================

st.title("🏦 Bank Customer Churn Prediction")

st.markdown(
    """
    An interactive Machine Learning dashboard that predicts
    customer churn using four classification algorithms.
    
    Enter customer details to generate predictions and compare
    model performance on an individual customer.
    """
)

tab1, tab2 = st.tabs([
    "🔮 Customer Prediction",
    "📊 Feature Correlation"
])


# ==========================================
# 5. CUSTOMER INPUT FORM
# ==========================================

with tab1:

    st.subheader("Enter Customer Information")

    with st.form("customer_prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            credit_score = st.number_input(
                "Credit Score",
                min_value=100,
                max_value=850,
                value=650
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
                "Age",
                min_value=18,
                max_value=100,
                value=42
            )

        with col2:

            tenure = st.slider(
                "Tenure (Years)",
                0, 10, 5
            )

            balance = st.number_input(
                "Account Balance ($)",
                min_value=0.0,
                max_value=3000000.0,
                value=85000.0,
                step=500.0
            )

            num_products = st.selectbox(
                "Number of Products",
                [1, 2, 3, 4]
            )

        with col3:

            has_cr_card = st.selectbox(
                "Has Credit Card?",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            active_member = st.selectbox(
                "Active Member?",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            salary = st.number_input(
                "Estimated Salary ($)",
                min_value=0.0,
                max_value=25000000.0,
                value=100000.0,
                step=1000.0
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            type="primary",
            use_container_width=True
        )


    # ==========================================
    # 6. RUN PREDICTIONS
    # ==========================================

    if submitted:

        input_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": active_member,
            "EstimatedSalary": salary
        }], columns=FEATURES)

        st.divider()

        st.subheader("Customer Profile")

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True
        )

        results = {}
        errors = {}

        for name, model in models.items():

            try:

                prediction = model.predict(input_df)[0]

                # Find churn probability for class 1
                probability = None

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(input_df)[0]

                    classes = list(model.classes_)

                    if 1 in classes:

                        probability = float(
                            probabilities[classes.index(1)]
                        )

                results[name] = {
                    "prediction": int(prediction),
                    "probability": probability
                }

            except Exception as e:

                errors[name] = str(e)


        # ==========================================
        # 7. PRIMARY MODEL RESULT
        # ==========================================

        st.subheader(f"Primary Model: {primary_model}")

        if primary_model in results:

            result = results[primary_model]

            prediction = result["prediction"]
            probability = result["probability"]

            col1, col2 = st.columns(2)

            with col1:

                if probability is not None:

                    st.metric(
                        "Churn Probability",
                        f"{probability * 100:.2f}%"
                    )

                else:

                    st.metric(
                        "Prediction",
                        "Churn" if prediction == 1 else "Stay"
                    )

            with col2:

                if prediction == 1:

                    st.error(
                        "⚠️ Customer is predicted to CHURN"
                    )

                else:

                    st.success(
                        "✅ Customer is predicted to STAY"
                    )

            if probability is not None:
                st.progress(probability)

        else:

            st.error(
                f"Primary model failed: {errors.get(primary_model)}"
            )


        # ==========================================
        # 8. FOUR-MODEL COMPARISON
        # ==========================================

        if compare_models:

            st.divider()

            st.subheader("⚖️ Four-Model Comparison")

            comparison_data = []

            for name, result in results.items():

                prediction = result["prediction"]
                probability = result["probability"]

                comparison_data.append({

                    "Model": name,

                    "Prediction": (
                        "Churn" if prediction == 1 else "Stay"
                    ),

                    "Churn Probability": (
                        f"{probability * 100:.2f}%"
                        if probability is not None
                        else "Not Available"
                    )
                })

            if comparison_data:

                st.dataframe(
                    pd.DataFrame(comparison_data),
                    use_container_width=True,
                    hide_index=True
                )

                # Majority voting
                churn_votes = sum(
                    result["prediction"] == 1
                    for result in results.values()
                )

                total_votes = len(results)

                st.subheader("Model Consensus")

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "Churn Predictions",
                        f"{churn_votes}/{total_votes}"
                    )

                with c2:

                    st.metric(
                        "Retention Predictions",
                        f"{total_votes - churn_votes}/{total_votes}"
                    )

                if churn_votes > total_votes / 2:

                    st.warning(
                        "Majority of models predict customer churn."
                    )

                elif churn_votes < total_votes / 2:

                    st.success(
                        "Majority of models predict customer retention."
                    )

                else:

                    st.info(
                        "The models are equally divided. No majority consensus."
                    )


        # ==========================================
        # 9. MODEL ERRORS
        # ==========================================

        if errors:

            with st.expander("View Model Errors"):

                for name, error in errors.items():

                    st.error(f"{name}: {error}")


# ==========================================
# 10. FEATURE CORRELATION
# ==========================================

with tab2:

    st.subheader("Feature Correlation Heatmap")

    if corr_matrix is not None:

        if isinstance(corr_matrix, pd.DataFrame):
            corr_df = corr_matrix
        else:
            corr_df = pd.DataFrame(corr_matrix)

        col1, col2 = st.columns([2, 1])

        with col1:

            fig, ax = plt.subplots(figsize=(10, 7))

            sns.heatmap(
                corr_df,
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

        with col2:

            st.subheader("Top Churn Correlations")

            if "Exited" in corr_df.columns:

                target_corr = (
                    corr_df["Exited"]
                    .drop("Exited")
                    .sort_values(
                        key=lambda x: x.abs(),
                        ascending=False
                    )
                )

                st.dataframe(
                    target_corr.rename(
                        "Correlation"
                    ).to_frame(),
                    use_container_width=True
                )

            else:

                st.info(
                    "Exited column not found in correlation matrix."
                )

    else:

        st.info(
            "Place corr_matrix.joblib in the project directory "
            "to display the heatmap."
        )


# ==========================================
# 11. FOOTER
# ==========================================

st.sidebar.divider()

st.sidebar.caption(
    "Bank Churn Prediction | Four ML Models"
)
