# Bank Customer Churn Prediction

A Machine Learning project that predicts whether a bank customer is likely to leave the bank (churn) based on customer demographics, account information, and banking activity.

The project uses four machine learning algorithms and provides an interactive **Streamlit dashboard** where users can enter customer details, generate churn predictions, and compare model outputs.

---

## 1. Project Overview

Customer churn is a major challenge for banks and financial institutions. Losing existing customers can affect revenue, increase customer acquisition costs, and reduce long-term business growth.

This project uses historical customer data to build machine learning models that identify customers who are likely to discontinue their banking relationship.

The application allows users to:
- Enter customer information through an interactive dashboard.
- Predict whether a customer is likely to churn or remain with the bank.
- Compare predictions from four different machine learning models.
- View churn risk probabilities where supported by the model.
- Explore correlations between customer attributes and churn.

### Business Objective

The primary objective is to help banks identify customers who may leave so that they can take proactive retention measures, such as:
- Offering personalized banking services.
- Providing loyalty benefits.
- Improving customer support.
- Identifying inactive or dissatisfied customers.
- Designing targeted customer retention campaigns.

---

## 2. Machine Learning Models

This project uses four classification algorithms:

| Model | Description |
|---|---|
| Naive Bayes | A probabilistic classification algorithm based on Bayes' theorem and feature independence assumptions. |
| XGBoost | A gradient-boosting algorithm that builds an ensemble of decision trees to improve predictive performance. |
| Gradient Boosting | An ensemble technique that builds trees sequentially, with each new tree correcting errors made by previous trees. |
| Support Vector Machine (SVM) | A supervised learning algorithm that finds an optimal decision boundary to separate customer classes. |

The models are trained to classify customers into two categories:

- **0 – Retained:** The customer is expected to stay with the bank.
- **1 – Churned:** The customer is predicted to leave the bank.

The dashboard compares the predictions made by these models for the same customer input.

---

## 3. Dataset Description

The project uses a bank customer churn dataset containing approximately 10,000 customer records.

### Input Features

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score. |
| Geography | Customer's geographical location. |
| Gender | Customer's gender. |
| Age | Customer's age. |
| Tenure | Number of years the customer has been associated with the bank. |
| Balance | Amount of money maintained in the customer's bank account. |
| NumOfProducts | Number of banking products used by the customer. |
| HasCrCard | Indicates whether the customer owns a credit card (1 = Yes, 0 = No). |
| IsActiveMember | Indicates whether the customer is an active bank member (1 = Yes, 0 = No). |
| EstimatedSalary | Estimated annual salary of the customer. |

### Target Variable

| Target | Meaning |
|---|---|
| Exited = 0 | Customer retained |
| Exited = 1 | Customer churned |

The following columns are excluded from model training because they are identifiers or personal names rather than useful predictive features:

- RowNumber
- CustomerId
- Surname

---

## 4. Data Preprocessing

Data preprocessing prepares raw customer information for machine learning.

The preprocessing workflow includes:

1. **Data inspection:** Understanding dataset dimensions, column types, missing values, and statistical distributions.
2. **Feature selection:** Removing irrelevant identifier columns.
3. **Separating features and target:** Using customer attributes as input (`X`) and `Exited` as the target (`y`).
4. **Categorical encoding:** Converting categorical variables such as Geography and Gender into numerical representations using One-Hot Encoding.
5. **Feature scaling:** Applying StandardScaler to numerical features.
6. **Train-test split:** Dividing the dataset into training and testing sets to evaluate model performance on unseen data.

### Preprocessing Pipeline

A `ColumnTransformer` is used to apply different transformations to numerical and categorical columns.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])
```

This ensures that numerical and categorical features are processed appropriately.

**Important:** The fitted preprocessing steps should be saved together with each model, preferably as a single Scikit-learn Pipeline. This ensures that new customer inputs receive the same transformations used during training.

---

## 5. Model Training and Evaluation

The four algorithms are trained using the prepared training dataset.

The models can be evaluated using the following classification metrics:

- **Accuracy:** Percentage of correctly classified customers.
- **Precision:** Percentage of customers predicted to churn who actually churned.
- **Recall:** Percentage of actual churned customers correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows true positives, true negatives, false positives, and false negatives.

For a bank churn prediction problem, recall and F1-score for the churn class are particularly important. Missing a customer who is likely to leave may be more costly than incorrectly flagging a customer who would have stayed.

Model performance should be calculated using a held-out test set rather than the individual customer predictions generated by the dashboard.

---

## 6. Streamlit Dashboard

The project includes an interactive dashboard developed using Streamlit.

### Dashboard Features

**1. Customer Information Input**

Users can enter:
- Credit score
- Geography
- Gender
- Age
- Tenure
- Account balance
- Number of banking products
- Credit card ownership
- Membership activity
- Estimated salary

**2. Individual Model Prediction**

The selected primary model predicts whether the customer is likely to churn or remain with the bank.

**3. Model Comparison**

The application compares predictions from:
- Naive Bayes
- XGBoost
- Gradient Boosting
- SVM

**4. Churn Probability**

When the selected model supports `predict_proba()`, the dashboard displays the estimated probability of churn.

Some SVM configurations do not support probability prediction. In that case, the application displays the classification result without presenting a probability.

**5. Model Voting**

The dashboard can summarize the predictions across successfully executed models to show how many models predict churn versus retention.

This is a simple prediction comparison, not a separately trained ensemble model.

**6. Correlation Analysis**

A correlation heatmap helps visualize relationships between numerical customer attributes and the churn target.

---

## 7. Project Structure

The recommended project directory structure is:

```text
bank-customer-churn-prediction/
│
├── app.py
│
├── model1.pkl
├── model2.pkl
├── model3.pkl
├── model4.pkl
│
├── corr_matrix.joblib
│
├── requirements.txt
│
├── README.md
│
└── dataset/
    └── Bank_Churn.csv
```

### File Description

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application. |
| `model1.pkl` | Trained Naive Bayes model. |
| `model2.pkl` | Trained XGBoost model. |
| `model3.pkl` | Trained Gradient Boosting model. |
| `model4.pkl` | Trained SVM model. |
| `corr_matrix.joblib` | Saved correlation matrix used for visualization. |
| `requirements.txt` | Python dependencies required to run the application. |
| `README.md` | Project documentation. |
| `dataset/` | Dataset directory, if the original dataset is included. |

---

## 8. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language. |
| Pandas | Data manipulation and analysis. |
| NumPy | Numerical operations. |
| Scikit-learn | Preprocessing, model training, and evaluation. |
| XGBoost | Gradient-boosting classification. |
| Joblib | Saving and loading trained ML models. |
| Streamlit | Interactive web application. |
| Matplotlib | Data visualization. |
| Seaborn | Statistical visualizations and correlation heatmaps. |

---

## 9. Installation and Setup

### Prerequisites

- Python 3.9 or later (use a version compatible with the libraries used to train the models).
- pip package manager.
- The trained model files.
- Git (optional, for cloning the repository).

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd bank-customer-churn-prediction
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file yet, install the packages using:

```bash
pip install streamlit pandas numpy joblib matplotlib seaborn scikit-learn xgboost
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

---

## 10. Model Serialization

Trained models are saved using Joblib so they can be reused without retraining every time the Streamlit application starts.

Example:

```python
import joblib

joblib.dump(trained_models["Naive Bayes"], "model1.pkl")
joblib.dump(trained_models["XGBoost"], "model2.pkl")
joblib.dump(trained_models["Gradient Boosting"], "model3.pkl")
joblib.dump(trained_models["SVM"], "model4.pkl")
```

The Streamlit application loads these serialized models and uses them to make predictions.

**Model compatibility:** The Python, Scikit-learn, and XGBoost versions used when loading the files should be compatible with the versions used during training. Always verify that all four files load successfully before deploying the application.

---

## 11. Future Enhancements

This project can be extended into a more complete MLOps workflow.

| Enhancement | Description |
|---|---|
| MLflow | Track experiments, parameters, metrics, and model versions. |
| DVC | Version datasets and model artifacts. |
| FastAPI | Expose churn predictions through REST APIs. |
| Docker | Containerize the application and its dependencies. |
| GitHub Actions | Automate testing and deployment through CI/CD. |
| Cloud Deployment | Deploy the application on AWS, Azure, or GCP. |
| Model Monitoring | Monitor prediction quality and changes in incoming data. |
| Data Drift Detection | Detect changes in customer data distributions. |
| Model Retraining | Retrain models when performance deteriorates or new data becomes available. |
| Model Registry | Maintain approved model versions for deployment. |

### Suggested MLOps Architecture

```text
Customer Dataset
       |
       v
Data Preprocessing
       |
       v
Feature Engineering
       |
       v
Model Training
       |
       v
Model Evaluation
       |
       v
Experiment Tracking (MLflow)
       |
       v
Model Serialization
       |
       v
Model Deployment
       |
       v
Streamlit Dashboard / FastAPI
       |
       v
Prediction Monitoring
       |
       v
Model Retraining
```

---

## 12. Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- Historical patterns may not fully reflect future customer behavior.
- A predicted churn result is an estimate, not a guarantee that a customer will leave.
- Probability estimates are available only for models that support probability prediction.
- The current dashboard compares model predictions but does not automatically retrain models.
- Correlation does not establish causation.
- Production deployment would require additional security, monitoring, validation, and model-versioning controls.

---

## 13. Conclusion

The Bank Customer Churn Prediction project demonstrates how machine learning can be applied to a real-world banking problem.

By training and comparing four classification algorithms and providing an interactive Streamlit interface, the project makes churn predictions accessible and understandable.

It also provides a foundation for learning the end-to-end machine learning lifecycle, including data preprocessing, model training, evaluation, serialization, deployment, and future MLOps integration.

---

## 14. Author

**Project:** Bank Customer Churn Prediction  
**Domain:** Machine Learning | Classification | MLOps Learning  
**Language:** Python  
**Application:** Streamlit

