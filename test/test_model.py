import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Path to the saved model
MODEL_PATH = Path("model1.pkl")


@pytest.fixture
def model():
    """Load the trained model before running tests."""
    loaded_model = joblib.load(MODEL_PATH)
    return loaded_model


def test_model_loading(model):
    """Check whether the model loads successfully."""
    assert model is not None


def test_model_has_prediction_method(model):
    """Check whether the model supports prediction."""
    assert hasattr(model, "predict")


def test_model_prediction(model):
    """Check whether the model generates predictions."""
    sample_input = pd.DataFrame(
        [[619, "France", "Female", 42, 2, 0.00, 1, 1, 1, 101348.88]],
        columns=[
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
    )

    prediction = model.predict(sample_input)

    assert prediction is not None
    assert len(prediction) == 1


def test_prediction_output(model):
    """Check whether predictions contain valid class labels."""
    sample_input = pd.DataFrame(
        [[619, "France", "Female", 42, 2, 0.00, 1, 1, 1, 101348.88]],
        
        columns=[
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
    )

    prediction = model.predict(sample_input)

    assert prediction[0] in [0, 1]


def test_prediction_probability(model):
    """Check whether predicted probabilities are valid."""
    if not hasattr(model, "predict_proba"):
        pytest.skip("Model does not support predict_proba")

    sample_input = pd.DataFrame(
        [[619, "France", "Female", 42, 2, 0.00, 1, 1, 1, 101348.88]],
        columns=[
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
    )

    probabilities = model.predict_proba(sample_input)

    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)