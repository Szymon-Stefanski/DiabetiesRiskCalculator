import pytest
import pickle as pkl
import numpy as np
from model.model import make_prediction


@pytest.fixture
def load_model():
    """
    Fixture to load the trained RandomForest model from the pickle file.
    Ensures the model is available for all test functions in this module.
    """
    with open("../model/model.pkl", "rb") as f:
        model = pkl.load(f)

    return model


def test_sanity_check(load_model):
    """
    Basic sanity check to verify that the model accepts the correct input shape
    and returns a prediction of expected length.
    """
    example = np.zeros((1, 21))
    prediction = make_prediction(load_model, example)
    assert len(prediction) == 1


def test_healthy_patient_invariance(load_model):
    """
    Invariance test for a healthy patient profile.
    Verifies that a low-risk input (low BMI, young, active) results in a negative prediction (0).
    """
    healthy_person = np.array([[
        0, 0, 1, 22, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 6, 8
    ]])
    prediction = make_prediction(load_model, healthy_person)
    assert prediction[0] == 0, "False positive: Model assigned risk to a healthy person"


def test_high_risk_patient(load_model):
    """
    Performance test for a high-risk patient profile.
    Verifies that a high-risk input (high BMI, elderly, comorbidities) results in a positive prediction (1).
    """
    sick_person = np.array([[
        1, 1, 1, 40, 1, 1, 1, 0, 0, 0, 1, 1, 0, 5, 20, 20, 1, 0, 13, 2, 1
    ]])
    prediction = make_prediction(load_model, sick_person)
    assert prediction[0] == 1, "False negative: Model missed a high-risk person"


def test_invalid_input_shape(load_model):
    """
    Boundary test to verify that the prediction function raises a ValueError
    when the input features dimensions do not match the expected model input (21 features).
    """
    bad_input = np.zeros((1, 10))
    with pytest.raises(ValueError):
        make_prediction(load_model, bad_input)


def test_prediction_probability_range(load_model):
    """
    Verifies that the model's 'predict_proba' output remains within
    the mathematically valid range of [0, 1].
    """
    example = np.zeros((1, 21))
    proba = load_model.predict_proba(example)[0][1]
    assert 0 <= proba <= 1
