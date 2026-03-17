import pytest
import pickle as pkl
import numpy as np
from model.model import make_prediction


@pytest.fixture
def load_model():
    with open("../model/model.pkl", "rb") as f:
        model = pkl.load(f)

    return model


def test_sanity_check(load_model):
    example = np.zeros((1, 21))
    prediction = make_prediction(load_model, example)
    assert len(prediction) == 1


def test_healthy_patient_invariance(load_model):
    healthy_person = np.array([[
        0, 0, 1, 22, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 6, 8
    ]])
    prediction = make_prediction(load_model, healthy_person)
    assert prediction[0] == 0, "False positive: Model assigned risk to a healthy person"


def test_high_risk_patient(load_model):
    sick_person = np.array([[
        1, 1, 1, 40, 1, 1, 1, 0, 0, 0, 1, 1, 0, 5, 20, 20, 1, 0, 13, 2, 1
    ]])
    prediction = make_prediction(load_model, sick_person)
    assert prediction[0] == 1, "False negative: Model missed a high-risk person"


def test_invalid_input_shape(load_model):
    bad_input = np.zeros((1, 10))
    with pytest.raises(ValueError):
        make_prediction(load_model, bad_input)


def test_prediction_probability_range(load_model):
    example = np.zeros((1, 21))
    proba = load_model.predict_proba(example)[0][1]
    assert 0 <= proba <= 1
