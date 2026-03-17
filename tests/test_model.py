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
