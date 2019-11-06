
from .SklearnServer import SklearnServer
import numpy as np

from unittest import mock

# Libraries to patch:
import joblib

EXPECTED_RESPONSE = np.array([0, 1])

class FakeModel:
    def predict(self, df):
        return EXPECTED_RESPONSE


@mock.patch("joblib.load", return_value=FakeModel())
def test_sklearn_server(*args, **kwargs):
    data = ["text 1", "text 2"]

    s = SklearnServer()
    result = s.predict(data)
    assert all(result == EXPECTED_RESPONSE)

