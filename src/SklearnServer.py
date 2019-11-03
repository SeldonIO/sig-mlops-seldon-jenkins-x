
import joblib, logging

class SklearnServer:
    def __init__(self):
        self._model = joblib.load(f"model.joblib")

    def predict(self, data, feature_names=[], metadata={}):
        logging.info(data)

        prediction = self._model.predict(data)

        logging.info(prediction)

        return prediction
