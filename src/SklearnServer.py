
import joblib, logging
from seldon_core.storage import Storage

class SklearnServer:
    def __init__(self, model_uri):
        output_dir = Storage.download(model_uri)
        self._model = joblib.load(f"{output_dir}/model.joblib")

    def predict(self, data, feature_names=[], metadata={}):
        logging.info(data)

        prediction = self._model.predict(data)

        logging.info(prediction)

        return prediction
