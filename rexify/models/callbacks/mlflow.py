import os

import mlflow
import tensorflow as tf


class MlflowCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        tracking_uri: str = os.environ.get("MLFLOW_TRACKING_URI"),
        experiment_name: str = os.environ.get("MLFLOW_EXPERIMENT_NAME"),
    ):
        super().__init__()
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

    def on_train_begin(self, logs=None):
        config = self.model.get_config()

        def parse(value):
            if type(value).__name__ == "ListWrapper":
                return list(value)
            return value

        params = {k: parse(v) for k, v in config.items()}
        mlflow.log_params(params)

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(logs)
