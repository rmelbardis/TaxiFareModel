# imports

import pandas as pd

from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from google.cloud import storage

from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel import params

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

### import model and define
from xgboost import XGBRegressor
model_type = XGBRegressor(max_depth=10, n_estimators=100, learning_rate=0.1)
model_name_version = 'XGBoost_v1'
estimator_name = 'XGBRegressor, depth=10, n=100, lrate=0.1'
###

class Trainer():

    def __init__(self, X, y, model, model_name_version='test', log_mode=False):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.model = model
        self.model_name_version = model_name_version
        self.log_mode = log_mode
        self.experiment_name = f'{params.EXPERIMENT_NAME}{self.model_name_version}'

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        full_pipeline = Pipeline([
            ('preproc', preproc_pipe),
            (f'{self.model_name_version}', self.model)
        ])
        return full_pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        self.y_pred = self.pipeline.predict(X_test)
        self.rmse = compute_rmse(self.y_pred, y_test)
        if self.log_mode:
            self.mlflow_log_param('estimator', estimator_name)
            self.mlflow_log_metric('rmse', self.rmse)
        print(self.rmse)
        return self.rmse


    """
    Submit metrics to MLFlow
    """
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(params.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(self.pipeline, 'model.joblib')
        print("model.joblib saved locally")

        client = storage.Client()
        bucket = client.bucket(params.BUCKET_NAME)
        blob = bucket.blob(params.STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')
        print(f"uploaded model.joblib to gcp cloud storage under \n => {params.STORAGE_LOCATION}")

if __name__ == "__main__":
    df = get_data(1000)
    df = clean_data(df)
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = Trainer(X_train, y_train, model_type, model_name_version=model_name_version, log_mode=True)
    model.run()
    model.evaluate(X_test, y_test)
    model.save_model()
