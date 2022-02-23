MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = '[UK] [London] [rmelbardis] '

BUCKET_NAME = 'wagon-data-805-melbardis'
BUCKET_TRAIN_DATA_PATH = 'data/train.csv'
BUCKET_TEST_DATA_PATH = 'data/test.csv'
STORAGE_LOCATION = 'models/taxifare/model.joblib'

PATH_TO_LOCAL_MODEL = 'model.joblib'
PATH_TO_GCP_MODEL = f"gs://{BUCKET_NAME}/{STORAGE_LOCATION}"
