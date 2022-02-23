import os

from TaxiFareModel import params
from google.cloud import storage
from termcolor import colored


def storage_upload(model_directory, bucket=params.BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)

    storage_location = '{}/{}/{}/{}'.format(
        'models',
        'taxi_fare_model',
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.joblib')
    print(colored("=> model.joblib uploaded to bucket {} inside {}".format(params.BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove('model.joblib')
