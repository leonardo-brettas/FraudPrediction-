from typing import List
from ninja import NinjaAPI, UploadedFile
from predict.schemas import (
    ModelResponse, 
    ModelRequest, 
    RegressionModelSchema, 
    RetrainModelRequest, 
    RetrainModelResponse
)
from predict.models import RegressionModel
from ninja.security import HttpBearer
from nubank.settings import API_KEY
from pandas import read_parquet, concat
from os.path import join
from time import time


class BearerToken(HttpBearer):
    def authenticate(self, request, token):
        if token == API_KEY:
            return token


class DataException(Exception):
    pass


class DeprecatedModelException(Exception):
    pass


api = NinjaAPI(auth=BearerToken())

@api.exception_handler(DataException)
def data_exception_handler(request, exc):    
    return api.create_response(request, {"message": "Can't handle sended file"}, status=400)

@api.exception_handler(DeprecatedModelException)
def deprecated_model_exception_handler(request, exc):
    return api.create_response(request, {"message": "This model is deprecated"}, status=400)

@api.post("/predict", response=ModelResponse)
def predict(request, payload: ModelRequest):
    if payload.version:
        model = RegressionModel.objects.get(id=payload.version) 
    else:
        model = RegressionModel.objects.latest('created_at')    
    if model.legacy_date:
        raise DeprecatedModelException
    return ModelResponse(id=payload.id, prediction=model.predict(payload))

@api.post("/retrain", response=RetrainModelResponse)
def retrain(request, training_file: UploadedFile, payload: RetrainModelRequest = None):
    try:
        data = read_parquet(training_file.file)
    except Exception:
        raise DataException
    if payload and payload.version:
        model = RegressionModel.add_new_data_to_the_model(data, payload.version)
    else:
        model = RegressionModel.add_new_data_to_the_model(data)
    return model

@api.get("/models", response=List[RegressionModelSchema])
def get_models(request):
    models = RegressionModel.objects.all()
    return models
