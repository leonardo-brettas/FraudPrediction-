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
from pandas import read_parquet
from os.path import join
from time import time



class BearerToken(HttpBearer):
    def authenticate(self, request, token):
        if token == API_KEY:
            return token

router = NinjaAPI(auth=BearerToken())

@router.post("/predict", response=ModelResponse)
def predict(request, payload: ModelRequest):
    model = RegressionModel.objects.get(id=payload.version) if payload.version else RegressionModel.objects.latest('created_at')  
    return ModelResponse(id=payload.id, prediction=model.predict(payload))

@router.post("/retrain", response=RetrainModelResponse)
def retrain(request, payload: RetrainModelRequest, training_file: UploadedFile):
    latest_model = RegressionModel.objects.get(id=payload.version) if payload.version else RegressionModel.objects.latest('created_at')
    data = read_parquet(join("storage", "training_sets", latest_model.training_file))
    new_data = read_parquet(training_file.file)
    data = data.append(new_data)
    new_training_file = "training_set" + str(int(time())) + ".parquet" 
    data.to_parquet(join("storage", "training_sets", new_training_file))
    score, regression_file = RegressionModel.train_new_model(data)    
    model = RegressionModel.objects.create(training_file=new_training_file, regression_file=regression_file, roc_auc=score)
    model.save()   
    return model

@router.get("/models", response=List[RegressionModelSchema])
def get_models(request):
    models = RegressionModel.objects.all()
    return models