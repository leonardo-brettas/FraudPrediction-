from ninja import Schema, ModelSchema
from predict.models import RegressionModel
from typing import Optional
from ninja import Schema, ModelSchema
from predict.models import RegressionModel


class ModelResponse(Schema):
    id: str
    prediction: float


class ModelRequest(Schema):
    id: str
    score_3: float
    score_4: float
    score_5: float
    score_6: float
    income: Optional[int] = None
    version: Optional[int] = None

    
class RetrainModelRequest(Schema):
    version: Optional[int] = None

    
class RetrainModelResponse(ModelSchema):
    class Config:
        model = RegressionModel
        model_fields = ['id', 'roc_auc']
    
    
class RegressionModelSchema(ModelSchema):
    class Config:
        model = RegressionModel
        model_fields = '__all__'
        