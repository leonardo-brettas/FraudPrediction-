from django.db import models
from joblib import load, dump
from pandas import DataFrame, read_parquet, concat
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from time import time
from os.path import join
from pickle import loads as pickle_loads
from nubank.settings import REDIS, S3
from imblearn.under_sampling import RandomUnderSampler
from io import BytesIO

class RegressionModel(models.Model):
    created_at = models.DateTimeField(verbose_name='Date that the model was created', auto_now_add=True)
    legacy_date = models.DateTimeField(verbose_name='Date that the model was deprecated', null=True, blank=True)
    training_file = models.CharField(verbose_name='File used to train the model', max_length=100)
    regression_file = models.CharField(verbose_name='File of the model', max_length=100)
    roc_auc = models.FloatField(verbose_name='ROC AUC of the model', null=False, blank=False)

    def __str__(self):
        return self.regression_file
    
    
    def check_if_model_on_redis(self):
        return REDIS.exists(self.regression_file)
    
    def set_model_expiration(self):
        REDIS.expire(self.regression_file, 60*60*1)
    
    def read_model_binary(self):
        if self.check_if_model_on_redis():
            binary_data = REDIS.get(self.regression_file)
            binary_io = BytesIO(binary_data)
            model = load(binary_io)
            return model
        else:
            s3_file = S3.get_object(Bucket="app", Key=self.regression_file)
            model_data = s3_file['Body'].read()
            REDIS.set(self.regression_file, model_data)
            self.set_model_expiration()
            model = load(BytesIO(model_data))
            return model
    
    def predict(self, payload):
        model = self.read_model_binary()
        features = DataFrame([payload.dict()])
        features = features[["score_3", "score_4", "score_5", "score_6"]]
        return model.predict_proba(features)[0][0]

    @staticmethod
    def train_new_model(data: DataFrame) -> Tuple[float, str]:
        data['score_3'] = data['score_3'].fillna(455)
        data['default'] = data['default'].fillna(False)
        if RegressionModel.check_if_data_is_unbalanced(data):
            data = RegressionModel.apply_random_under_sampling(data)
        x_train, x_test, y_train, y_test = train_test_split(data[["score_3", "score_4", "score_5", "score_6"]],
                                                            data["default"], test_size=0.25, random_state=42)
        clf = LogisticRegression(C=0.1)
        clf.fit(x_train, y_train)
        validation_predictions = clf.predict_proba(x_test)[:, 1]
        score = roc_auc_score(y_test, validation_predictions)
        regression_file = str(int(time())) + ".pkl"    
        dump(clf, join("storage", "logistic_regressions", regression_file))
        S3.upload_file(join("storage", "logistic_regressions", regression_file), "app", regression_file)
        return score, regression_file
    
    @staticmethod
    def add_new_data_to_the_model(new_data: DataFrame, version:int = None) -> Tuple[float, str]:
        if version:
            latest_model = RegressionModel.objects.get(id=version) 
        else:
            latest_model = RegressionModel.objects.latest('created_at')
        data = read_parquet(join("storage", "training_sets", latest_model.training_file))
        data = concat([data, new_data])
        new_training_file = "training_set" + str(int(time())) + ".parquet" 
        data.to_parquet(join("storage", "training_sets", new_training_file))
        score, regression_file = RegressionModel.train_new_model(data)    
        model = RegressionModel.objects.create(training_file=new_training_file, 
                                               regression_file=regression_file, 
                                               roc_auc=score)
        model.save()   
        return model
    
    @staticmethod
    def check_if_data_is_unbalanced(data: DataFrame) -> bool:
        return data['default'].value_counts()[0] != data['default'].value_counts()[1]
    
    @staticmethod
    def apply_random_under_sampling(data: DataFrame) -> DataFrame:
        rus = RandomUnderSampler(random_state=42)
        x_res, y_res = rus.fit_resample(data[["score_3", "score_4", "score_5", "score_6"]], data["default"])
        data = concat([x_res, y_res], axis=1)
        return data