from django.db import models
from joblib import load
from pandas import DataFrame
from os.path import join
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from joblib import dump
from time import time
from os.path import join


class RegressionModel(models.Model):
    created_at = models.DateTimeField(verbose_name='Date that the model was created', auto_now_add=True)
    legacy_date = models.DateTimeField(verbose_name='Date that the model was deprecated', null=True, blank=True)
    training_file = models.CharField(verbose_name='File used to train the model', max_length=100)
    regression_file = models.CharField(verbose_name='File of the model', max_length=100)
    roc_auc = models.FloatField(verbose_name='ROC AUC of the model', null=False, blank=False)

    def __str__(self):
        return self.regression_file
    
    def predict(self, payload):
        model = load(join("storage", "logistic_regressions", self.regression_file))
        features = DataFrame([payload.dict()])
        features = features[["score_3", "score_4", "score_5", "score_6"]]
        return model.predict_proba(features)[0][0]

    @staticmethod
    def train_new_model(data: DataFrame) -> Tuple[float, str]:
        data['score_3'] = data['score_3'].fillna(455)
        data['default'] = data['default'].fillna(False)
        x_train, x_test, y_train, y_test = train_test_split(data[["score_3", "score_4", "score_5", "score_6"]],
                                                            data["default"], test_size=0.25, random_state=42)
        clf = LogisticRegression(C=0.1)
        clf.fit(x_train, y_train)
        validation_predictions = clf.predict_proba(x_test)[:, 1]
        score = roc_auc_score(y_test, validation_predictions)
        regression_file = str(int(time())) + ".pkl"    
        dump(clf, join("storage", "logistic_regressions", regression_file))
        return score, regression_file