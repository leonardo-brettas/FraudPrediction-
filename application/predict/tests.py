from django.test import TestCase
from .models import RegressionModel
from pandas import DataFrame

class RegressionModelTestCase(TestCase):
    def test_apply_random_under_sampling(self):
        data = DataFrame({"default": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                          "score_3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          "score_4": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          "score_5": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          "score_6": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        data = RegressionModel.apply_random_under_sampling(data)
        self.assertEqual(data["default"].value_counts().tolist(), [2, 2])
        
    def test_check_if_data_is_unbalanced(self):
        data = DataFrame({"default": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                          "score_3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          "score_4": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          "score_5": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          "score_6": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        self.assertTrue(RegressionModel.check_if_data_is_unbalanced(data))