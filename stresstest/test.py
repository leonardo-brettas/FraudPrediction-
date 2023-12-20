from locust import HttpUser, task, between

class PredictUser(HttpUser):
    host = "http://localhost"
    wait_time = between(0.1, 0.2)

    @task
    def predict(self):
        headers = {"Authorization": f"Bearer f272d86b7a52173f044afae1401a0af1f9f4503032682f8ac0efc17439d6995f"}
        self.client.post("/predict",headers=headers, json={
            "id": "string",
            "score_3": 0,
            "score_4": 0,
            "score_5": 0,
            "score_6": 0,
            "income": 0,
            "version": 0
        })