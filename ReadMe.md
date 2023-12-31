
# I found this technichal challenge from NuBank on https://github.com/leniel/NubankDataScientist and re-implemented it from scratch. 

## Overview

* The original files are on `storage/originalfiles`
* The project was developed using Django-Ninja, so the docs are on swagger `/docs`
![1 Minunte Stress Test](image.png)
## Next Steps

* Add Dockerfile for production enviroments
* Increase unitary test coverage

# Challenge Description

One of the most important decisions we make here at Nubank is who we give credit cards to. We want to make sure that we're only
giving credit cards to people who are likely to pay us back.

We have some data from people who we have given a credit card to in the past, and one of our data scientists has created a model
that tries to predict whether someone will pay their credit card bills based on this data.  He claims that the model has really good performance for
this problem, with an AUC score of 0.59 (AUC stands for Area Under the receiver operating characteristic Curve, a common performance metric for classification models) .

We want to start using this model to make approve or decline decisions in real time, but the data scientist has no idea
how to move his research into production.

The data scientist gives you three files:
 - `model.py`: the script which he used to train his model.
 - `training_set.parquet`: the data which he used to train his model.
 - `pip.txt`: the versions of libraries he used in his model

Your task is to create a simple HTTP service that allows us to use this model in production. It should have a POST endpoint
`/predict` which accepts and returns JSON content type payloads. Low latency is an important requirement, as other services will hit this endpoint
whenever data is available for a new possible customer, and will use the predictions that come from your service to make the decision to
send or not a credit card to each customer.

Example input:
```
json
{
    "id": "8db4206f-8878-174d-7a23-dd2c4f4ef5a0",
    "score_3": 480.0,
    "score_4": 105.2,
    "score_5": 0.8514,
    "score_6": 94.2,
    "income": 50000
}
```

Example output:
```
json
{
    "id": "8db4206f-8878-174d-7a23-dd2c4f4ef5a0",
    "prediction": 0.1495
}
```

Once you're comfortable with your solution, you may want to tackle the issue of retraining: Periodically, once we have
collected some more data, we may want to retrain the model including this extra data. However, we usually need to keep the
old versions working, as they might still be useful. Update your service so that it supports running multiple versions of
the model simultaneously. You can assume that when we want to retrain, a parquet file with the new data will be provided.

We will evaluate your code in a similar way that we usually evaluate code that we send to production, so we expect production
quality code and tests. Also, pay attention to code organization and make sure it is readable and clean.

You should deliver a git repository with your code and a short README file outlining the solution and explaining how to
build and run the code, we just ask that you keep your repository private (GitLab and BitBucket offer free private
repositories).

We know this might be your first experience with Python, so don't worry if your code is not idiomatic. Feel free to ask any questions,
but please note that we won't be able to give you feedback about your code before your deliver. However, we're more than willing to help you
with understanding the domain or picking a library, for instance.

Lastly, there is no need to rush with the solution: delivering your exercise earlier than the due date is not a criteria
we take into account when evaluating the exercise: so if you finish earlier than that, please take some time to see what
you could improve. Also, if you think the time frame may not be enough for any reason, don't hesitate to ask for more
time.