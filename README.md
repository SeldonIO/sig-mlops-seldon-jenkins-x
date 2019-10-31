
# A practical guide to MLOps with Seldon Core and Jenkins X

This tutorial provides an end-to-end tutorial that shows you how to build you MLOps pipeline with Seldon Core and Jenkins X:

* Seldon Core is a machine learning deployment & orchestration engine in Kubernetes
* Jenkins X provides automated CI+CD for Kubernetes with Preview Environments on Pull Requests



## Intuitive explanation

Before we proceed, we want to understand what we will be trying to achieve. 

And what better way of doing this than by diving into an architectural diagram.

[TODO ARCHITECTURE]

## Requirements

* A Kubernetes cluster running v1.13+ (this was run using GKE)
* The [jx CLI](https://github.com/jenkins-x/jx/) version 2.0.916
* Jenkins-X installed in your cluster (you can set it up with the [jx boot tutorial](https://jenkins-x.io/docs/getting-started/setup/boot/))
* Seldon Core [v0.5.0 installed]() in your cluster

Once you set everything up, we'll be ready to kick off 🚀

# Setting up repo

Now we want to start setting up our repo. For this we will create the following structure:

* `jenkins-x.yml` - File specifying the CI / CD steps 
* `Makefile` - Commands to build and test model
* `README.(md|ipynb)` - This file!
* `gitops/` - Folder containing the state of our production cluster
* `src`
    * `ModelName.py` - Model server wrapper file
    * `test_ModelName.py` - Unit test for model server
    * `requirements-dev.txt` - Requirements for testing
    * `requirements.txt` - Requiremnets for prod


## Let's train a model locally

Let's have a look at the model we're using for text classification.


```python
%%writefile requirements-dev.txt
scikit-learn==0.20.1
pytest==5.1.1
joblib==0.13.2
```

    Overwriting requirements-dev.txt



```python
!pip install requiremnets-dev.txt
```


```python
from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=42)

twenty_test = fetch_20newsgroups(
    subset='test', categories=categories, shuffle=True, random_state=42)

# Printing the top 3 newstories
print("\n".join(twenty_train.data[0].split("\n")[:3]))
```

    From: sd345@city.ac.uk (Michael Collier)
    Subject: Converting images to HP LaserJet III?
    Nntp-Posting-Host: hampton



```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(twenty_train.data, twenty_train.target)
```




    Pipeline(memory=None,
             steps=[('vect',
                     CountVectorizer(analyzer='word', binary=False,
                                     decode_error='strict',
                                     dtype=<class 'numpy.int64'>, encoding='utf-8',
                                     input='content', lowercase=True, max_df=1.0,
                                     max_features=None, min_df=1,
                                     ngram_range=(1, 1), preprocessor=None,
                                     stop_words=None, strip_accents=None,
                                     token_pattern='(?u)\\b\\w\\w+\\b',
                                     tokenizer=None, vocabulary=None)),
                    ('tfidf',
                     TfidfTransformer(norm='l2', smooth_idf=True,
                                      sublinear_tf=False, use_idf=True)),
                    ('clf',
                     MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))],
             verbose=False)




```python
# Let's try one
idx = 0
print(f"CONTENT:{twenty_test.data[idx][35:230]}\n\n-----------\n")
print(f"PREDICTED CLASS: {categories[twenty_test.target[idx]]}")
```

    CONTENT:
    Subject: Re: HELP for Kidney Stones ..............
    Organization: The Avant-Garde of the Now, Ltd.
    Lines: 12
    NNTP-Posting-Host: ucsd.edu
    
    As I recall from my bout with kidney stones, there isn't 
    
    -----------
    
    PREDICTED CLASS: comp.graphics



```python
import numpy as np

predicted = text_clf.predict(twenty_test.data)
print(f"Accuracy: {np.mean(predicted == twenty_test.target):.2f}")
```

    Accuracy: 0.83


## Deploy the model

Now we want to be able to deploy the model we just trained


```python
import joblib
joblib.dump(text_clf, "model.joblib")
```




    ['model.joblib']




```python
!gsutil mb gs://news_classifier/
```

    Creating gs://news_classifier/...



```python
!gsutil cp model.joblib gs://news_classifier/model/model.joblib
```

    Copying file://model.joblib [Content-Type=application/octet-stream]...
    / [1 files][  4.4 MiB/  4.4 MiB]                                                
    Operation completed over 1 objects/4.4 MiB.                                      



```python
!gsutil acl ch -r -u AllUsers:R gs://news_classifier
```

    Updated ACL on gs://news_classifier/model
    Updated ACL on gs://news_classifier/model.joblib
    Updated ACL on gs://news_classifier/model/model.joblib



```python
!mkdir -p src/
```


```python
%%writefile src/SklearnServer.py

import joblib, logging
from seldon_core.storage import Storage

class SklearnServer:
    def __init__(self, model_uri):
        output_dir = Storage.download(model_uri)
        self._model = joblib.load(f"{output_dir}/model.joblib")

    def predict(self, data, feature_names=[], metadata={}):
        logging.info(data)

        prediction = self._model.predict(data)

        logging.info(prediction)

        return prediction
```

    Overwriting src/SklearnServer.py



```python
%%writefile src/requirements.txt
scikit-learn==0.20.1
joblib==0.13.2
```

    Overwriting src/requirements.txt



```python
%%writefile src/seldon_model.conf
MODEL_NAME=SklearnServer
API_TYPE=REST
SERVICE_TYPE=MODEL
PERSISTENCE=0
```

    Overwriting src/seldon_model.conf



```bash
%%bash
SELDON_BASE_WRAPPER="seldonio/seldon-core-s2i-python36:0.12"
s2i build src/. $SELDON_BASE_WRAPPER sklearn-server:0.1 \
    --environment-file src/seldon_model.conf
```

    ---> Installing application source...
    ---> Installing dependencies ...
    Looking in links: /whl
    Collecting scikit-learn==0.20.1 (from -r requirements.txt (line 1))
      WARNING: Url '/whl' is ignored. It is either a non-existing path or lacks a specific scheme.
    Downloading https://files.pythonhosted.org/packages/10/26/d04320c3edf2d59b1fcd0720b46753d4d603a76e68d8ad10a9b92ab06db2/scikit_learn-0.20.1-cp36-cp36m-manylinux1_x86_64.whl (5.4MB)
    Collecting joblib==0.13.2 (from -r requirements.txt (line 2))
      WARNING: Url '/whl' is ignored. It is either a non-existing path or lacks a specific scheme.
    Downloading https://files.pythonhosted.org/packages/cd/c1/50a758e8247561e58cb87305b1e90b171b8c767b15b12a1734001f41d356/joblib-0.13.2-py2.py3-none-any.whl (278kB)
    Requirement already satisfied: numpy>=1.8.2 in /usr/local/lib/python3.6/site-packages (from scikit-learn==0.20.1->-r requirements.txt (line 1)) (1.17.2)
    Collecting scipy>=0.13.3 (from scikit-learn==0.20.1->-r requirements.txt (line 1))
      WARNING: Url '/whl' is ignored. It is either a non-existing path or lacks a specific scheme.
    Downloading https://files.pythonhosted.org/packages/29/50/a552a5aff252ae915f522e44642bb49a7b7b31677f9580cfd11bcc869976/scipy-1.3.1-cp36-cp36m-manylinux1_x86_64.whl (25.2MB)
    Installing collected packages: scipy, scikit-learn, joblib
    Successfully installed joblib-0.13.2 scikit-learn-0.20.1 scipy-1.3.1
    WARNING: Url '/whl' is ignored. It is either a non-existing path or lacks a specific scheme.
    WARNING: You are using pip version 19.1, however version 19.3.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.
    Build completed successfully



```bash
%%bash
YOUR_DOCKER_USERNAME="axsauze"

docker tag sklearn-server:0.1 $YOUR_DOCKER_USERNAME/sklearn-server:0.1
docker push $YOUR_DOCKER_USERNAME/sklearn-server:0.1
```

    Process is interrupted.



```python
%%writefile test_deployment.yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: news-classifier-server
  namespace: default
  creationTimestamp: 
spec:
  name: news-classifier-server
  predictors:
  - name: default
    graph:
      name: news-classifier-server-processor
      endpoint:
        type: REST
      type: MODEL
      children: []
      parameters:
      - name: model_uri
        type: STRING
        value: "gs://news_classifier/model/"
    componentSpecs:
    - metadata:
        creationTimestamp: '2019-10-12T16:00:00Z'
      spec:
        containers:
        - image: axsauze/sklearn-server:0.1
          name: news-classifier-server-processor
          env:
          - name: SELDON_LOG_LEVEL
            value: DEBUG
        terminationGracePeriodSeconds: 1
    replicas: 1
    engineResources: {}
    svcOrchSpec: {}
    traffic: 100
    explainer:
      containerSpec:
        name: ''
        resources: {}
  annotations:
    seldon.io/engine-seldon-log-messages-externally: 'true'
status: {}
```

    Overwriting test_deployment.yaml



```python
!kubectl apply -f test_deployment.yaml
```

    seldondeployment.machinelearning.seldon.io/news-classifier-server created



```python
??sc.predict
```


```python
from seldon_core.seldon_client import SeldonClient
import numpy as np

url = !kubectl get svc ambassador -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

sc = SeldonClient(
    gateway="ambassador", 
    gateway_endpoint="localhost:80",
    deployment_name="news-classifier-server",
    payload_type="ndarray",
    namespace="default",
    transport="rest")

response = sc.predict(data=np.array([twenty_test.data[0]]))

response.response.data
```




    ndarray {
      values {
        number_value: 2.0
      }
    }




```bash
%%bash
curl -X POST -H 'Content-Type: application/json' \
     -d "{'data': {'names': ['text'], 'ndarray': ['Hello world this is a test']}}" \
    http://localhost/seldon/default/news-classifier-server/api/v0.1/predictions
```

    {
      "meta": {
        "puid": "so6n21pkf70fm66eka28lc63cr",
        "tags": {
        },
        "routing": {
        },
        "requestPath": {
          "news-classifier-server-processor": "axsauze/sklearn-server:0.1"
        },
        "metrics": []
      },
      "data": {
        "names": [],
        "ndarray": [2.0]
      }
    }

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   350  100   278  100    72   7942   2057 --:--:-- --:--:-- --:--:-- 10294



```python

```
