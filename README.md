
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
* `VERSION` - A file containing the version which is updated upon each release
* `charts/` - Folder containing the deployment configuration information
* `integration/` - Folder containing integration tests using KIND
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
!make install_dev
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
joblib.dump(text_clf, "src/model.joblib")
```




    ['src/model.joblib']




```python
%%writefile src/SklearnServer.py

import joblib, logging

class SklearnServer:
    def __init__(self):
        self._model = joblib.load(f"model.joblib")

    def predict(self, data, feature_names=[], metadata={}):
        logging.info(data)

        prediction = self._model.predict(data)

        logging.info(prediction)

        return prediction
```

    Overwriting src/SklearnServer.py



```python
%%writefile src/test_SklearnServer.py

from .SklearnServer import SklearnServer
import os

def test_sklearn_server():
    data = ["From: brian@ucsd.edu (Brian Kantor)\nSubject: Re: HELP for Kidney Stones ..............\nOrganization: The Avant-Garde of the Now, Ltd.\nLines: 12\nNNTP-Posting-Host: ucsd.edu\n\nAs I recall from my bout with kidney stones, there isn't any\nmedication that can do anything about them except relieve the pain.\n\nEither they pass, or they have to be broken up with sound, or they have\nto be extracted surgically.\n\nWhen I was in, the X-ray tech happened to mention that she'd had kidney\nstones and children, and the childbirth hurt less.\n\nDemerol worked, although I nearly got arrested on my way home when I barfed\nall over the police car parked just outside the ER.\n\t- Brian\n",
            'From: rind@enterprise.bih.harvard.edu (David Rind)\nSubject: Re: Candida(yeast) Bloom, Fact or Fiction\nOrganization: Beth Israel Hospital, Harvard Medical School, Boston Mass., USA\nLines: 37\nNNTP-Posting-Host: enterprise.bih.harvard.edu\n\nIn article <1993Apr26.103242.1@vms.ocom.okstate.edu>\n banschbach@vms.ocom.okstate.edu writes:\n>are in a different class.  The big question seems to be is it reasonable to \n>use them in patients with GI distress or sinus problems that *could* be due \n>to candida blooms following the use of broad-spectrum antibiotics?\n\nI guess I\'m still not clear on what the term "candida bloom" means,\nbut certainly it is well known that thrush (superficial candidal\ninfections on mucous membranes) can occur after antibiotic use.\nThis has nothing to do with systemic yeast syndrome, the "quack"\ndiagnosis that has been being discussed.\n\n\n>found in the sinus mucus membranes than is candida.  Women have been known \n>for a very long time to suffer from candida blooms in the vagina and a \n>women is lucky to find a physician who is willing to treat the cause and \n>not give give her advise to use the OTC anti-fungal creams.\n\nLucky how?  Since a recent article (randomized controlled trial) of\noral yogurt on reducing vaginal candidiasis, I\'ve mentioned to a \nnumber of patients with frequent vaginal yeast infections that they\ncould try eating 6 ounces of yogurt daily.  It turns out most would\nrather just use anti-fungal creams when they get yeast infections.\n\n>yogurt dangerous).  If this were a standard part of medical practice, as \n>Gordon R. says it is, then the incidence of GI distress and vaginal yeast \n>infections should decline.\n\nAgain, this just isn\'t what the systemic yeast syndrome is about, and\nhas nothing to do with the quack therapies that were being discussed.\nThere is some evidence that attempts to reinoculate the GI tract with\nbacteria after antibiotic therapy don\'t seem to be very helpful in\nreducing diarrhea, but I don\'t think anyone would view this as a\nquack therapy.\n-- \nDavid Rind\nrind@enterprise.bih.harvard.edu\n']
    labels = [2, 2]

    s = SklearnServer()
    result = s.predict(data)
    assert all(result == labels)
```

    Overwriting src/test_SklearnServer.py



```python
!make test
```

    cat: VERSION: No such file or directory
    Makefile:25: warning: overriding recipe for target 'make'
    Makefile:22: warning: ignoring old recipe for target 'make'
    pytest -s --verbose -W ignore 2>&1
    [1m============================= test session starts ==============================[0m
    platform linux -- Python 3.7.3, pytest-5.1.1, py-1.8.0, pluggy-0.12.0 -- /home/alejandro/miniconda3/envs/reddit-classification/bin/python
    cachedir: .pytest_cache
    rootdir: /home/alejandro/Programming/kubernetes/seldon/sig-mlops-example
    plugins: cov-2.7.1, forked-1.0.2, localserver-0.5.0
    collected 1 item                                                               [0m
    
    src/test_SklearnServer.py::test_sklearn_server [32mPASSED[0m
    
    [32m[1m============================== 1 passed in 1.72s ===============================[0m



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
    Collecting scipy>=0.13.3 (from scikit-learn==0.20.1->-r requirements.txt (line 1))
      WARNING: Url '/whl' is ignored. It is either a non-existing path or lacks a specific scheme.
    Downloading https://files.pythonhosted.org/packages/29/50/a552a5aff252ae915f522e44642bb49a7b7b31677f9580cfd11bcc869976/scipy-1.3.1-cp36-cp36m-manylinux1_x86_64.whl (25.2MB)
    Requirement already satisfied: numpy>=1.8.2 in /usr/local/lib/python3.6/site-packages (from scikit-learn==0.20.1->-r requirements.txt (line 1)) (1.17.2)
    Installing collected packages: scipy, scikit-learn, joblib
    Successfully installed joblib-0.13.2 scikit-learn-0.20.1 scipy-1.3.1
    WARNING: Url '/whl' is ignored. It is either a non-existing path or lacks a specific scheme.
    WARNING: You are using pip version 19.1, however version 19.3.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.
    Build completed successfully



```bash
%%bash
YOUR_DOCKER_USERNAME="seldonio"

docker tag sklearn-server:0.1 $YOUR_DOCKER_USERNAME/sklearn-server:0.1
# docker push $YOUR_DOCKER_USERNAME/sklearn-server:0.1
```


```python
!cat charts/sklearn-model-server/templates/sklearn-seldon-deployment.yaml
```

    apiVersion: machinelearning.seldon.io/v1alpha2
    kind: SeldonDeployment
    metadata:
      name: {{ .Values.model.name }}
    spec:
      name: {{ .Values.model.name }}
      predictors:
      - name: default
        graph:
          name: {{ .Values.model.name }}-processor
          endpoint:
            type: REST
          type: MODEL
          children: []
          parameters:
          - name: model_uri
            type: STRING
            value: "gs://news_classifier/model/"
        componentSpecs:
        - spec:
            containers:
            - image: "{{ .Values.image.respository }}:{{ .Values.image.tag }}"
              imagePullPolicy: {{ .Values.image.pullPolicy }}
              name: {{ .Values.model.name }}-processor
              env:
    {{- range $pkey, $pval := .Values.env }}
              - name: {{ $pkey }}
                value: {{ quote $pval }}
    {{- end }}
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
    



```python
!kubectl apply -f gitops/test_deployment.yaml
```

    seldondeployment.machinelearning.seldon.io/news-classifier-server created



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
!kubectl delete -f gitops/test_deployment.yaml
```

    seldondeployment.machinelearning.seldon.io "news-classifier-server" deleted


# Setting up CI before CD

We have now separated our model development into two chunks: 

* The first one involves the creation of a model serve, and the second one involves the CI of the model server, and the second involves the deployment of models that create the model.


## Using the Jenkins X pipeline

In order to do this we will be able to first run some tests and the push to the docker repo.

For this we will be leveraging the Jenkins X file, we'll first start with a simple file that just runs the tests:


```python
%%writefile jenkins-x.yml
buildPack: none
pipelineConfig:
  pipelines:
    release:
      pipeline:
        agent:
          image: seldonio/core-builder:0.4
        stages:
          - name: test-sklearn-server
            steps:
            - name: run-tests
              command: make
              args:
              - install_dev
              - test
    pullRequest:
      pipeline:
        agent:
          image: seldonio/core-builder:0.4
        stages:
          - name: test-sklearn-server
            steps:
            - name: run-tests
              command: make
              args:
              - install_dev
              - test
```

    Overwriting jenkins-x.yml


The `jenkins-x.yml` file is pretty easy to understand if we read through the different steps.

Basically we can define the steps of what happens upon `release` - i.e. when a PR / Commit is added to master - and what happens upon `pullRequest` - whenever someone opens a pull request.

You can see that the steps are exactly the same for both release and PR for now - namely, we run `make install_dev test` which basically installs all the dependencies and runs all the tests.

### Setting up the repo with the pipeline

In order for the Pipeline to be executed on PR and release, we must import it into our Jenkins X cluster. 

We can do this by running this command:


```python
!jx import --no-draft=true
```

As soon as we import the repository into Jenkins X, the release path gets triggered.

We can see the activities that have been triggered by running:


```python
!jx get activities
```


```python
And we can actually see the logs of what is happening at every step by running:
```


```python
!jx get build logs "$GIT_USERNAME/seldon-jx-mlops/master #1 release"
```


```python
As we can see, the `release` trigger is working as expected. We can now trigger the PR by opening a PR.

For this, let's add a small change and push a PR:
```


```bash
%%bash 

# Create new branch and move into it
git checkout -b feature-1

# Add an extra space at the end
echo " " >> jenkins-x.yml
git add jenkins-x
git commit -m "Added extra space to trigger master"
git push origin feature-1

# Now create pull request
git request-pull -p origin/master ./
```


```python
Once we create the pull request we can visualise that the PR has been created and the bot has commented.

We would now also be able to see that the tests are now running, and similar to above we can see the logs with:
```


```python
!kubectl get build logs "$GIT_USERNAME/seldon-jx-mlops/pr-1 #1 pr-build"
```

### Pushing images automatically
Now that we're able to build some tests, we want to update the images so we can have the latest on each release.

For this, we will have to add a couple of things, including:

1. The task in the `jenkins-x.yml` file that would allow us to build and push the image
2. The config in the `jenkins-x.yml` to provide docker authentications (to push images)
3. A script that starts a docker daemon and then builds+psuhes the images

#### JX Task to Build and Push image

For this, we would just have to append the following task in our jenkins file:
    
```
    - name: build-and-push-images
      command: bash
      args:
      - assets/scripts/build_and_push_docker_daemon.sh
```

#### Config to provide docker authentication

This piece is slightly more extensive, as we will need to use Docker to build out containers due to the dependency on `s2i` to build the model wrappers.

First we need to define the volumes that we'll be mounting to the container.

The first few volumes before basically consist of the core components that docker will need to be able to run.
```
          volumes:
            - name: modules
              hostPath:
                path: /lib/modules
                type: Directory
            - name: cgroup
              hostPath:
                path: /sys/fs/cgroup
                type: Directory
            - name: dind-storage
              emptyDir: {}
```
We also want to mount the docker credentials which we will generate in the next step.
```
            - name: jenkins-docker-config-volume
              secret:
                items:
                - key: config.json
                  path: config.json
                secretName: jenkins-docker-cfg
```
Once we've created the volumes, now we just need to mount them. This can be done as follows:
```
        options:
          containerOptions:
            volumeMounts:
              - mountPath: /lib/modules
                name: modules
                readOnly: true
              - mountPath: /sys/fs/cgroup
                name: cgroup
              - name: dind-storage
                mountPath: /var/lib/docker                 
```
And finally we also mount the docker auth configuration so we don't have to run `docker login`:
```
              - mountPath: /builder/home/.docker
                name: jenkins-docker-config-volume
```

And to finalise, we need to make sure that the pod can run with privileged context.

The reason why this is required is in order to be able to run the docker daemon:
```
            securityContext:
              privileged: true
```

### Updating Jenkins X file and testing

Now that we've gotten a breakdown of the different additions for the `jenkins-x.yml` file, we can update it:


```python
%%writefile jenkins-x.yml
buildPack: none
pipelineConfig:
  pipelines:
    release:
      pipeline:
        agent:
          image: seldonio/core-builder:0.4
        stages:
        - name: build-and-test
          parallel:
          - name: test-and-deploy-sklearn-server
            steps:
            - name: test-sklearn-server
              steps:
              - name: run-tests
                command: make
                args:
                - install_dev
                - test
            - name: build-and-push-images
              command: bash
              args:
              - assets/scripts/build_and_push_docker_daemon.sh
        options:
          containerOptions:
            volumeMounts:
              - mountPath: /lib/modules
                name: modules
                readOnly: true
              - mountPath: /sys/fs/cgroup
                name: cgroup
              - name: dind-storage
                mountPath: /var/lib/docker
              - mountPath: /builder/home/.docker
                name: jenkins-docker-config-volume
            securityContext:
              privileged: true
          volumes:
            - name: modules
              hostPath:
                path: /lib/modules
                type: Directory
            - name: cgroup
              hostPath:
                path: /sys/fs/cgroup
                type: Directory
            - name: dind-storage
              emptyDir: {}
            - name: jenkins-docker-config-volume
              secret:
                items:
                - key: config.json
                  path: config.json
                secretName: jenkins-docker-cfg
    pullRequest:
      pipeline:
        agent:
          image: seldonio/core-builder:0.4
        stages:
        - name: build-and-test
          parallel:
          - name: test-and-deploy-sklearn-server
            steps:
            - name: test-sklearn-server
              steps:
              - name: run-tests
                command: make
                args:
                - install_dev
                - test
            - name: build-and-push-images
              command: bash
              args:
              - assets/scripts/build_and_push_docker_daemon.sh
        options:
          containerOptions:
            volumeMounts:
              - mountPath: /lib/modules
                name: modules
                readOnly: true
              - mountPath: /sys/fs/cgroup
                name: cgroup
              - name: dind-storage
                mountPath: /var/lib/docker
              - mountPath: /builder/home/.docker
                name: jenkins-docker-config-volume
            securityContext:
              privileged: true
          volumes:
            - name: modules
              hostPath:
                path: /lib/modules
                type: Directory
            - name: cgroup
              hostPath:
                path: /sys/fs/cgroup
                type: Directory
            - name: dind-storage
              emptyDir: {}
            - name: jenkins-docker-config-volume
              secret:
                items:
                - key: config.json
                  path: config.json
                secretName: jenkins-docker-cfg
```

    Overwriting jenkins-x.yml



```python

```
