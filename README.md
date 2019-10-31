
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
!mkdir -p src/
```


```python
%%writefile src/ModelWrapper.py

from seldon_core.storage import Storage
import joblib

class ModelWrapper:
    def __init__(self, model_uri):
        model_file = os.path.join(Storage.download(model_uri), JOBLIB_FILE)
        self._model = joblib.load(model_file)

        
```


```python

```
