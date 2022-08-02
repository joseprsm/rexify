# Rexify :t-rex:

![GitHub](https://img.shields.io/github/license/joseprsm/rexify)

Rexify is a library to streamline recommender systems model development. It is built on
top of [Tensorflow Recommenders](https://github.com/tensorflow/recommenders) models and 
[Kubeflow](https://github.com/kubeflow/pipelines) pipelines.

In essence, Rexify adapts dynamically to your data, and outputs high-performing TensorFlow
models that may be used wherever you want, independently of your data. Rexify also includes modules to deal with feature engineering as Scikit-Learn Transformers 
and Pipelines.  


## Installation

For now, you'll have to install Rexify from source

```shell
$ pip install git+https://github.com/joseprsm/rexify.git
```

## Quickstart

There are many ways you can use Rexify on your projects.

### As a package

````python
import json
import pandas as pd

from rexify.features import FeatureExtractor 
from rexify.models import Recommender

events = pd.read_csv('path/to/events/data')
schema = json.load('path/to/schema')

feat = FeatureExtractor(schema)
preprocessed_data = feat.fit_transform(events)

model = Recommender(schema, **feat.model_params)
model.compile()

model.fit(preprocessed_data)
````

### As a prebuilt pipeline

```shell
$ python -m rexify.pipeline
```

Which should output a `pipeline.json` file. You can then upload this file manually to 
either a Kubeflow Pipeline or Vertex AI Pipelines instance, and it should run seamlessly. 

You can also check the [Kubeflow Pipeline](https://kubeflow-pipelines.readthedocs.io/en/latest/source/kfp.client.html#kfp.Client.create_run_from_pipeline_package)
and [Vertex AI](https://cloud.google.com/vertex-ai/docs/pipelines/run-pipeline#create_a_pipeline_run) 
documentation to learn how to submit these pipelines programmatically.  
