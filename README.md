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

events = pd.read_csv('path/to/event/data')
schema = json.load('path/to/schema')

feat = FeatureExtractor(schema)
prep_data = feat.fit_transform(events)
ds = feat.make_dataset(prep_data)

model = Recommender(schema, **feat.model_params)
model.compile()
model.fit(ds)
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

The prebuilt pipeline consists of 5 components:

1. `download`, which downloads the event data from URLs set on the `$INPUT_DATA_URL` and `$SCHEMA_URL` environment variables
2. `load`, which prepares the data downloaded in the previous step
3. `train`, which trains a `Recommender` model on the preprocessed data
4. `index`, which trains a [ScaNN](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html) model to retrieve the nearest neighbors
5. `retrieval`, which basically retrieves the nearest _k_ neighbors for each of the known users
