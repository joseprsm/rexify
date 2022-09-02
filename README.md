<p align="center">
    <br>
    <img src="https://storage.googleapis.com/rexify/1659986918545.png" height="200"/>
    <br>
<p>

<p align="center">
    <a href="https://circleci.com/gh/joseprsm/rexify">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/joseprsm/rexify?style=flat-square">
    </a>
    <a href="https://github.com/joseprsm/rexify/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/joseprsm/rexify?style=flat-square">
    </a>
    <a href="https://rexify.readthedocs.io">
        <img alt="Documentation" src="https://img.shields.io/badge/documentation-online-success?style=flat-square">
    </a>
    <a href="https://pypi.org/project/rexify/">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/joseprsm/rexify?style=flat-square">
    </a>
</p>

Rexify is a library to streamline recommender systems model development. It is built on
top of [Tensorflow Recommenders](https://github.com/tensorflow/recommenders) models and 
[Kubeflow](https://github.com/kubeflow/pipelines) pipelines.

In essence, Rexify adapts dynamically to your data, and outputs high-performing TensorFlow
models that may be used wherever you want, independently of your data. Rexify also includes
modules to deal with feature engineering as Scikit-Learn Transformers and Pipelines.

With Rexify, users may easily train Recommender Systems models, just by specifying what their
data looks like. Rexify also comes equipped with pre-built machine learning pipelines which can
be used serverlessly. 

## What is Rexify?

Rexify is a low-code personalization tool, that makes use of traditional machine learning 
frameworks, such as Scikit-Learn and TensorFlow, and Kubeflow to create scalable Recommender Systems
workflows that anyone can use.

### Who is it for?

Rexify is a project that simplifies and standardizes the workflow of recommender systems. It is 
mostly geared towards people with little to no machine learning knowledge, that want to implement
somewhat scalable Recommender Systems in their applications.

## Installation

The easiest way to install Rexify is via `pip`:

```shell
pip install rexify
```

## Quick Tour

Rexify is meant to be usable right out of the box. All you need to set up your model is interaction data - something that kind of looks like this:

| user_id | item_id | timestamp  | item_name   | event_type  |
|---------|---------|------------|-------------|-------------|
| 22      | 67      | 2021/05/13 | Blue Jeans  | Purchase    |
| 37      | 9       | 2021/04/11 | White Shirt | Page View   |
| 22      | 473     | 2021/04/11 | Red Purse   | Add to Cart |
| ...     | ...     | ...        | ...         | ...         |
| 358     | 51      | 2021/04/11 | Bracelet    | Purchase    |

Additionally, we'll have to have configured a schema for the data.
This schema is what will allow Rexify to generate a dynamic model and preprocessing steps.
The schema should be comprised of three dictionaries: `user`, `ìtem`, `context`.

Each of these dictionaries should consist of features and internal data types, 
such as: `id`, `categorical`, `timestamp`, `text`. More data types will be available 
in the future.

```json
{
  "user": {
    "user_id": "id"
  },
  "item": {
    "item_id": "id",
    "timestamp": "timestamp",
    "item_name": "text"
  },
  "context": {
    "event_type": "categorical"
  }
}
```

Essentially, what Rexify will do is take the schema, and dynamically adapt to the data.

### As a package

There are two main components in Rexify workflows: `FeatureExtractor` and `Recommender`.

The `FeatureExtractor` is a scikit-learn Transformer that basically takes the schema of the data, and transforms the event data accordingly. Another method `.make_dataset()`, converts the transformed data into a `tf.data.Dataset`, all correctly configured to be fed to the `Recommender` model.

`Recommender` is a `tfrs.Model` that basically implements the Query and Candidate towers. During training, the Query tower will take the user ID, user features, and context, to learn an embedding; the Candidate tower will do the same for the item ID and its features. 

More information about how the `FeatureExtractor` and the `Recommender` works can be found [here](https://rexify.readthedocs.io/en/latest/overview/architecture.html). 

A sample Rexify workflow should sort of look like this:

````python
import json
import pandas as pd

from rexify.features import FeatureExtractor
from rexify.models import Recommender

events = pd.read_csv('path/to/events/data')
with open('path/to/schema') as f:
    schema = json.load(f)

feat = FeatureExtractor(schema)
ds = feat.fit_transform(events).batch(512)

model = Recommender(**feat.model_params)
model.compile()
model.fit(ds)
````

When training is complete, you'll have a trained `tf.keras.Model` ready to be used, as you normally would. 

### As a prebuilt pipeline

After cloning this project and setting up the necessary environment variables, you can run:

```shell
python -m rexify.pipeline
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


### Via the demo application

After cloning the project, install the demo dependencies and run the Streamlit application:

```shell
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

Or, if you're using docker:

```shell
docker run joseprsm/rexify-demo
```

You can then follow the steps here to set up your pipeline. 

During setup, you'll be asked to either input a publicly available dataset URL or use a sample data set.
After that, you'll have a form to help you set up the schema for the data.

Finally, after hitting "Compile", you'll have your Pipeline Spec ready. The resulting JSON file can then 
be uploaded to Vertex AI Pipelines or Kubeflow, seamlessly.

The key difference from this pipeline to the prebuilt one is that instead of using the `download` 
component to download the schema, it will pass it as an argument to the pipeline, and then use a `copy` 
component to pass it down as an artifact.

## License

[MIT](https://github.com/joseprsm/rexify/blob/main/LICENSE)
