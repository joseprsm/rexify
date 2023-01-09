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

Rexify is a library to streamline recommender systems model development.

In essence, Rexify adapts dynamically to your data, and outputs high-performing TensorFlow
models that may be used wherever you want, independently of your data. Rexify also includes
modules to deal with feature engineering as Scikit-Learn Transformers and Pipelines.

With Rexify, users may easily train Recommender Systems models, just by specifying what their
data looks like. Rexify also comes equipped with pre-built machine learning pipelines which can
be used serverlessly. 

## What is Rexify?

Rexify is a low-code personalization tool, that makes use of traditional machine learning 
frameworks, such as Scikit-Learn and TensorFlow, to create scalable Recommender Systems
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

Rexify is meant to be usable right out of the box. All you need to set up your model is interaction
data - something that kind of looks like this:

| user_id | item_id | timestamp  | event_type  |
|---------|---------|------------|-------------|
| 22      | 67      | 2021/05/13 | Purchase    |
| 37      | 9       | 2021/04/11 | Page View   |
| 22      | 473     | 2021/04/11 | Add to Cart |
| ...     | ...     | ...        | ...         |
| 358     | 51      | 2021/04/11 | Purchase    |

Additionally, we'll have to have configured a schema for the data.
This schema is what will allow Rexify to generate a dynamic model and preprocessing steps.
The schema should be comprised of two dictionaries (`user`, `ìtem`) and two key-value 
pairs: `event_type` (which should point to the column of the event type) and `timestamp` (
which should point to the timestamp column)

Each of these dictionaries should consist of features and internal data types, 
such as: `id`, `category`, `number`. More data types will be available 
in the future.

```json
{
  "user": {
    "user_id": "id",
    "age": "number"
  },
  "item": {
    "item_id": "id",
    "category": "category"
  },
  "timestamp": "timestamp"
  "event_type": "event_type"
}
```

Essentially, what Rexify will do is take the schema, and dynamically adapt to the data.

There are two main components in Rexify workflows: `FeatureExtractor` and `Recommender`.

The `FeatureExtractor` is a scikit-learn Transformer that basically takes the schema of 
the data, and transforms the event data accordingly. Another method `.make_dataset()`, 
converts the transformed data into a `tf.data.Dataset`, all correctly configured to be fed
to the `Recommender` model.

`Recommender` is a `tfrs.Model` that basically implements the Query and Candidate towers. 
During training, the Query tower will take the user ID, user features, and context, to 
learn an embedding; the Candidate tower will do the same for the item ID and its features. 

More information about how the `FeatureExtractor` and the `Recommender` works can be found 
[here](https://rexify.readthedocs.io/en/latest/overview/architecture.html). 

A sample Rexify workflow should sort of look like this:

````python

import pandas as pd

from rexify import Schema, FeatureExtractor, Recommender

events = pd.read_csv('path/to/events/data')

schema = Schema.load('path/to/schema')

feat = FeatureExtractor(schema, users='path/to/users/data', items='path/to/events/data')
x = feat.fit_transform(events)
x = feat.make_dataset(x)

model = Recommender(**feat.model_params)
model.compile()
model.fit(events, batch_size=512)
````

When training is complete, you'll have a trained `tf.keras.Model` ready to be used, as
you normally would. 

## License

[MIT](https://github.com/joseprsm/rexify/blob/main/LICENSE)
