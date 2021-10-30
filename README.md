# Rexify :t-rex:

[![CircleCI](https://circleci.com/gh/joseprsm/rexify/tree/main.svg?style=shield&circle-token=d2f4a46a4daf02ba3c0e1968ebde4a0d2e50df36)](https://circleci.com/gh/joseprsm/rexify/tree/main)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2fa652f8d3564387acfbb085572d49f1)](https://www.codacy.com/gh/joseprsm/rexify/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=joseprsm/rexify&amp;utm_campaign=Badge_Grade)
![GitHub](https://img.shields.io/github/license/joseprsm/rexify)

Rexify is a library to streamline recommender systems model development. It is built on
top of [Tensorflow Recommenders](https://github.com/tensorflow/recommenders) models and 
[TFX](https://github.com/tensorflow/tfx) pipelines.


## Installation

For now, you can only install Rexify from source:
```shell
pip install git+https://www.github.com/joseprsm/rexify
```

## Quick start

All you need to set up your model is interaction data - something that kind of looks like this: 

|  user_id | item_id | timestamp | item_name | event_type | 
| --- | --- | --- | --- | --- |
| 22 | 67 | 2021/05/13 | Blue Jeans | Purchase |
| 37 | 9 | 2021/04/11 | White Shirt | Page View |
| 22 | 473 | 2021/04/11 | Red Purse | Add to Cart |
| ... | ... | ... | ... | ... |
| 358 | 51 | 2021/04/11 | Bracelet | Purchase |

Additionally, we'll have to have configured a schema for the data.
This schema is what will allow Rexify to generate a dynamic model and preprocessing steps.
The schema should be comprised of three dictionaries: `user`, `ìtem`, `context`. 

Each of these dictionaries should consist of features and internal data types, 
such as: `categorical`, `continuous`, `timestamp`, `text`. More data types will be available 
in the future.

```json
{
  "user": {
    "user_id": "categorical"
  },
  "item": {
    "item_id": "categorical",
    "timestamp": "timestamp",
    "item_name": "text"
  },
  "context": {
    "event_type": "categorical"
  }
}
``` 

Then, we got to set up our runs.

```
rexify config --schema path/to/schema/json
```

And finally, now that we have everything set up, all we got to do is run `rexify run`, 
and point to the path where the CSV are stored like so:

```
rexify run -e path/to/events/csv 
```

