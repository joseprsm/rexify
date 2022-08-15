# Architecture

Rexify has two main components: the `FeatureExtractor` and the `Recommender`. 

The former basically takes the original data, and learns all the transformations
that need to be applied to the dataset. The output is a `tf.data.Dataset` with the 
right structure to be passed on to the `Recommender` model. 

This `Recommender` is a TensorFlow model with a dynamic architecture, which adapts
itself according to the schema fed to the `FeatureExtractor`.

## Feature Extractor

The `FeatureExtractor` is a scikit-learn Transformer. It implements a `.fit()` 
and a `.transform()` method that apply a set of transformations on the data.

Essentially, it has a `_ppl` attribute which is a `sklearn.pipeline.Pipeline`; 
the pipeline steps are set according to the `schema` passed during instantiation, 
which are scikit-learn Transformers themselves.

For example, an attribute classified as `id` would create a pipeline step with a 
`sklearn.compose.ColumnTransformer`, composed of  a single `sklearn.preprocessing.OrdinalEncoder` 
Transformer. 

Additionally, it subclasses `rexify.features.TfDatasetGenerator`, which converts 
the output of the transformations of the `FeatureExtractor` into a `tf.data.Dataset`, 
with a nested structure such as this:

```
{
  "query": {
    "user_id": tf.Tensor([]),
    "user_features": tf.Tensor([]),
    "context": tf.Tensor([]),
  },
  "candidate": {
    "item_id": tf.Tensor([]),
    "item_features": tf.Tensor([])
  }
}
```

With this structure, the Recommender model can call a different set of layers for 
the user and item ID attributes, and the remaining transformed features.  

## Recommender

The `Recommender` is a `tfrs.models.Model`, which subclasses `tf.keras.Model` 
and overrides the `.train_step()` method. According to the [TensorFlow Recommenders documentation](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/models/Model):

> Many recommender models are relatively complex, and do not neatly 
> fit into supervised or unsupervised paradigms. This base class makes it easy to 
> define custom training and test losses for such complex models.

In this case, we use the Recommender model, to create a two tower model architecture, as explained [here](https://research.google/pubs/pub48840/). 
In short, it's composed of two main models, a Query model and a Candidate model, both of which learn
to represent queries and candidates in the same vector space.

<p align="center">
    <img src="https://1.bp.blogspot.com/-ww8cKT3nIb8/X2pdWAWWNmI/AAAAAAAADl8/pkeFRxizkXYbDGbOcaAnZkorjEuqtrabgCLcBGAsYHQ/s0/TF%2BRecommenders%2B06.gif" style="width:500px">
</p>

Basically, it takes the `tf.data.Dataset` output by the `FeatureExtractor` and passes it by the two 
Query and Candidate model towers. Due to the nested structure of the dataset, we're able to get and apply
different transformations to different sets of features.

### Query Tower

The Query Tower is responsible with learning a representation for the queries. That representation is a
combination between the user embedding, and the features learned from the remaining
user and context attributes.

Essentially, it takes the user ID attribute and passes it to an Embedding layer. The user and context 
features are concatenated and passed to a model composed of Dense layers. The output of that model and 
the user embedding are then concatenated and subsequently fed to another set of Dense layers.

The resulting vector should represent a single query, which can be used to compute the similarity
to the candidate vectors.

### Candidate Tower

In essence, the Candidate Tower shares the same behavior as the Query's. The key difference is that instead 
of using the user ID and features and context, it solely uses the item ID and remaining features.

On a deeper level, it takes the item ID attribute and passes it to an Embedding layer. The item features are
passed to a set of Dense layers. The output of these layers and the Embedding layer are then concatenated and
then passed to another set of Dense layers.

The resulting vector should represent a single candidate, or item, in this case, which can be used to compute 
the similarity to a query vector or between other candidate vectors.