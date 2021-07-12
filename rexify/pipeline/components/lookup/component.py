from typing import Text, Optional

from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.types import standard_artifacts, Channel, channel_utils
from tfx.types.component_spec import ComponentSpec, ChannelParameter, ExecutionParameter

from . import executor


class EmbeddingLookupSpec(ComponentSpec):

    INPUTS = {
        "examples": ChannelParameter(type=standard_artifacts.Examples),
        "model": ChannelParameter(type=standard_artifacts.Model)
    }

    OUTPUTS = {
        "lookup_model": ChannelParameter(type=standard_artifacts.Model)
    }

    PARAMETERS = {
        "query_model": ExecutionParameter(type=Text),
        "feature_key": ExecutionParameter(type=Text),
        "schema": ExecutionParameter(type=Text)
    }


class EmbeddingLookup(BaseComponent):

    SPEC_CLASS = EmbeddingLookupSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 examples: Channel,
                 model: Channel,
                 lookup_model: Channel = None,
                 query_model: Optional[Text] = None,
                 feature_key: Optional[Text] = None,
                 schema: Optional[Text] = None):

        lookup_model = lookup_model or channel_utils.as_channel([standard_artifacts.Model()])
        spec = EmbeddingLookupSpec(
            examples=examples,
            model=model,
            lookup_model=lookup_model,
            query_model=query_model,
            feature_key=feature_key,
            schema=schema)
        super(EmbeddingLookup, self).__init__(spec=spec)
