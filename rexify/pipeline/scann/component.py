from typing import Optional, Text

from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.types import standard_artifacts, Channel, channel_utils
from tfx.types.component_spec import ComponentSpec, ChannelParameter, ExecutionParameter

from . import executor


class ScaNNGenSpec(ComponentSpec):

    INPUTS = {
        "candidates": ChannelParameter(standard_artifacts.Examples),
        "model": ChannelParameter(standard_artifacts.Model),
        "lookup_model": ChannelParameter(standard_artifacts.Model),
    }

    OUTPUTS = {
        "index": ChannelParameter(standard_artifacts.Model)
    }

    PARAMETERS = {
        "schema": ExecutionParameter(Text),
        "feature_key": ExecutionParameter(Text)
    }


class ScaNNGen(BaseComponent):

    SPEC_CLASS = ScaNNGenSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 candidates: Channel,
                 model: Channel,
                 lookup_model: Channel,
                 index: Optional[Channel] = None,
                 schema: Optional[Text] = None,
                 feature_key: Optional[Text] = None):
        index = index or channel_utils.as_channel([standard_artifacts.Model()])
        spec = ScaNNGenSpec(
            candidates=candidates,
            model=model,
            lookup_model=lookup_model,
            index=index,
            schema=schema,
            feature_key=feature_key)
        super().__init__(spec=spec)
