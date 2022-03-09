from typing import Text, Optional

from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.types import ComponentSpec, standard_artifacts, channel_utils
from tfx.types.component_spec import ExecutionParameter, ChannelParameter

import tempfile

from . import executor


class DownloaderSpec(ComponentSpec):

    INPUTS = {}

    OUTPUTS = {
        'events_path': ChannelParameter(standard_artifacts.String),
        'users_path': ChannelParameter(standard_artifacts.String),
        'items_path': ChannelParameter(standard_artifacts.String)
    }

    PARAMETERS = {}


class Downloader(BaseComponent):

    SPEC_CLASS = DownloaderSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 output_path: Optional[str] = None):
        output_path = output_path or channel_utils.as_channel([
            standard_artifacts.String(tempfile.mkdtemp())])
        spec = DownloaderSpec(output_path=output_path)
        super(Downloader, self).__init__(spec=spec)
