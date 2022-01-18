from typing import Dict, List, Any, Optional

import os
import numpy as np
import pandas as pd
import sqlalchemy as db

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.proto.orchestration import execution_result_pb2

DB_USER = os.environ.get('DB_USER')
DB_PASS = os.environ.get('DB_PASS')
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT')
DB_NAME = os.environ.get('DB_NAME')

DATABASE_URL = os.environ.get('DB_URL', f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

engine = db.create_engine(DATABASE_URL)
metadata = db.MetaData
conn = engine.connect()


class Executor(base_executor.BaseExecutor):

    def Do(self,
           input_dict: Dict[str, List[types.Artifact]],
           output_dict: Dict[str, List[types.Artifact]],
           exec_properties: Dict[str, Any]) -> Optional[execution_result_pb2.ExecutorOutput]:

        events = self._get('events')  # (id, user_id, item_id)
        context = self._get('event_features')  # (id, event_id, feature_id, value)
        features = self._get('features')  # (id, key)

        context = context.merge(features, left_on='feature_id', right_on='id')  # (feature_id, event_id, key, value)
        context.drop('feature_id', inplace=True)

        context_g = context.groupby('event_id').agg({'key': list, 'value': list})

        contexts = list()
        for event_id in context_g.index:
            a = pd.DataFrame(np.stack(context_g.loc[event_id].values)).T.set_index(0).T
            a.index = [event_id]
            contexts.append(a)

        contexts = pd.concat(contexts).reset_index()
        contexts['event_id'] = contexts.pop('index')

        events = events.merge(contexts, left_on='id', right_on='event_id').drop('id', axis=1)

    @staticmethod
    def _get(table) -> pd.DataFrame:
        events = db.Table(table, metadata, autoload=True, autoload_with=engine)
        query = db.select([events])
        return pd.DataFrame(conn.execute(query).fetchall())
