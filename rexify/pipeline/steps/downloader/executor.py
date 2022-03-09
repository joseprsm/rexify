from typing import Dict, List, Any, Optional

import os
import numpy as np
import pandas as pd
import sqlalchemy as db

from tfx import types
from tfx.dsl.io import fileio
from tfx.dsl.components.base import base_executor
from tfx.proto.orchestration import execution_result_pb2
from tfx.types import artifact_utils

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

        events: pd.DataFrame = self.get('event')    # (event_id, user_id, item_id, *context)
        users: pd.DataFrame = self.get('user')      # (user_id, *user_features)
        items: pd.DataFrame = self.get('item')      # (item_id, *item_features)

        events = events.merge(users, on='user_id')  # (event_id, user_id, item_id, *context, *user_features)
        events = events.merge(items, on='item_id')  # (event_id, user_id, item_id, *context, *all_features)

        self.save(events, artifact_utils.get_single_uri(output_dict['events_path']), 'events.csv')
        self.save(events, artifact_utils.get_single_uri(output_dict['users_path']), 'users.csv')
        self.save(events, artifact_utils.get_single_uri(output_dict['items_path']), 'items.csv')

    def get(self, table: str) -> pd.DataFrame:

        if table not in ['event', 'user', 'item']:
            raise Exception('Table not found')

        data = self.query(table + 's')
        context = self.query(f'{table}_features')
        features = self.query('features')

        context = context.merge(features, left_on='feature_id', right_on='id')
        context = context[[f'{table}_id', 'key', 'value']]
        context = context.groupby(f'{table}_id').agg({'key': list, 'value': list})
        context = pd.concat([self.explode(id_, context) for id_ in context.index]).reset_index()

        context[f'{table}_id'] = context.pop('index')

        return data.merge(context, left_on='id', right_on=f'{table}_id')\
            .drop('id', axis=1).set_index(f'{table}_id')

    @staticmethod
    def save(data: pd.DataFrame, path: str, name: str):
        fileio.makedirs(os.path.dirname(path))
        data.to_csv(os.path.join(path, name))

    @staticmethod
    def query(table) -> pd.DataFrame:
        events = db.Table(table, metadata, autoload=True, autoload_with=engine)
        query = db.select([events])
        return pd.DataFrame(conn.execute(query).fetchall())

    @staticmethod
    def explode(id_, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(np.stack(df.loc[id_].values)).T.set_index(0).T
        features.index = [id_]
        return features
