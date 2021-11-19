# noinspection PyPackageRequirements
from airflow.api.client.local_client import Client

from fastapi import APIRouter

PIPELINE_NAME = 'rexify'

router = APIRouter(tags=['pipelines'])

c = Client(None, None)


@router.post('/run')
def run():
    # c.trigger_dag(PIPELINE_NAME)
    return {'msg': 'Rexify pipeline running'}

