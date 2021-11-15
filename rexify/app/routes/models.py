from typing import List

from pydantic import BaseModel
from fastapi import APIRouter

MODELS = []

router = APIRouter()


class Model(BaseModel):

    name: str


class Models(BaseModel):

    models: List[Model]


@router.get('/models')
def get_models():
    return {'models': MODELS}


@router.post('/models')
def create_model(*, model: Model):
    model_entry = Model(name=model.name)
    MODELS.append(model_entry.dict())
    return model_entry


@router.get('/models/{model_name}')
def get_model(*, model_name: str):
    result = [recipe for recipe in MODELS if recipe["name"] == model_name]
    if result:
        return result[0]


@router.put('/models/{model_name}')
def update_model(*, model_name: int):
    return {'msg': f'Model {model_name} updated'}


@router.delete('/models/{model_name}')
def delete_model(*, model_name: str):
    return {'msg': f'Model {model_name} deleted'}
