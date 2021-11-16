from fastapi import APIRouter

from rexify.app.schemas import Model

MODELS = []

router = APIRouter(
    prefix='/models',
    tags=['models']
)


@router.get('/')
def get_models():
    return {'models': MODELS}


@router.post('/')
def create_model(*, model: Model):
    model_entry = Model(name=model.name)
    MODELS.append(model_entry.dict())
    return model_entry


@router.get('/{model_name}')
def get_model(*, model_name: str):
    result = [recipe for recipe in MODELS if recipe["name"] == model_name]
    if result:
        return result[0]


@router.put('/{model_name}')
def update_model(*, model_name: int):
    return {'msg': f'Model {model_name} updated'}


@router.delete('/{model_name}')
def delete_model(*, model_name: str):
    return {'msg': f'Model {model_name} deleted'}
