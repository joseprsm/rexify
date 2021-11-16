from fastapi import APIRouter

from rexify.app.schemas import Item

ITEMS = []

router = APIRouter(
    prefix='/items',
    tags=['items']
)


@router.get('/')
def get_items():
    return {'items': ITEMS}


@router.post('/')
def create_items():
    new_entry_id = len(ITEMS) + 1
    user_entry = Item(
        id=new_entry_id)
    ITEMS.append(user_entry.dict())
    return user_entry


@router.get('/{item_id}')
def get_item(*, item_id: int):
    result = [recipe for recipe in ITEMS if recipe["id"] == item_id]
    if result:
        return result[0]


@router.put('/{item_id}')
def update_item(*, item_id: int):
    return {'msg': f'Item {item_id} updated'}


@router.delete('/items/{item_id}')
def delete_item(*, item_id: int):
    return {'msg': f'Item {item_id} deleted'}
