from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

ITEMS = []

router = APIRouter()


class Item(BaseModel):

    id: int


class Items(BaseModel):

    items: List[Item]


@router.get('/items')
def get_items():
    return {'items': ITEMS}


@router.post('/items')
def create_items():
    new_entry_id = len(ITEMS) + 1
    user_entry = Item(
        id=new_entry_id)
    ITEMS.append(user_entry.dict())
    return user_entry


@router.get('/items/{item_id}')
def get_item(*, item_id: int):
    result = [recipe for recipe in ITEMS if recipe["id"] == item_id]
    if result:
        return result[0]


@router.put('/items/{item_id}')
def update_item(*, item_id: int):
    return {'msg': f'Item {item_id} updated'}


@router.delete('/items/{item_id}')
def delete_item(*, item_id: int):
    return {'msg': f'Item {item_id} deleted'}
