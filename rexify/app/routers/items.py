from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from rexify.app import crud
from rexify.app.db import get_db
from rexify.app.schemas import Item, Feature, BaseTarget

ITEMS = []

router = APIRouter(
    prefix='/items',
    tags=['items']
)


@router.get('/')
def get_items(skip: Optional[int] = 0, limit: Optional[int] = 10, db: Session = Depends(get_db)):
    return crud.get_items(db, skip, limit)


@router.post('/')
def create_item(item: BaseTarget, db: Session = Depends(get_db)):
    db_item = crud.get_item_id(db, item.external_id)
    if db_item:
        raise HTTPException(status_code=400, detail="Item already registered")
    return crud.create_item(db, item)


@router.get('/{item_id}')
def get_item(item_id: int, db: Session = Depends(get_db)):
    item = crud.get_item(db, item_id=item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.put('/{item_id}')
def update_item(*, item_id: int):
    return {'msg': f'Item {item_id} updated'}


@router.delete('/items/{item_id}')
def delete_item(*, item_id: int):
    return {'msg': f'Item {item_id} deleted'}
