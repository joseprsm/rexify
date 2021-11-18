from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session

from rexify.app import crud
from rexify.app.db import get_db
from rexify.app.schemas import BaseTarget

import rexify.app.crud.items

router = APIRouter(
    prefix='/items',
    tags=['items'])


@router.get('/')
def get_items(skip: Optional[int] = 0, limit: Optional[int] = 10, db: Session = Depends(get_db)):
    return crud.items.get_list(db, skip, limit)


@router.post('/')
def create_item(item: BaseTarget, db: Session = Depends(get_db)):
    db_item = crud.items.get_id(db, external_id=item.external_id)
    if db_item:
        raise HTTPException(status_code=400, detail="Item already registered")
    return crud.items.create(db, item)


@router.get('/{item_id}')
def get_item(item_id: int, db: Session = Depends(get_db)):
    item = crud.items.get(db, item_id=item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.put('/{item_id}')
def update_item(*, item_id: int):
    return {'msg': f'Item {item_id} updated'}


@router.delete('/{item_id}')
def delete_item(item_id: int, db: Session = Depends(get_db)):
    return crud.items.delete(db, item_id=item_id)
