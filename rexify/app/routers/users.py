from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session

from rexify.app import crud
from rexify.app.db import get_db
from rexify.app.schemas import BaseTarget

import rexify.app.crud.users

router = APIRouter(
    prefix='/users',
    tags=['users'])


@router.get('/')
def get_users(skip: Optional[int] = 0, limit: Optional[int] = 10, db: Session = Depends(get_db)):
    return crud.users.get_list(db, skip, limit)


@router.post('/')
def create_user(user: BaseTarget, db: Session = Depends(get_db)):
    db_user = crud.users.get_id(db, user.external_id)
    if db_user:
        raise HTTPException(status_code=400, detail="User already registered")
    return crud.users.create(db, user)


@router.get('/{user_id}')
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = crud.users.get(db, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.delete('/{user_id}')
def delete_user(*, user_id: int):
    return {'msg': f'User {user_id} deleted'}


@router.put('/{user_id}')
def update_user(user_id: int, db: Session = Depends(get_db)):
    return crud.users.delete(db, user_id=user_id)
