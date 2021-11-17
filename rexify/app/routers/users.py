from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session

from rexify.app import crud
from rexify.app.db import get_db
from rexify.app.schemas import User

USERS = []

router = APIRouter(
    prefix='/users',
    tags=['users']
)


@router.get('/')
def get_users(skip: Optional[int] = 0, limit: Optional[int] = 10, db: Session = Depends(get_db)):
    return crud.get_users(db, skip, limit)


@router.post('/', response_model=User)
def create_user(user: User, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_external_id(db, user.external_id)
    if db_user:
        raise HTTPException(status_code=400, detail="User already registered")
    return crud.create_user(db, user)


@router.get('/{user_id}')
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.delete('/{user_id}')
def delete_user(*, user_id: int):
    return {'msg': f'User {user_id} deleted'}


@router.put('/{user_id}')
def update_user(*, user_id: int):
    return {'msg': f'User {user_id} updated'}
