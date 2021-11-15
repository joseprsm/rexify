from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

USERS = []

router = APIRouter()


class User(BaseModel):

    id: Optional[int]


class Users(BaseModel):

    users: List[User]


@router.get('/users')
def get_users():
    return {'users': USERS}


@router.post('/users')
def create_users():
    new_entry_id = len(USERS) + 1
    user_entry = User(id=new_entry_id)
    USERS.append(user_entry.dict())
    return user_entry


@router.get('/users/{user_id}')
def get_user(*, user_id: int):
    result = [user for user in USERS if user["id"] == user_id]
    if result:
        return result[0]


@router.delete('/users/{user_id}')
def delete_user(*, user_id: int):
    return {'msg': f'User {user_id} deleted'}


@router.put('/users/{user_id}')
def update_user(*, user_id: int):
    return {'msg': f'User {user_id} updated'}
