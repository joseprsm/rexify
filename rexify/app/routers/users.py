from fastapi import APIRouter

from rexify.app.schemas import User

USERS = []

router = APIRouter(
    prefix='/users',
    tags=['users']
)


@router.get('/')
def get_users():
    return {'users': USERS}


@router.post('/')
def create_users():
    new_entry_id = len(USERS) + 1
    user_entry = User(id=new_entry_id)
    USERS.append(user_entry.dict())
    return user_entry


@router.get('/{user_id}')
def get_user(*, user_id: int):
    result = [user for user in USERS if user["id"] == user_id]
    if result:
        return result[0]


@router.delete('/{user_id}')
def delete_user(*, user_id: int):
    return {'msg': f'User {user_id} deleted'}


@router.put('/{user_id}')
def update_user(*, user_id: int):
    return {'msg': f'User {user_id} updated'}
