from typing import Optional, List
from pydantic import BaseModel


class Item(BaseModel):

    id: Optional[int]


class Items(BaseModel):

    items: List[Item]


class Event(BaseModel):

    id: Optional[int]
    user_id: int
    item_id: int


class Events(BaseModel):

    events: List[Event]


class Model(BaseModel):

    name: str


class Models(BaseModel):

    models: List[Model]


class User(BaseModel):

    id: Optional[int]


class Users(BaseModel):

    users: List[User]
