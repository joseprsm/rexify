from typing import Optional, List
from pydantic import BaseModel


class Feature(BaseModel):

    id: Optional[int]
    key: str
    value: str


class Item(BaseModel):

    id: Optional[int]
    external_id: str
    features: Optional[List[Feature]]


class Items(BaseModel):

    items: List[Item]


class Event(BaseModel):

    id: Optional[int]
    user_id: int
    item_id: int
    context: Optional[List[Feature]]


class Events(BaseModel):

    events: List[Event]


class Model(BaseModel):

    name: str


class Models(BaseModel):

    models: List[Model]


class User(BaseModel):

    id: Optional[int]
    external_id: str
    features: Optional[List[Feature]]


class Users(BaseModel):

    users: List[User]
