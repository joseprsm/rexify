from typing import Optional, List
from pydantic import BaseModel


class BaseFeature(BaseModel):

    key: str
    value: Optional[str]


class Feature(BaseFeature):

    id: int


class BaseTarget(BaseModel):

    external_id: str
    features: Optional[List[BaseFeature]]


class Item(BaseTarget):

    id: int


class Items(BaseModel):

    items: List[Item]


class BaseEvent(BaseModel):

    user_id: int
    item_id: int
    context: Optional[List[BaseFeature]]


class Event(BaseEvent):

    id: Optional[int]


class Events(BaseModel):

    events: List[Event]


class Model(BaseModel):

    name: str


class Models(BaseModel):

    models: List[Model]


class User(BaseTarget):

    id: int


class Users(BaseModel):

    users: List[User]
