from sqlalchemy import Column, Integer, ForeignKey, String

from .base import Base


class _Identifiable(Base):

    id = Column(Integer, primary_key=True, index=True)


class User(_Identifiable):

    __tablename__ = 'users'


class Item(_Identifiable):

    __tablename__ = 'items'


class Event(_Identifiable):

    __tablename__ = 'events'
    user_id = Column(Integer, ForeignKey('users.id'))
    item_id = Column(Integer, ForeignKey('items.id'))


class Feature(_Identifiable):

    __tablename__ = 'features'
    key = Column(String)


class _TargetFeatures(_Identifiable):

    value = Column(String)


class UserFeatures(_TargetFeatures):

    __tablename__ = 'user_features'
    user_id = Column(Integer, ForeignKey('users.id'))


class ItemFeatures(_TargetFeatures):

    __tablename__ = 'item_features'
    item_id = Column(Integer, ForeignKey('items.id'))


class EventFeatures(_TargetFeatures):

    __tablename__ = 'event_features'
    event_id = Column(Integer, ForeignKey('events.id'))
