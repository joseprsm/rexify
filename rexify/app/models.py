from sqlalchemy import Column, Integer, ForeignKey, String

from rexify.app.db import Base


class User(Base):

    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String)


class Item(Base):

    __tablename__ = 'items'
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String)


class Event(Base):

    __tablename__ = 'events'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    item_id = Column(Integer, ForeignKey('items.id'))


class Feature(Base):

    __tablename__ = 'features'
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String)


class UserFeatures(Base):

    __tablename__ = 'user_features'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    feature_id = Column(Integer, ForeignKey('features.id'))
    value = Column(String)


class ItemFeatures(Base):

    __tablename__ = 'item_features'
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey('items.id'))
    feature_id = Column(Integer, ForeignKey('features.id'))
    value = Column(String)


class EventFeatures(Base):

    __tablename__ = 'event_features'
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey('events.id'))
    feature_id = Column(Integer, ForeignKey('features.id'))
    value = Column(String)
