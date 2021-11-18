from typing import List

from sqlalchemy.orm import Session

from rexify.app import models, schemas
from rexify.app.crud import features


def get(db: Session, event_id: int):
    return db.query(models.Event).filter(models.Event.id == event_id).first()


def get_list(db: Session, skip: int = 0, limit: int = 20):
    return db.query(models.Event).offset(skip).limit(limit).all()


def create(db: Session, event: schemas.BaseEvent):
    db_event = models.Event(user_id=event.user_id, item_id=event.item_id)
    db.add(db_event)
    db.commit()
    db.refresh(db_event)

    for feature in event.context:
        if bool(features.get_by_key(db, feature.key)):
            create_feature(db, db_event.id, feature)

    return db_event


def update(db: Session, user_id: int, feature_list: List[schemas.Feature]):
    raise NotImplementedError


def delete(db: Session, event_id: int):
    db.query(models.Event).filter(models.Event.id == event_id).delete()


def get_features(db: Session, event_id: int):
    return db.query(models.EventFeature).filter(models.EventFeature.event_id == event_id).all()


def create_feature(db: Session, event_id: int, feature: schemas.Feature):
    feature_id: int = features.get_by_key(db, key=feature.key).id

    event_feature = models.EventFeature(
        item_id=event_id,
        feature_id=feature_id,
        value=feature.value)

    db.add(event_feature)
    db.commit()
    db.refresh(event_feature)
    return event_feature
