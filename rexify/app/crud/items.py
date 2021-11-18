from typing import List

from sqlalchemy.orm import Session

from rexify.app import models, schemas
from rexify.app.crud import features


def get(db: Session, item_id: int):
    item_features = get_features(db, item_id)
    item = db.query(models.Item).filter(models.Item.id == item_id).first()
    return {
        'id': item.id,
        'features': [{
            features.get(db, feature.id).key: feature.value}
            for feature in item_features
        ]
    }


def get_list(db: Session, skip: int = 0, limit: int = 20):
    return db.query(models.Item).offset(skip).limit(limit).all()


def create(db: Session, item: schemas.BaseTarget):
    db_item = models.Item(external_id=item.external_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)

    for feature in item.features:
        if bool(features.get_by_key(db, feature.key)):
            create_feature(db, db_item.id, feature)

    return db_item


def get_id(db: Session, external_id: str):
    return db.query(models.Item).filter(models.Item.external_id == external_id).first()


def update(db: Session, user_id: int, feature_list: List[schemas.Feature]):
    raise NotImplementedError


def delete(db: Session, item_id: int):
    # deletes all item features
    db.query(models.ItemFeature).filter(models.ItemFeature.item_id == item_id).delete()

    # deletes all event features with the deleted item_id
    event_features = get_features(db, item_id)
    for event_feature in event_features:
        db.query(models.EventFeature).filter(models.EventFeature.id == event_feature.id).delete()

    # deletes all events with the deleted item_id
    db.query(models.Event).filter(models.Event.item_id == item_id).delete()

    # deletes the item
    db.query(models.Item).filter(models.Item.id == item_id).delete()

    db.commit()


def get_features(db: Session, item_id: int):
    return db.query(models.ItemFeature).filter(models.ItemFeature.item_id == item_id).all()


def create_feature(db: Session, item_id: int, feature: schemas.Feature):
    feature_id: int = features.get_by_key(db, key=feature.key).id

    item_feature = models.ItemFeature(
        item_id=item_id,
        feature_id=feature_id,
        value=feature.value)

    db.add(item_feature)
    db.commit()
    db.refresh(item_feature)
    return item_feature


def get_events(db: Session, item_id: int):
    return db.query(models.Event).filter(models.Event.item_id == item_id).all()
