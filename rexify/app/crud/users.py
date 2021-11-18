from typing import List

from sqlalchemy.orm import Session

from rexify.app import models, schemas
from rexify.app.crud import features


def get(db: Session, user_id: int):
    user_features = get_features(db, user_id)
    user = db.query(models.User).filter(models.User.id == user_id).first()
    return {
        'id': user.id,
        'features': [{
            features.get(db, feature.id).key: feature.value}
            for feature in user_features
        ]
    }


def get_list(db: Session, skip: int = 0, limit: int = 20):
    return db.query(models.User).offset(skip).limit(limit).all()


def create(db: Session, user: schemas.BaseTarget):
    db_user = models.User(external_id=user.external_id)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    for feature in user.features:
        if bool(features.get_by_key(db, feature.key)):
            create_feature(db, db_user.id, feature)

    return db_user


def get_id(db: Session, external_id: str):
    return db.query(models.User).filter(models.User.external_id == external_id).first()


def update(db: Session, user_id: int, feature_list: List[schemas.Feature]):
    raise NotImplementedError


def delete(db: Session, user_id: int):
    db.query(models.User).filter(models.User.id == user_id).delete()


def get_features(db: Session, user_id: int):
    return db.query(models.UserFeature).filter(models.UserFeature.user_id == user_id).all()


def create_feature(db: Session, user_id: int, feature: schemas.Feature):
    feature_id: int = features.get_by_key(db, key=feature.key).id

    user_feature = models.UserFeature(
        user_id=user_id,
        feature_id=feature_id,
        value=feature.value)

    db.add(user_feature)
    db.commit()
    db.refresh(user_feature)
    return user_feature
