from typing import List

from sqlalchemy.orm import Session

from rexify.app import models, schemas


def get(db: Session, feature_id: int):
    return db.query(models.Feature).filter(models.Feature.id == feature_id).first()


def get_by_key(db: Session, key: str):
    return db.query(models.Feature).filter(models.Feature.key == key).first()


def get_list(db: Session, skip: int = 0, limit: int = 20):
    return db.query(models.Feature).offset(skip).limit(limit).all()


def create(db: Session, feature: schemas.BaseFeature):
    db_feature = models.Feature(key=feature.key)
    db.add(db_feature)
    db.commit()
    db.refresh(db_feature)
    return db_feature


def update(db: Session, user_id: int, feature_list: List[schemas.Feature]):
    raise NotImplementedError


def delete(db: Session, feature_id: int):
    db.query(models.Feature).filter(models.Feature.id == feature_id).delete()
