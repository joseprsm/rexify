from typing import Union, Type

from sqlalchemy.orm import Session

from . import models, schemas


def _get_single_target(db: Session, model_obj, id_: int):
    return db.query(model_obj).filter(model_obj.id == id_).first()


def _get_target_list(db: Session, model_obj, skip: int = 0, limit: int = 100):
    return db.query(model_obj).offset(skip).limit(limit).all()


def _get_by_external_id(db: Session,
                        external_id: str,
                        model_obj: Union[Type[models.User], Type[models.Item]]):
    return db.query(model_obj).filter(model_obj.external_id == external_id).first()


def get_user(db: Session, user_id: int):
    return _get_single_target(db, models.User, user_id)


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return _get_target_list(db, models.User, skip, limit)


def create_user(db: Session, user: schemas.BaseTarget):
    db_user = models.User(external_id=user.external_id)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    for feature in user.features:
        if bool(get_feature_by_key(db, feature.key)):
            create_user_feature(db, db_user.id, feature)

    return db_user


def get_user_id(db: Session, external_id: str):
    return _get_by_external_id(db, external_id, models.User)


def get_item(db: Session, item_id: int):
    return _get_single_target(db, models.Item, item_id)


def get_items(db: Session, skip: int = 0, limit: int = 100):
    return _get_target_list(db, models.Item, skip, limit)


def create_item(db: Session, item: schemas.BaseTarget):
    db_item = models.Item(external_id=item.external_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)

    for feature in item.features:
        if bool(get_feature_by_key(db, feature.key)):
            create_item_feature(db, db_item.id, feature)

    return db_item


def get_item_id(db: Session, external_id: str):
    return _get_by_external_id(db, external_id, models.Item)


def get_event(db: Session, event_id: int):
    return _get_single_target(db, models.Event, event_id)


def get_events(db: Session, skip: int = 0, limit: int = 100):
    return _get_target_list(db, models.Event, skip, limit)


def get_feature_by_key(db: Session, key: str):
    return db.query(models.Feature).filter(models.Feature.key == key).first()


def get_feature(db: Session, feature_id: int):
    return _get_single_target(db, models.Feature, feature_id)


def get_features(db: Session, skip: int = 0, limit: int = 100):
    return _get_target_list(db, models.Feature, skip, limit)


def create_feature(db: Session, feature: schemas.BaseFeature):
    db_feature = models.Feature(key=feature.key)
    db.add(db_feature)
    db.commit()
    db.refresh(db_feature)
    return db_feature


def create_user_feature(db: Session, user_id: int, feature: schemas.Feature):
    feature_id = get_feature_by_key(db, key=feature.key).id
    user_feature = models.UserFeature(
        user_id=user_id,
        feature_id=feature_id,
        value=feature.value)
    db.add(user_feature)
    db.commit()
    db.refresh(user_feature)
    return user_feature


def get_user_features(db: Session, user_id: int):
    return db.query(models.UserFeature).filter(models.UserFeature.user_id == user_id).all()


def create_item_feature(db: Session, item_id: int, feature: schemas.Feature):
    feature_id = get_feature_by_key(db, key=feature.key).id
    item_feature = models.ItemFeature(
        item_id=item_id,
        feature_id=feature_id,
        value=feature.value)
    db.add(item_feature)
    db.commit()
    db.refresh(item_feature)
    return item_feature


def get_item_features(db: Session, item_id: int):
    return db.query(models.ItemFeature).filter(models.ItemFeature.item_id == item_id).all()


