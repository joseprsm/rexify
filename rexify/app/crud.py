from sqlalchemy.orm import Session

from . import models, schemas


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def get_item(db: Session, item_id: int):
    return db.query(models.Item).filter(models.Item.id == item_id).first()


def get_items(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Item).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.User):
    db_user = models.User(external_id=user.external_id)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return user


def get_user_by_external_id(db: Session, external_id: str):
    return db.query(models.User).filter(models.User.external_id == external_id).first()
