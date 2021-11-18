from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session

from rexify.app import crud
from rexify.app.db import get_db
from rexify.app.schemas import BaseFeature

import rexify.app.crud.features


router = APIRouter(
    prefix='/features',
    tags=['features'])


@router.get('/')
def get_features(skip: Optional[int] = 0, limit: Optional[int] = 10, db: Session = Depends(get_db)):
    return crud.features.get_list(db, skip, limit)


@router.post('/')
def create_feature(feature: BaseFeature, db: Session = Depends(get_db)):
    db_feature = crud.features.get_by_key(db, feature.key)
    if db_feature:
        raise HTTPException(status_code=400, detail="Feature already registered")
    return crud.features.create(db, feature)


@router.get('/{feature_id}')
def get_feature(feature_id: int, db: Session = Depends(get_db)):
    feature = crud.features.get(db, feature_id=feature_id)
    if feature is None:
        raise HTTPException(status_code=404, detail="Feature not found")
    return feature


@router.delete('/{feature_id}')
def delete_feature(feature_id: int, db: Session = Depends(get_db)):
    return crud.features.delete(db, feature_id=feature_id)


@router.delete('/{feature_id}')
def update_feature(feature_id: int):
    return {'msg': f'Feature {feature_id} updated.'}
