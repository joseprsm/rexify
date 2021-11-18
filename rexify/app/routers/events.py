from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session

from rexify.app import crud
from rexify.app.db import get_db
from rexify.app.schemas import BaseEvent

import rexify.app.crud.events


router = APIRouter(
    prefix='/events',
    tags=['events'])


@router.get('/')
def get_events(skip: Optional[int] = 0, limit: Optional[int] = 10, db: Session = Depends(get_db)):
    return crud.events.get_list(db, skip, limit)


@router.post('/')
def create_event(event: BaseEvent, db: Session = Depends(get_db)):
    return crud.events.create(db, event)


@router.get('/{event_id}')
def get_event(event_id: int, db: Session = Depends(get_db)):
    event = crud.events.get(db, event_id=event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="User not found")
    return event


@router.put('/{event_id}')
def update_event(*, event_id: int):
    return {'msg': f'Event {event_id} updated'}


@router.delete('/{event_id}')
def delete_event(event_id: int, db: Session = Depends(get_db)):
    return crud.events.delete(db, event_id=event_id)
