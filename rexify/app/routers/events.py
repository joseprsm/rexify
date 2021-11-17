from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from rexify.app import crud
from rexify.app.db import get_db
from rexify.app.schemas import Event, BaseEvent

router = APIRouter(
    prefix='/events',
    tags=['events'])


@router.get('/')
def get_events(skip: Optional[int] = 0, limit: Optional[int] = 10, db: Session = Depends(get_db)):
    return crud.get_events(db, skip, limit)


@router.post('/')
def create_event(event: BaseEvent, db: Session = Depends(get_db)):
    return crud.create_event(db, event)


@router.get('/{event_id}')
def get_event(*, event_id: int):
    result = [recipe for recipe in EVENTS if recipe["id"] == event_id]
    if result:
        return result[0]


@router.put('/{event_id}')
def update_event(*, event_id: int):
    return {'msg': f'Event {event_id} updated'}


@router.delete('/{event_id}')
def delete_event(*, event_id: int):
    return {'msg': f'Event {event_id} deleted'}

