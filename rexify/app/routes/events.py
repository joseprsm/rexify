from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

EVENTS = []

router = APIRouter()


class Event(BaseModel):

    id: Optional[int]
    user_id: int
    item_id: int


class Events(BaseModel):

    events: List[Event]


@router.get('/events')
def get_events():
    return {'events': EVENTS}


@router.post('/events')
def create_events(*, event: Event):
    new_entry_id = len(EVENTS) + 1
    event_entry = Event(
        id=new_entry_id,
        user_id=event.user_id,
        item_id=event.item_id)
    EVENTS.append(event_entry.dict())
    return event_entry


@router.get('/events/{event_id}')
def get_event(*, event_id: int):
    result = [recipe for recipe in EVENTS if recipe["id"] == event_id]
    if result:
        return result[0]


@router.put('/events/{event_id}')
def update_event(*, event_id: int):
    return {'msg': f'Event {event_id} updated'}


@router.delete('/events/{event_id}')
def delete_event(*, event_id: int):
    return {'msg': f'Event {event_id} deleted'}

