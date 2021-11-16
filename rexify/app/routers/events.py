from fastapi import APIRouter

from rexify.app.schemas import Event

EVENTS = []

router = APIRouter(
    prefix='/events',
    tags=['events']
)


@router.get('/')
def get_events():
    return {'events': EVENTS}


@router.post('/')
def create_events(*, event: Event):
    new_entry_id = len(EVENTS) + 1
    event_entry = Event(
        id=new_entry_id,
        user_id=event.user_id,
        item_id=event.item_id)
    EVENTS.append(event_entry.dict())
    return event_entry


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

