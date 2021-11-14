import uvicorn

from fastapi import FastAPI

from rexify.app.routes.users import router as user_router
from rexify.app.routes.items import router as item_router
from rexify.app.routes.events import router as event_router

app = FastAPI(title='Rexify')
app.include_router(user_router)
app.include_router(item_router)
app.include_router(event_router)

if __name__ == '__main__':
    # noinspection PyTypeChecker
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
