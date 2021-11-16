import uvicorn

from fastapi import FastAPI

from rexify.app.routers import users, items, models, events

app = FastAPI(title='Rexify')
app.include_router(users.router)
app.include_router(items.router)
app.include_router(events.router)
app.include_router(models.router)

if __name__ == '__main__':
    # noinspection PyTypeChecker
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
