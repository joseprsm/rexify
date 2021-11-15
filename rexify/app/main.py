import uvicorn

from fastapi import FastAPI

from rexify.app.routes import user_router, item_router, event_router, model_router

app = FastAPI(title='Rexify')
app.include_router(user_router)
app.include_router(item_router)
app.include_router(event_router)
app.include_router(model_router)

if __name__ == '__main__':
    # noinspection PyTypeChecker
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
