from fastapi import FastAPI
from .routers import data_process

app = FastAPI(debug=True)

app.include_router(data_process.router)

