from fastapi import FastAPI
from .routers import data_process
from .routers import project
from .routers import user

app = FastAPI(debug=True)

app.include_router(data_process.router)
app.include_router(project.router)
app.include_router(user.router)

