from fastapi import FastAPI


from .routers import data_process
from .routers import project
from .routers import user
from .routers import label

app = FastAPI(debug=True)

app.include_router(data_process.router)
app.include_router(project.router)
app.include_router(user.router)
app.include_router(label.router)


