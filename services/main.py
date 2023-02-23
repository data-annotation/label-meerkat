from fastapi import FastAPI
import uvicorn


from .routers import data_process
from .routers import project
from .routers import user
from .routers import label

app = FastAPI(debug=True)

app.include_router(data_process.router)
app.include_router(project.router)
app.include_router(user.router)
app.include_router(label.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)