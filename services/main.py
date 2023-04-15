import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from services.frontend import projects
from services.routers import base_query
from services.routers import data_process
from services.routers import label
from services.routers import project
from services.routers import user

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_process.router)
app.include_router(project.router)
app.include_router(user.router)
app.include_router(label.router)
app.include_router(base_query.router)

# projects.init(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
