from fastapi import APIRouter
from sqlalchemy import select

import meerkat as mk
from services.orm.anno_project import project
from . import engine

router = APIRouter(
    prefix="/projects",
    tags=["project"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
def project_list():
    """
    list all project

    """
    conn = engine.connect()
    projects = conn.execute(select(project.c.id,
                                   project.c.name,
                                   project.c.create_time,
                                   project.c.update_time))
    res = []
    for p_id, p_name, p_create_time, p_update_time in projects:
        res.append({'id': p_id,
                    'name': p_name,
                    'create_time': p_create_time,
                    'update_time': p_update_time})
    return res


@router.get("/{project_id}")
def single_project(project_id: int):
    """
    get a project data

    """
    conn = engine.connect()
    project_res = conn.execute(select(project.c.id,
                                      project.c.name,
                                      project.c.file_path,
                                      project.c.config).where(project.c.id == project_id)).fetchone()

    df = mk.read(project_res[2])
    config = project_res[3]

    res = df[config['columns']+['id']].to_pandas().to_dict('records')
    return res

