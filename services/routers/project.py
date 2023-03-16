from fastapi import APIRouter
from sqlalchemy import select

import meerkat as mk
from services.orm.anno_project import project
from fastapi import Response
from . import engine
from ..orm.anno_project import label_result

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
def get_single_project(project_id: int, response: Response):
  """
  get a project data

  """
  conn = engine.connect()
  project_res = dict(conn.execute(select(project.c.id,
                                         project.c.name,
                                         project.c.file_path,
                                         project.c.config)
                                  .where(project.c.id == project_id)))
  if not project_res:
    response.status_code = 400
    return 'Project Not Found'
  label_res = dict(conn.execute(select(label_result.c.id,
                                       label_result.c.name,
                                       label_result.c.user_id,
                                       label_result.c.project_id,
                                       label_result.c.config,
                                       label_result.c.create_time,
                                       label_result.c.update_time,
                                       label_result.c.file_path)
                                .where(label_result.c.project_id == project_id)
                                .order_by(label_result.c.create_time)
                                .limit(1).fetchone()))

  df = mk.read(project_res[2])
  config = project_res[3]

  res = df[config['columns'] + ['id']].to_pandas().to_dict('records')
  return res

