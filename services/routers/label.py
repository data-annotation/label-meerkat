from fastapi import APIRouter
from sqlalchemy import select

from services.orm.anno_project import label_result
from . import engine

router = APIRouter(
    prefix="/labels",
    tags=["label"],
    responses={404: {"description": "Not found"}},
)


@router.post("/")
def create_label(project_id: int, name: str, config_data: dict, label_data: dict):
  """
  create a new label for a project

  """
  conn = engine.connect()
  conn.execute(label_result.insert(),
               {"project_id": project_id,
                "name": name,
                "config": config_data,
                "iteration": label_data})
  return True


@router.get("/project/{project_id}")
def list_label_result_of_a_project(project_id: int):
  """
  get label result list for a project

  """
  conn = engine.connect()
  sql = select(label_result.c.id,
               label_result.c.name,
               label_result.c.user_id,
               label_result.c.project_id,
               label_result.c.config,
               label_result.c.create_time,
               label_result.c.update_time).where(label_result.c.project_id == project_id)

  label_result_list = conn.execute(sql)
  res = [{'id': _id,
          'name': name,
          'user_id': user_id,
          'project_id': project_id,
          'config': config,
          'create_time': create_time,
          'update_time': update_time}
         for _id, name, user_id, project_id, config, create_time, update_time in label_result_list]
  return res


@router.get("/{label_result_id}")
def get_label_result(label_result_id: int):
  """
  get label result

  """
  conn = engine.connect()
  sql = select(label_result.c.id,
               label_result.c.name,
               label_result.c.user_id,
               label_result.c.project_id,
               label_result.c.config,
               label_result.c.iteration,
               label_result.c.create_time,
               label_result.c.update_time).where(label_result.c.id == label_result_id)

  res = conn.execute(sql).fetchone()

  return {'id': res[0],
          'name': res[1],
          'user_id': res[2],
          'project_id': res[3],
          'config': res[4],
          'create_time': res[5],
          'update_time': res[6]}


@router.patch("/{label_result_id}")
def update_label_result(label_result_id: int):
  """
  update a label result

  """
  conn = engine.connect()
  sql = select(label_result.c.id,
               label_result.c.name,
               label_result.c.user_id,
               label_result.c.project_id,
               label_result.c.config,
               label_result.c.iteration,
               label_result.c.create_time,
               label_result.c.update_time).where(label_result.c.id == label_result_id)

  res = conn.execute(sql).fetchone()

  return {'id': res[0],
          'name': res[1],
          'user_id': res[2],
          'project_id': res[3],
          'config': res[4],
          'create_time': res[5],
          'update_time': res[6]}
