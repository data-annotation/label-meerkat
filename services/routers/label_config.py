from fastapi import APIRouter
from sqlalchemy import select

import meerkat as mk
from services.orm.tables import label_config
from services.orm.tables import project
from . import engine

router = APIRouter(
    prefix="/label_configs",
    tags=["label_config"],
    responses={404: {"description": "Not found"}},
)


@router.post("/")
def create_label_config(project_id: int, name: str, config_data: dict):
    """
    add a new label config for a project

    """
    conn = engine.connect()
    conn.execute(label_config.insert(),
                 {"project_id": project_id, "name": name, "extra": config_data})
    return True


@router.get("/")
def list_label_config(project_id: int):
    """
    get all label config of a project

    """
    conn = engine.connect()
    sql = select(label_config.c.id,
                 label_config.c.project_id,
                 label_config.c.extra,
                 label_config.c.name,
                 label_config.c.create_time,
                 label_config.c.update_time).where(label_config.c.project_id == project_id)
    label_config_res = conn.execute(sql)
    res = [{'id': _id,
            'project_id': project_id,
            'extra': extra,
            'name': name,
            'create_time': create_time,
            'update_time': update_time}
           for _id, project_id, extra, name, create_time, update_time in label_config_res]

    return res


@router.get("/{label_config_id}")
def get_single_label_config(label_config_id: int):
    """
    get a label config

    """
    conn = engine.connect()
    sql = select(label_config.c.id,
                 label_config.c.project_id,
                 label_config.c.extra,
                 label_config.c.name,
                 label_config.c.create_time,
                 label_config.c.update_time).where(label_config.c.id == label_config_id)
    label_config_res = conn.execute(sql).fetchone()

    return {'id': label_config_res[0],
            'project_id': label_config_res[1],
            'extra': label_config_res[2],
            'name': label_config_res[3],
            'create_time': label_config_res[4],
            'update_time': label_config_res[5]}



