import json
import os.path
import uuid
from typing import Union

import pandas as pd
from fastapi import APIRouter
from fastapi import Body
from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import select

import meerkat as mk
from services.config import label_base_path
from services.config import model_path
from services.config import predict_result_path
from services.config import project_base_path
from services.model.AL import one_training_iteration
from services.model.arch import predict_pipeline
from services.orm.anno_project import label_result
from services.orm.anno_project import project
from services.orm.anno_project import user
from services.routers import engine
from fastapi import BackgroundTasks


router = APIRouter(
    prefix="/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)


@router.get("/predict/{label_id}")
def list_label_result_of_a_project(label_id: int):
  """
  get label result list for a project

  """

  conn = engine.connect()
  sql = select(label_result.c.id,
               label_result.c.current_model).where(label_result.c.id == label_id)

  label_record = conn.execute(sql).fetchone()
  result_path = os.path.join(predict_result_path, str(label_id))
  with open(result_path+f'/{label_record[1]}.json', 'r') as f:
    res = json.load(f)
  return res



@router.get("/models/{model_id}/status")
def get_model_status(model_id: int):
  """
  get label result list for a project

  """

  conn = engine.connect()
  sql = select(label_result.c.id,
               label_result.c.current_model).where(label_result.c.id == label_id)

  label_record = conn.execute(sql).fetchone()
  result_path = os.path.join(predict_result_path, str(label_id))
  with open(result_path+f'/{label_record[1]}.json', 'r') as f:
    res = json.load(f)
  return res