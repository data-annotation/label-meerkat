import json
import os.path

from fastapi import APIRouter
from sqlalchemy import select

from services.config import predict_result_path
from services.orm.tables import engine
from services.orm.tables import label_result
from services.orm.tables import model_info


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
  get model info by model id

  """

  conn = engine.connect()
  sql = select(model_info.c.id,
               model_info.c.model_uuid,
               model_info.c.label_id,
               model_info.c.extra,
               model_info.c.status,
               model_info.c.iteration).where(model_info.c.id == model_id)

  model_info_record = conn.execute(sql).fetchone()._asdict()
  return model_info_record
