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
def predict_unlabeled_data(label_id: int, model_id: int = None):
  f"""
  predict unlabeled data using trained model

  if model id is None use the recent update model to predict

  response:
     like the labeled data result, see /labels/<label_id>

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

  response:
  status, int: 0 空闲状态，1 training
  iteration, int:迭代训练次数，触发过多少次训练
  extra, object: 初始时为空{}
    train_begin, bool: True
    train_end, bool: False
    total_steps, int: 总训练步骤数
    current_step, int: 当前为所处训练步骤数
    progress, str: 1/10 所处步骤训练进度
    begin_time, float: timestamp

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
