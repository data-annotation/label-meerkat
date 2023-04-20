from fastapi import APIRouter
from sqlalchemy import select

from services.orm.tables import engine
from services.orm.tables import model_info


router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={404: {"description": "Not found"}},
)


@router.get("/{model_id}")
def get_model_info(model_id: int):
    """
    get model info by model id

    response:
    status: 0 空闲状态，1 training
    iteration:迭代训练次数，触发过多少次训练
    extra:
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
                 model_info.c.iteration,
                 model_info.c.create_time,
                 model_info.c.update_time).where(model_info.c.id == model_id)

    model_info_record = conn.execute(sql).fetchone()._asdict()
    return model_info_record


@router.get("/by_label/{label_id}")
def get_models_info_by_label_id(label_id: int):
    """
    get model list by label id

    """

    conn = engine.connect()
    sql = select(model_info.c.id,
                 model_info.c.model_uuid,
                 model_info.c.label_id,
                 model_info.c.extra,
                 model_info.c.iteration,
                 model_info.c.create_time,
                 model_info.c.update_time).where(model_info.c.label_id == label_id)

    model_info_records = conn.execute(sql).mappings().all()
    return model_info_records
