import os.path
import meerkat as mk
import numpy as np

from fastapi import APIRouter
from fastapi import HTTPException
from sqlalchemy import select

from services.config import project_base_path
from services.config import label_base_path
from services.config import predict_result_path
from services.orm.tables import engine
from services.orm.tables import get_model_by_id
from services.orm.tables import model_info
from services.orm.tables import get_label_by_id
from services.orm.tables import get_models_by_label_id
from services.orm.tables import get_project_by_id
from services.routers import select_model_for_train
from services.model.arch import predict_pipeline

router = APIRouter(
    prefix="/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)



@router.post("/predict/{label_id}")
def predict_unlabeled_data(label_id: int,
                           model_id: int,
                           data_ids: list = None,
                           re_predict: bool = False):
    """
    predict unlabeled data using trained model

    if model id is None use the recent update model to predict

    response:
       like the labeled data result, see /labels/<label_id>

    """

    model_res = get_model_by_id(model_id)
    label_res = get_label_by_id(model_res['label_id'])
    project_res = get_project_by_id(label_res["project_id"])

    if not all([label_res, project_res, model_res]):
        raise HTTPException(status_code=400, detail="Model/Label/Project not found!")

    model_id = model_res['model_uuid']
    if not re_predict:
      last_predict_res = mk.read(os.path.join(predict_result_path,
                                              str(label_id),
                                              f'{model_id}.mk'))
      if data_ids:
        last_predict_res = last_predict_res[last_predict_res['id'].isin(data_ids)]
      predict_res = (last_predict_res
                     .to_pandas()
                     .fillna(np.nan)
                     .replace([np.nan], [None])
                     .to_dict('records'))
    else:
      project_data = mk.read(os.path.join(project_base_path,
                                          f'{project_res["file_path"]}.mk')).to_pandas()
      label_data = mk.read(os.path.join(label_base_path,
                                        f'{label_res["file_path"]}.mk')).to_pandas()
      merge_data = project_data.merge(label_data, how='left', on='id')

      data_columns = project_res['config'].get('columns', [])
      label_columns = label_res['config'].get('columns', [])
      columns = set(data_columns + label_columns)

      unlabeled_data = merge_data[merge_data['label'].isnull()][columns]
      if data_ids:
        unlabeled_data = unlabeled_data[unlabeled_data['id'].isin(data_ids)]
      _, _, predict_res = predict_pipeline(unlabeled_data,
                                           model_id=model_id,
                                           label_id=label_id,
                                           column_1='sentence1',
                                           column_2='sentence2',
                                           explanation_column='sentence2')
      predict_res = predict_res.to_pandas().to_dict('records')
    return predict_res


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

    sql = select(model_info.c.id,
                 model_info.c.model_uuid,
                 model_info.c.label_id,
                 model_info.c.extra,
                 model_info.c.status,
                 model_info.c.iteration).where(model_info.c.id == model_id)
    with engine.connect() as conn:
      model_info_record = conn.execute(sql).fetchone()._asdict()
    return model_info_record
