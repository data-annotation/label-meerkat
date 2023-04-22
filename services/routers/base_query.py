import json
import os.path
import meerkat as mk
import numpy as np

from fastapi import APIRouter
from fastapi import HTTPException
from sqlalchemy import select

from services.config import predict_result_path
from services.config import project_base_path
from services.config import label_base_path
from services.orm.tables import engine
from services.orm.tables import label_result
from services.orm.tables import model_info

from services.orm.tables import get_label_by_id
from services.orm.tables import get_labels_by_project_id
from services.orm.tables import get_models_by_label_id
from services.orm.tables import get_project_by_id
from services.orm.tables import get_single_project_label
from services.routers import select_model_for_train
from services.model.arch import predict_pipeline

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

    # conn = engine.connect()
    # sql = select(label_result.c.id,
    #              label_result.c.current_model,
    #              label_result.c.project_id,
    #              label_result.c.file_path).where(label_result.c.id == label_id)

    # label_record = conn.execute(sql).fetchone()

    label_res = get_label_by_id(label_id)
    project_res = get_project_by_id(label_res["project_id"])
    model_res = get_models_by_label_id(label_id=label_id)
    if not label_res:
        raise HTTPException(status_code=400, detail="Model not found!")

    selected_model, model_num = select_model_for_train(model_res)
    model_id = selected_model['model_uuid']
    project_data = mk.read(os.path.join(project_base_path,
                                        f'{project_res["file_path"]}.mk')).to_pandas()
    label_data = mk.read(os.path.join(label_base_path,
                                      f'{label_res["file_path"]}.mk')).to_pandas()
    merge_data = project_data.merge(label_data, how='left', on='id')

    # project_with_label = project_data.join(label_data.set_index('id'), on='id')
    data_columns = project_res['config'].get('columns', [])
    label_columns = label_res['config'].get('columns', [])
    columns = set(data_columns + label_columns)

    unlabeled_data = merge_data[merge_data['label'].isnull()][columns]
    predicted_rationale, predicted_label = predict_pipeline(unlabeled_data, model_id=model_id, label_id=label_id,                                                         column_1='sentence1', column_2='sentence2', explanation_column='sentence2')
    pre_dict = {}
    pre_dict['explanation'] = predicted_rationale
    pre_dict['label'] = predicted_label
    return pre_dict


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
