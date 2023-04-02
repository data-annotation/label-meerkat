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
from services.config import project_base_path
from services.model.AL import one_training_iteration
from services.model.arch import predict_pipeline
from services.orm.tables import label_result
from services.orm.tables import model_info
from services.orm.tables import project
from services.orm.tables import user
from services.routers import engine
from fastapi import BackgroundTasks

router = APIRouter(
    prefix="/labels",
    tags=["label"],
    responses={404: {"description": "Not found"}},
)


"""
single sentence
{
  'sentence_column': 'abc',
  'id_column': 'id',
  'explanation_column': 'explanation'
  'label_choice': ['rose', 'jack']
}

{
  'sentence_column': 'abc',
  'id_column': 'id',
  'explanation_column': 'explanation'
  'label_choice': {'type1': ['ture', 'false', 'unknown'],
                   'type2': ['ture', 'false', 'unknown']}
}

sentence relation
{ 
  'id_1_column': 'id_1',
  'id_2_column': 'id_2',
  'sentence_column_1': 'column_name_1',
  'sentence_column_2': 'column_name_2'
  'label_choice': ['rose', 'jack'],
  'explanation_column': 'explanation'
}

{ 
  'column_1': 'column_name_1',
  'label_column': 'label'
  'label_choice': ['entailment', 'neutral', 'contradiction'],
  'explanation_column': 'explanation_1'
}

  { 
    "label_column": "label",
    "label_choice": ["entailment", "neutral", "contradiction"],
    "explanation_column": "explanation_1"
  }


"""


class SingleSentence1(BaseModel):
  config_type: str = 'single_sentence_1'
  label_column: str
  label_choice: list
  explanation_column: str
  id_column: str = 'id'


class SingleSentence2(SingleSentence1):
  config_type: str = 'single_sentence_1'
  label_choice: dict


class SentenceRelation(BaseModel):
  config_type: str = 'sentence_relation'
  label_column: str
  label_choice: Union[str, list]
  explanation_column: str


class UpdateLabel(BaseModel):
  label_data: Union[dict, list]


# @router.post("/")
def create_label_with_train(project_id: int,
                            name: str,
                            config_data: Union[SingleSentence1, SingleSentence2, SentenceRelation, dict],
                            token: str,
                            label_data: Union[list, dict],
                            background_tasks: BackgroundTasks):
  """
  create a new label for a project

  """
  conn = engine.connect()
  project_res = conn.execute(select(project.c.id,
                                    project.c.name,
                                    project.c.file_path,
                                    project.c.config).where(project.c.id == project_id)).fetchone()
  project_data = mk.read(os.path.join(project_base_path, f'{project_res[2]}.mk')).to_pandas()
  user_res = conn.execute(select(user.c.id).where(user.c.token == token)).fetchone()
  model_uuid = uuid.uuid4().hex
  label_data = pd.DataFrame(label_data)

  project_with_label = project_data.join(label_data.set_index('id'), on='id')

  background_tasks.add_task(one_training_iteration,
                            labeled_data=project_with_label[project_with_label['label'].notnull()][['premise',
                                                                                                    'hypothesis',
                                                                                                    'label',
                                                                                                    'explanation_1']],
                            model_id=model_uuid)

  label_uuid = uuid.uuid4().hex
  label_extra = {'labeled_num': len(label_data),
                 'dataset_num': len(project_data)}
  conn.execute(label_result
               .insert()
               .values({"project_id": project_id,
                        "name": name,
                        "extra": label_extra,
                        "user_id": user_res[0],
                        "config": config_data.dict(),
                        "current_model": model_uuid,
                        "file_path": label_uuid}))

  label_result_df = mk.from_pandas(label_data, index=False)
  label_result_df.write(os.path.join(label_base_path, f'{label_uuid}.mk'))

  no_label_data = (project_with_label[project_with_label['label'].isnull()]
                   [['premise', 'hypothesis', 'label', 'explanation_1']])
  print('End training')

  new_label_result = conn.execute(select(label_result.c.id).where(label_result.c.current_model == model_uuid)).fetchone()

  background_tasks.add_task(predict_pipeline,
                            data_predict=no_label_data,
                            model_id=model_uuid,
                            label_id=new_label_result[0])

  return {'label_id': new_label_result[0],
          'model_id': model_uuid}


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
               label_result.c.create_time,
               label_result.c.update_time,
               label_result.c.file_path).where(label_result.c.id == label_result_id)

  res = conn.execute(sql).fetchone()
  label_full = mk.read(os.path.join(label_base_path, f'{res[7]}.mk')).to_pandas().to_dict()

  return {'id': res[0],
          'name': res[1],
          'user_id': res[2],
          'project_id': res[3],
          'config': res[4],
          'create_time': res[5],
          'update_time': res[6],
          'labels': label_full}


@router.patch("/{label_result_id}")
def update_label_result(label_result_id: int,
                        label_data: dict = Body(embed=True)):
    """
    update a label result

    """
    conn = engine.connect()
    sql = select(label_result.c.id,
                 label_result.c.user_id,
                 label_result.c.project_id,
                 label_result.c.file_path,
                 label_result.c.last_model,
                 label_result.c.current_model,
                 label_result.c.iteration,
                 label_result.c.config).where(label_result.c.id == label_result_id)

    label_res: dict = conn.execute(sql).fetchone()._asdict()
    if not label_res:
        raise HTTPException(status_code=400, detail="Labels can not found!")

    label_mk_df = mk.read(os.path.join(label_base_path, f"{label_res['file_path']}.mk"))
    label_data = pd.DataFrame(label_data)

    label_full = pd.concat([label_mk_df.to_pandas(index=False), label_data])

    label_full_mk = mk.from_pandas(label_full, index=False)
    label_full_mk.write(os.path.join(label_base_path, f"{label_res['file_path']}.mk"))

    return {'label_id': label_result_id}


@router.get("/{label_result_id}/state")
def get_label_state(label_result_id: int):
  """
  get training or predict state after create or update a project label

  """
  sql = select(label_result.c.id,
               label_result.c.extra).where(label_result.c.id == label_result_id)

  res = engine.connect().execute(sql).fetchone()
  if not res:
    raise HTTPException(status_code=400, detail="Labels can not found!")

  return {'state': res[1]}


@router.patch("/{label_id}")
def only_update_label(label_id: int,
                      label_data: dict = Body(embed=True)):
  """
  update a label result

  """
  coon = engine.connect()
  label_res = dict(coon.execute(select(label_result.c.id,
                                       label_result.c.name,
                                       label_result.c.user_id,
                                       label_result.c.project_id,
                                       label_result.c.config,
                                       label_result.c.extra,
                                       label_result.c.create_time,
                                       label_result.c.update_time,
                                       label_result.c.file_path)
                                .where(label_result.c.label_id == label_id)
                                .order_by(label_result.c.create_time)
                                .limit(1))
                   .fetchone())

  if not label_res:
    raise HTTPException(status_code=400,
                        detail="Labels can not found!")

  new_model_id = uuid.uuid4().hex
  label_mk_df = mk.read(os.path.join(label_base_path, f"{label_res['file_path']}.mk"))
  label_data = pd.DataFrame(label_data)

  label_full = pd.concat([label_mk_df.to_pandas(), label_data])
  label_full_mk = mk.from_pandas(label_full, index=False)
  label_full_mk.write(os.path.join(label_base_path, f"{label_res['file_path']}.mk"))

  sql = (label_result
         .update()
         .where(label_result.c.id == label_res['id'])
         .values({'last_model': label_result.c.current_model,
                  'current_model': new_model_id}))
  coon.execute(sql)

  return {'label_id': label_res['id'],
          'model_id': new_model_id}


@router.patch("/{label_id}/models")
def only_update_label(label_id: int):
  """
  update a label result

  """
  coon = engine.connect()
  models = dict(coon.execute(select(model_info.c.id,
                                    model_info.c.name,
                                    model_info.c.model_path,
                                    model_info.c.extra,
                                    model_info.c.last_model,
                                    model_info.c.create_time,
                                    model_info.c.update_time,

                                    model_info.c.file_path)
                             .where(model_info.c.label_id == label_id)
                             .order_by(label_result.c.create_time)))

  if not label_res:
    raise HTTPException(status_code=400,
                        detail="Labels can not found!")

  new_model_id = uuid.uuid4().hex
  label_mk_df = mk.read(os.path.join(label_base_path, f"{label_res['file_path']}.mk"))
  label_data = pd.DataFrame(label_data)

  label_full = pd.concat([label_mk_df.to_pandas(), label_data])
  label_full_mk = mk.from_pandas(label_full, index=False)
  label_full_mk.write(os.path.join(label_base_path, f"{label_res['file_path']}.mk"))

  sql = (label_result
         .update()
         .where(label_result.c.id == label_res['id'])
         .values({'last_model': label_result.c.current_model,
                  'current_model': new_model_id}))
  coon.execute(sql)

  return {'label_id': label_res['id'],
          'model_id': new_model_id}
