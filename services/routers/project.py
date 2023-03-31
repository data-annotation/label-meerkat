import io
import json
import os
import uuid
import zipfile
from typing import List
from typing import Union

from fastapi import BackgroundTasks

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi import Body
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy import select
from sentence_splitter import SentenceSplitter

import meerkat as mk
from services.orm.anno_project import project
from fastapi import Response
from . import engine
from services.config import label_base_path
from services.config import project_base_path
from services.orm.anno_project import label_result
from ..model.AL import one_training_iteration
from pydantic import BaseModel

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
    j = project.join(label_result, project.c.id == label_result.c.project_id)
    projects = conn.execute(select(project.c.id,
                                   project.c.name,
                                   project.c.create_time,
                                   project.c.update_time,
                                   func.json_group_array(
                                       func.json_object(
                                           'label_id', label_result.c.id,
                                           'user_id', label_result.c.user_id,
                                           'config', label_result.c.config,
                                           'current_model', label_result.c.current_model,
                                           'create_time', label_result.c.create_time,
                                           'update_time', label_result.c.update_time
                                       )
                                   ).label('labels'))
                            .select_from(j).group_by(project.c.id)).mappings().all()
    return projects


def get_project_and_label_by_project_id(project_id: int,
                                        label_id: int = None,
                                        coon=None):
  """
  通过project_id获取 project 和 label 信息
  """
  conn = coon or engine.connect()
  project_res = dict(conn.execute(select(project.c.id,
                                         project.c.name,
                                         project.c.file_path,
                                         project.c.config,
                                         project.c.create_time,
                                         project.c.update_time)
                                  .where(project.c.id == project_id))
                     .fetchone())
  if not project_res:
    return None, None

  # 当亲总是获取初始的一个label
  cond = [label_result.c.project_id == project_id]
  if label_id:
    cond.append(label_result.c.id == label_id)
  label_res = dict(conn.execute(select(label_result.c.id,
                                       label_result.c.name,
                                       label_result.c.user_id,
                                       label_result.c.project_id,
                                       label_result.c.config,
                                       label_result.c.extra,
                                       label_result.c.last_model,
                                       label_result.c.current_model,
                                       label_result.c.create_time,
                                       label_result.c.update_time,
                                       label_result.c.file_path)
                                .where(and_(label_result.c.project_id == project_id))
                                .order_by(label_result.c.create_time)
                                .limit(1))
                   .fetchone())
  return project_res, label_res


@router.get("/{project_id}")
def get_single_project(project_id: int,
                       response: Response,
                       label_id: int = None,
                       size: int = 1000,
                       num: int = 0,
                       with_data: bool = False,
                       with_label: bool = False):
  """
  get a project data

  """
  project_res, label_res = get_project_and_label_by_project_id(project_id, label_id=label_id)
  if not project_res:
    response.status_code = 400
    return 'Project Not Found'

  project_data = mk.read(os.path.join(project_base_path, f"{project_res['file_path']}.mk")).to_pandas()
  total_num = len(project_data)

  res = {'project_meta': project_res,
         'label_meta': label_res,
         'data_num': total_num,
         'label_num': 0}
  if with_label and label_res:
    label_data = mk.read(os.path.join(label_base_path, f"{label_res['file_path']}.mk")).to_pandas()
    label_column = label_res['config']['label_column']
    label_data[label_column] = label_data[label_column].astype(int)
    merged_data = project_data.iloc[size*num:size*(num+1)].join(label_data.set_index('id'), on='id')
    merged_data = merged_data.fillna(np.nan).replace([np.nan], [None])
    res['data'] = merged_data.to_dict('records')
    res['label_num'] = len(label_data)
  elif with_data:
    res['data'] = project_data.iloc[size*num:size*(num+1)].to_dict('records')

  return res

class model_id_response(BaseModel):
  new_model_id:int = uuid.uuid4().hex
@router.post("/{project_id}/training", response_model=model_id_response)
def trigger_project_train(project_id: int,
                          background_tasks: BackgroundTasks,
                          response: Response,
                          label_id: int = None):
  """
  get a project data

  """
  project_res, label_res = get_project_and_label_by_project_id(project_id, label_id=label_id)
  if not project_res:
    response.status_code = 400
    return 'Project Not Found'

  project_data = mk.read(os.path.join(project_base_path,
                                      f'{project_res["file_path"]}.mk')).to_pandas()
  label_data = mk.read(os.path.join(label_base_path,
                                    f'{label_res["file_path"]}.mk')).to_pandas()

  current_model = label_res['current_model']
  new_model_id = uuid.uuid4().hex
  project_with_label = project_data.join(label_data.set_index('id'), on='id')
  columns = project_res['config'].get('columns', [])
  columns += label_res['config'].get('columns', [])
  all_labeled_project_data = project_with_label[project_with_label['label'].notnull()][['premise',
                                                                                        'hypothesis',
                                                                                        'label',
                                                                                        'explanation_1']]

  background_tasks.add_task(one_training_iteration,
                            labeled_data=all_labeled_project_data,
                            model_id=new_model_id,
                            old_model_id=current_model)

  return {"new_model_id": new_model_id}

import datetime
class project_label_result(BaseModel):
  id:int = 1
  name:str = 'jack'
  user_id:int = 1
  project_id:int = 1
  config: dict = {'label_choice': ['entailment', 'neutral', 'contradiction'],
                  'sentence_column_1': 'premise',
                  'sentence_column_2': 'hypothesis'}
  create_time:datetime.datetime = datetime.datetime.now()
  update_time:datetime.datetime = datetime.datetime.now()

@router.get("/labels/{project_id}",response_model=List[project_label_result])
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

  label_result_list = conn.execute(sql).mappings().all()

  return label_result_list

# '''
#   get model list of a project
# '''
# class model_list_project_response(BaseModel):
#   model_path:str = "model_data/model1/"
#   model_id:int = uuid.uuid4().hex

# @router.get("/{project_id}/models")
# def model_list_project(project_id:int, response_model=model_list_project_response):
#   return {"model_path": "model_data/model1/",
#           "model_id":uuid.uuid4().hex}



