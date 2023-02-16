import os.path
import uuid
from typing import Union

from fastapi import APIRouter
from sqlalchemy import select

from services.orm.anno_project import label_result
from . import engine
from pydantic import BaseModel
from meerkat import dataframe as mk_df
from ..config import label_base_path
from ..orm.anno_project import user


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
  id_1_column: str
  id_2_column: str
  sentence_column_1: str
  sentence_column_2: str
  label_choice: list
  explanation_column: str


@router.post("/")
def create_label(project_id: int,
                 name: str,
                 config_data: Union[SingleSentence1, SingleSentence2, SentenceRelation, dict],
                 token: str,
                 label_data: list):
  """
  create a new label for a project

  """
  conn = engine.connect()
  user_res = conn.execute(select(user.c.id).where(user.c.token == token)).fetchone()
  label_res_uuid = uuid.uuid4().hex
  model_uuid = uuid.uuid4().hex
  conn.execute(label_result.insert(),
               {"project_id": project_id,
                "name": name,
                "user_id": user_res[0],
                "file_path": label_res_uuid,
                "config": config_data.dict(),
                "iteration": [{'epoch': 1, 'model': model_uuid}]})
  label_result_df = mk_df.DataFrame(label_data)
  label_result_df.write(os.path.join(label_base_path, f'{label_res_uuid}.mk'))

  return True



@router.post("/trigger")
def one_iteration(project_id: int,
                  name: str,
                  config_data: Union[SingleSentence1, SingleSentence2, SentenceRelation, dict],
                  token: str,
                  label_data: list):
  """
  trigger one iteration

  """
  # conn = engine.connect()
  # user_res = conn.execute(select(user.c.id).where(user.c.token == token)).fetchone()
  # label_res_uuid = uuid.uuid4().hex
  # model_uuid = uuid.uuid4().hex
  # conn.execute(label_result.insert(),
  #              {"project_id": project_id,
  #               "name": name,
  #               "user_id": user_res[0],
  #               "file_path": label_res_uuid,
  #               "config": config_data.dict(),
  #               "iteration": [{'epoch': 1, 'model': model_uuid}]})
  # label_result_df = mk_df.DataFrame(label_data)
  # label_result_df.write(os.path.join(label_base_path, f'{label_res_uuid}.mk'))

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
