from enum import Enum
from typing import Any
from typing import List

import uvicorn
from fastapi import APIRouter
from fastapi import FastAPI
from fastapi import Form
from fastapi import Query
from fastapi import Response
from fastapi import UploadFile
from pydantic import BaseModel
from pydantic import Field

router = APIRouter(
    tags=["docs"],
    responses={404: {"description": "Not found"}},
)


class ConfigName(str, Enum):
  tow_sentence = "tow_sentence"
  single_sentence = "single_sentence"


config_mapping = {
  ConfigName.tow_sentence.value: {'columns': ['sentence1', 'sentence2']},
  ConfigName.single_sentence.value: {'columns': ['sentence']}
}


class ProjectCreateResult(BaseModel):
  saved_file: str = Field(description='project 保存的文件子路径')
  project_id: int = Field(description='project id')

@router.post("/projects")
def upload_file_and_process(files: List[UploadFile],
                            response: Response,
                            token: str = Query(None, description="预分配的用户token，用来识别用户身份"),
                            name: str = Query(None, description="project 名称"),
                            processed: bool = Query(True, description="upload data"),
                            config: str = Form(None, description="json文本，project的配置，包含数据的列信息等"),
                            config_name: ConfigName = ConfigName.tow_sentence) -> ProjectCreateResult:
  """create a project"""

  return {'saved_file': 'project/e58c6ee090cb4f698a3c8cbba4564caa.mk',
          'project_id': 2}


class ProjectListResponse(BaseModel):
  id: int = Field(description='project id')
  name: str = Field(description='project name')
  create_time: str = Field(description='project create time')
  update_time: str = Field(description='project update time')
  labels: dict = Field(description='project label list')

@router.get("/projects")
def upload_file_and_process(files: List[UploadFile],
                            response: Response) -> ProjectListResponse:
  """create a project"""

  return [{"id": 1,
           "name": "faire_tale_label",
           "create_time": "2023-03-30T15:24:58",
           "update_time": "2023-03-30T15:24:58",
           "labels": [{"label_id": 1,
                       "user_id": 1,
                       "config": {"label_choice": ["entailment",
                                                   "neutral",
                                                   "contradiction"],
                                  "sentence_column_1": "premise",
                                  "sentence_column_2": "hypothesis"},
                       "current_model": None,
                       "create_time": "2023-03-30 15:24:58",
                       "update_time": "2023-03-30 15:24:58"}]}]




@router.get("projects/{project_id}")
def get_project_data(project_id: int,
                     response: Response,
                     label_id=Query(description="label id")) -> ProjectCreateResult:
  """get project and label data"""

  return {"id": 1,
           "name": "faire_tale_label",
           "create_time": "2023-03-30T15:24:58",
           "update_time": "2023-03-30T15:24:58",
           "labels": [{"label_id": 1,
                       "user_id": 1,
                       "config": {"label_choice": ["entailment",
                                                   "neutral",
                                                   "contradiction"],
                                  "sentence_column_1": "premise",
                                  "sentence_column_2": "hypothesis"},
                       "current_model": None,
                       "create_time": "2023-03-30 15:24:58",
                       "update_time": "2023-03-30 15:24:58"}


@router.get("projects/{project_id}/models")
def get_project_models(response: Response,
                       project_id: int) -> Any:
  """get project models"""

  return {'saved_file': 'project/e58c6ee090cb4f698a3c8cbba4564caa.mk',
          'project_id': 2}


@router.post("projects/{project_id}/training")
def get_project_models(response: Response,
                       project_id: int) -> Any:
  """get project models"""

  return {'saved_file': 'project/e58c6ee090cb4f698a3c8cbba4564caa.mk',
          'project_id': 2}


@router.get("labels/{label_id}/predict")
def get_project_models(response: Response,
                       label_id: int) -> Any:
  """get project models"""

  return {'saved_file': 'project/e58c6ee090cb4f698a3c8cbba4564caa.mk',
          'project_id': 2}


@router.patch("labels/{label_id}")
def get_project_models(response: Response,
                       label_id: int) -> Any:
  """get project models"""

  return {'saved_file': 'project/e58c6ee090cb4f698a3c8cbba4564caa.mk',
          'project_id': 2}


@router.get("labels/{label_id}/unlabeled_data")
def get_project_models(response: Response,
                       label_id: int) -> Any:
  """get project models"""

  return {'saved_file': 'project/e58c6ee090cb4f698a3c8cbba4564caa.mk',
          'project_id': 2}


app = FastAPI(debug=True)
app.include_router(router)

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=9001)
