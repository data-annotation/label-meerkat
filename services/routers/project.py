import io
import json
import os
import uuid
import zipfile
from enum import Enum
from typing import List
from typing import Union

import meerkat as mk
import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi import BackgroundTasks
from fastapi import Form
from fastapi import Response
from fastapi import UploadFile
from sentence_splitter import SentenceSplitter
from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy import select

from services.config import label_base_path
from services.config import project_base_path
from services.orm.tables import label_result
from services.orm.tables import project
from . import encode_model
from . import engine
from ..model.AL import one_training_iteration
from ..orm.tables import get_label_by_id
from ..orm.tables import get_labels_by_project_id
from ..orm.tables import get_project_by_id
from ..orm.tables import model_info
from ..orm.tables import user

router = APIRouter(
    prefix="/projects",
    tags=["project"],
    responses={404: {"description": "Not found"}},
)


class ConfigName(str, Enum):
    esnil = 'esnil'


config_mapping = {
    ConfigName.esnil.value: {'columns': ['sentence1', 'sentence2', 'id'],
                             'default_label_config': {'choice': ['entailment',
                                                                 'neutral',
                                                                 'contradiction'],
                                                      'columns': ['label', 'id', 'explanation'],
                                                      'label_column': 'label',
                                                      'label_data_type': 'index',
                                                      'id_column': 'id'}}
}

splitter = SentenceSplitter(language='en')

def text_to_sentence(text: Union[str, bytes], name: str = None):
    text = text.decode() if isinstance(text, bytes) else text
    paragraphs = [i.strip('\n\t').strip()
                  for i in text.split('\n')
                  if i.strip('\n\t').strip() != '']
    res = []
    for p_index, paragraph in enumerate(paragraphs):
        res.extend([{'paragraph': p_index,
                     'index': idx,
                     'sentence': sentence,
                     **({'name': name}
                        if name else {})}
                    for idx, sentence in enumerate(splitter.split(paragraph))])
    return res


def dataframe_process(df: pd.DataFrame):
    res = []
    for _, row in df.iterrows():
        res.extend(text_to_sentence(row.content, row.title))
    return res


def zip_process(file: UploadFile):
    zfile = zipfile.ZipFile(io.BytesIO(file.file.read()))
    zfile.extractall('zipf/')
    filepath = 'zipf/'
    return filepath_process(filepath)


def filepath_process(filepath):
    for item in os.listdir(filepath):
        if os.path.isdir(item):
            filepath_process(filepath + '/' + item)
        elif item.endswith('txt'):
            with open(filepath + item, 'r') as f:
                res = text_to_sentence(f.read(), item[:-5])
        elif item.endswith('csv'):
            with open(filepath + item, 'rb') as f:
                res = dataframe_process(pd.read_csv(f))
        elif item.endswith('xlsx'):
            with open(filepath + item, 'r') as f:
                res = dataframe_process(pd.read_excel(f.read()))
        elif item.endswith('json'):
            with open(filepath + item, 'r') as f:
                for text in json.load(f):
                    dataframe_process(text['content', text['title']])
    return res


def calc_cosine_distance(a: np.array, b: np.array):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@router.get("")
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
                                           'config', func.json(label_result.c.config),
                                           'current_model', label_result.c.current_model,
                                           'create_time', label_result.c.create_time,
                                           'update_time', label_result.c.update_time
                                       )
                                   ).label('labels'))
                            .select_from(j).group_by(project.c.id)).fetchall()
    res = []
    for p in projects:
        p = p._asdict()
        p['labels'] = json.loads(p['labels'])
        res.append(p)
    return res


@router.post("")
def new_project(files: List[UploadFile],
                response: Response,
                token: str = None,
                name: str = None,
                init_label: bool = True,
                config: str = Form(None),
                config_name: ConfigName = ConfigName.esnil):
    """upload multi files by user
    file type supported is txt,json,csv,xlsx

    .txt file will use filename as text title

    .csv and .xlsx file each row will seem as a text, and each row must include a 'title' column and 'content' column
    |  title  |   content   |
    -------------------------
    | Rapunzel| foo bar ... |

    .json file must is a list of dict
    [{'title': 'Rapunzel', 'content': 'foo bar ....'}]

    """
    config = json.loads(config) if config else config_mapping[config_name.value]
    try:
        res = pd.DataFrame(columns=config['columns'])
        for file in files:
            if file.filename.endswith('.csv'):
                res = pd.concat([res, pd.read_csv(io.BytesIO(file.file.read()))], ignore_index=True)
            elif file.filename.endswith('xlsx'):
                res = pd.concat([res, pd.read_excel(file.file.read())], ignore_index=True)
            elif file.filename.endswith('json'):
                res = pd.concat([res, pd.DataFrame(json.load(file.file))], ignore_index=True)
    except Exception as e:
        # raise e
        response.status_code = 400
        return 'Process data error, please check data content'

    df = mk.DataFrame.from_pandas(res, index=False)
    if 'id' not in config['columns']:
        config.append('id')
    if 'id' not in res.columns:
        df.create_primary_key("id")
    project_data_file = uuid.uuid4().hex
    project_name = name or project_data_file
    saved_path = os.path.join(project_base_path, f"{project_data_file}.mk")

    if token is not None:
        with engine.begin() as conn:
            user_res = conn.execute(select(user.c.id).where(user.c.token == token)).fetchone()
            inserted = conn.execute(project
                                    .insert()
                                    .values({"name": project_name,
                                             "user_id": user_res[0],
                                             'file_path': project_data_file,
                                             'config': config}).returning(project.c.id)).scalar_one()

            if inserted:
                label_config = config.get('default_label_config')
                init_label_data = pd.DataFrame(columns=label_config.get('columns'), index=None)
                new_label_uuid = uuid.uuid4().hex
                conn.execute(label_result
                             .insert()
                             .values({"project_id": inserted,
                                      "name": f'{name}_label',
                                      "user_id": user_res[0],
                                      "extra":   {'labeled_num': 0},
                                      "config": label_config,
                                      "file_path": new_label_uuid}))
                label_mk = mk.from_pandas(init_label_data, index=False)
                label_mk.write(os.path.join(label_base_path, f'{new_label_uuid}.mk'))
            df.write(saved_path)

    return {'saved_file': saved_path,
            'project_id': inserted}


@router.get("/{project_id}")
def get_single_project(project_id: int,
                       response: Response,
                       label_id: int = None,
                       size: int = 1000,
                       num: int = 0):
    """
    get a project data

    """
    project_res = get_project_by_id(project_id)

    if not project_res:
        response.status_code = 400
        return 'Project Not Found'

    res = {'project_meta': project_res,
           'data_num': 0,
           'label_num': 0}

    project_data_path = os.path.join(project_base_path, f"{project_res['file_path']}.mk")
    if os.path.exists(project_data_path):
        project_data = mk.read(project_data_path).to_pandas()
        total_num = len(project_data)
        project_data = project_data.iloc[size * num:size * (num + 1)]
        if label_id:
            label_res = get_label_by_id(label_id, project_id=project_id)
            label_data_path = os.path.join(label_base_path, f"{label_res['file_path']}.mk")
            label_data = mk.read(label_data_path).to_pandas()
            label_column = label_res['config']['label_column']
            res['label_num'] = len(label_data)
            label_data[label_column] = label_data[label_column].astype(int)
            merged_data = project_data.join(label_data.set_index('id'), on='id')
            project_data = merged_data.fillna(np.nan).replace([np.nan], [None])
        res['project_data'] = project_data.to_dict('records')
        res['total_num'] = total_num

    return res


@router.post("/{project_id}/training")
def trigger_project_train(project_id: int,
                          background_tasks: BackgroundTasks,
                          response: Response,
                          label_id: int):
    """
    get a project data

    """
    project_res = get_project_by_id(project_id)
    label_res = get_label_by_id(label_id=label_id, project_id=project_id)

    if not project_res:
        response.status_code = 400
        return 'Project Not Found'

    project_data = mk.read(os.path.join(project_base_path,
                                        f'{project_res["file_path"]}.mk')).to_pandas()
    label_data = mk.read(os.path.join(label_base_path,
                                      f'{label_res["file_path"]}.mk')).to_pandas()

    current_model = label_res['current_model']
    new_model_id = uuid.uuid4().hex
    project_with_label = label_data.merge(project_data, how='left', on='id')
    # project_with_label = project_data.join(label_data.set_index('id'), on='id')
    data_columns = project_res['config'].get('columns', [])
    label_columns = label_res['config'].get('columns', [])
    columns = set(data_columns + label_columns)
    all_labeled_project_data = project_with_label[project_with_label['label'].notnull()][columns]

    with engine.begin() as conn:
        conn.execute(label_result
                     .update()
                     .where(label_result.c.id == label_id)
                     .values({"last_model": label_result.c.current_model,
                              "current_model": new_model_id,
                              "iteration": label_result.c.iteration+1}))
        # conn.execute(model_info
        #              .insert()
        #              .values({"label_id": label_id,
        #                       "model_uuid": new_model_id,
        #                       "extra": {'train_begin': True},
        #                       "config": label_config,
        #                       "iteration": new_label_uuid}))
    background_tasks.add_task(one_training_iteration,
                              column_1=data_columns[0],
                              column_2=data_columns[1],
                              explanation_column=label_columns[-1],
                              labeled_data=all_labeled_project_data,
                              model_id=new_model_id,
                              old_model_id=current_model,
                              )

    return new_model_id


@router.get("/{project_id}/labels")
def list_label_result_of_a_project(project_id: int, response: Response):
    """
    get label result list for a project

    """
    label_res = get_labels_by_project_id(project_id=project_id)
    if not label_res:
        response.status_code = 204
    return label_res or []

