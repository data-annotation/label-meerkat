import io
import json
import os
import uuid
from enum import Enum
from typing import List
from typing import Union
import zipfile

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi import Response
from fastapi import UploadFile
from sentence_splitter import SentenceSplitter
from sqlalchemy import select

import meerkat as mk
from . import engine
from . import model
from ..config import project_base_path
from ..orm.anno_project import project
from ..orm.anno_project import user

router = APIRouter(
    prefix="/file",
    tags=["file"],
    responses={404: {"description": "Not found"}},
)


class LanguageType(Enum):
    en = 'en'
    zh = 'zh'


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


@router.post("/data/process")
def upload_file_and_process(files: List[UploadFile],
                            response: Response,
                            token: str = None,
                            name: str = None):
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
    res = []
    try:
        for file in files:
            if file.filename.endswith('.txt'):
                res.extend(text_to_sentence(file.file.read(), file.filename[:-5]))
            elif file.filename.endswith('.csv'):
                res.extend(dataframe_process(pd.read_csv(io.BytesIO(file.file.read()))))
            elif file.filename.endswith('xlsx'):
                res.extend(dataframe_process(pd.read_excel(file.file.read())))
            elif file.filename.endswith('json'):
                for text in json.load(file.file):
                    res.extend(dataframe_process(text['content', text['title']]))
            elif file.filename.endswith('zip') or file.filename.endswith('rar'):
                res.extend(zip_process(file))


    except Exception as e:
        # raise e
        response.status_code = 400
        return 'Process data error, please check data content'

    df = mk.DataFrame(res)
    df['embed'] = model.encode(df['sentence'].to_list())
    project_name = name or uuid.uuid4().hex
    save_processed_file_name = f"{project_name}.mk"
    saved_path = os.path.join(project_base_path, save_processed_file_name)

    if token is not None:
        with engine.connect() as conn:
            user_res = conn.execute(select(user.c.id).where(user.c.token == token)).fetchone()
            conn.execute(project.insert(), {"name": project_name, "user_id": user_res[0], 'file_path': saved_path})
            df.write(saved_path)
    return {'res': res,
            'saved_file': save_processed_file_name}


@router.post("/data/match")
def upload_file_and_process(data: str, search_words, response: Response):
    """search the best match sentence in processed data by search_words

    """
    df = mk.read(f'{data}.mk')
    kw_embed = model.encode(search_words)
    df['scores'] = df['embed'].map(
        lambda x: np.dot(x, kw_embed) / (np.linalg.norm(x) * np.linalg.norm(kw_embed))).squeeze()
    sort_by_keyword_df = df.sort(by='scores', ascending=False)
    return sort_by_keyword_df.head(10).to_pandas().to_dict('records')


