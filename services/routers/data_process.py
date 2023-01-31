import io
import json
import uuid
from enum import Enum
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi import Response
from fastapi import UploadFile
from sentence_splitter import SentenceSplitter

import meerkat as mk
# from meerkat.ops.embed import transformers
# from meerkat.ops.embed.registry import encoders
import sqlite3

from sentence_transformers import SentenceTransformer

sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

# encoders._obj_map['transformers'] = transformers

router = APIRouter(
    prefix="/file",
    tags=["file"],
    responses={404: {"description": "Not found"}},
)


class LanguageType(Enum):
    eng = 'eng'
    chn = 'chn'


splitter = SentenceSplitter(language='en')


def text_to_sentence(text: Union[str, bytes], name: str = None):
    text = text.decode() if isinstance(text, bytes) else text
    paragraphs = [i.strip('\n\t').strip()
                  for i in text.split('\n')
                  if i.strip('\n\t').strip() != '']
    res = []
    for p_idex, paragraph in enumerate(paragraphs):
        res.extend([{'paragraph': p_idex,
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


def calc_cosine_distance(a: np.array, b: np.array):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@router.post("/data/process")
def upload_file_and_process(files: List[UploadFile], response: Response):
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
    except Exception as e:
        raise e
        response.status_code = 400
        return 'Process data faield, please check data content'

    df = mk.DataFrame(res)
    df['embed'] = model.encode(df['sentence'].to_list())
    # df: mk.DataFrame = mk.embed(
    #     df,
    #     input="sentence",
    #     modality="text",
    #     encoder='transformers',
    #     out_col='embed',
    #     batch_size=32,
    # )
    save_processed_file_name = f"processed-{uuid.uuid4().hex}.mk"
    df.write(save_processed_file_name)

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

