import json

from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Boolean
from sqlalchemy import Table
from sqlalchemy import JSON
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import MetaData
from sqlalchemy import Sequence
from sqlalchemy import Index
from sqlalchemy import TypeDecorator
from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy import create_engine

from services.const import ModelStatus

metadata_obj = MetaData()
import pprint


class JSONEncodedDict(TypeDecorator):
    impl = JSON

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


project = Table(
    "project",
    metadata_obj,
    Column("id", Integer, Sequence("project_id_seq"), primary_key=True, comment="annotation project id"),
    Column("name", String, index=True, unique=True, nullable=False, comment="project name"),
    Column("create_time", DateTime(timezone=True), server_default=func.now()),
    Column("update_time", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
    Column("user_id", Integer, ForeignKey("user.id"), nullable=False),
    Column("file_path", String, index=True, unique=True, nullable=False, comment="project data file path"),
    Column("config", JSON, nullable=False)
)

user = Table(
    "user",
    metadata_obj,
    Column("id", Integer, Sequence("user_id_seq"), primary_key=True),
    Column("name", String, index=True, unique=True, nullable=False),
    Column("create_time", DateTime(timezone=True), server_default=func.now()),
    Column("update_time", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
    Column("token", String, index=True, unique=True, nullable=False, comment="identify the user"),
)


label_result = Table(
    "label_result",
    metadata_obj,
    Column("id", Integer, Sequence("label_id_seq"), index=True, primary_key=True),
    Column("name", String, nullable=False),
    Column("create_time", DateTime(timezone=True), server_default=func.now()),
    Column("update_time", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
    Column("user_id", Integer, ForeignKey("user.id"), nullable=False),
    Column("project_id", Integer, ForeignKey("project.id"), nullable=False),
    Column("config", JSON, nullable=False),
    Column("extra", JSON, default=dict(), nullable=False,
           server_default=text("'{}'")),
    Column("iteration", Integer, default=0),
    Column("last_model", String),
    Column("current_model", String),
    Column("file_path", String,  unique=True, nullable=False)
)


model_info = Table(
    "model_info",
    metadata_obj,
    Column("id", Integer, Sequence("label_id_seq"), index=True, primary_key=True),
    Column("model_uuid", String, nullable=False, index=True, unique=True),
    Column("create_time", DateTime(timezone=True), server_default=func.now()),
    Column("update_time", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
    Column("label_id", Integer, ForeignKey("label_result.id"), nullable=False),
    Column("deleted", Boolean, default=False),
    Column("extra", JSON, default=dict(), nullable=False,
           server_default=text("'{}'")),
    Column("status", Integer, default=1, comment="0表示未在训练，1表示正在训练"),
    Column("iteration", Integer, default=1, comment="模型的迭代次数"),
    Column("data_num", Integer, default=0, comment="模型训练使用的数据量"),
)

# label_config = Table(
#     "label_config",
#     metadata_obj,
#     Column("id", Integer, Sequence("label_config_id_seq"), primary_key=True),
#     Column("name", String, nullable=False),
#     Column("user_id", String, nullable=False),
#     Column("project_id", Integer, ForeignKey("project.id"), nullable=False),
#     Column("extra", JSON, server_default=func.now()),
#     Column("create_time", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
#     Column("update_time", DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
# )

Index("ux_label_result_name_user_id", label_result.c.name, label_result.c.user_id, unique=True)

basic_label_cols = [label_result.c.id,
                    label_result.c.name,
                    label_result.c.user_id,
                    label_result.c.project_id,
                    label_result.c.config,
                    label_result.c.extra,
                    label_result.c.last_model,
                    label_result.c.current_model,
                    label_result.c.create_time,
                    label_result.c.update_time,
                    label_result.c.iteration,
                    label_result.c.file_path]

basic_model_cols = [model_info.c.id,
                    model_info.c.model_uuid,
                    model_info.c.status,
                    model_info.c.label_id,
                    model_info.c.extra,
                    model_info.c.deleted,
                    model_info.c.data_num,
                    model_info.c.create_time,
                    model_info.c.update_time]

def get_project_by_id(project_id: int, conn=None):
    with engine.connect() as conn:
      project_res = (conn.execute(select(project.c.id,
                                         project.c.name,
                                         project.c.file_path,
                                         project.c.config,
                                         project.c.create_time,
                                         project.c.update_time)
                                  .where(project.c.id == project_id))
                     .fetchone()._asdict())
    return project_res if project_res else None


def get_label_by_id(label_id: int, project_id: int = None):
  cond = [label_result.c.id == label_id]
  if project_id:
      cond.append(label_result.c.project_id == project_id)
  with engine.connect() as conn:
    label_res = (conn.execute(select(*basic_label_cols)
                              .where(and_(*cond)))
                 .fetchone()._asdict())
  return label_res if label_res else None


def get_single_project_label(project_id: int, label_id: int = None):
  if label_id:
    res = get_label_by_id(label_id=label_id)
  else:
    with engine.connect() as conn:
      res = (conn.execute(select(*basic_label_cols)
                           .where(and_(label_result.c.project_id == project_id))
                           .order_by(label_result.c.update_time.desc()))
              .fetchone()._asdict()) or None
  return res


def get_labels_by_project_id(project_id: int, conn=None):
  sql = select(*basic_label_cols).where(label_result.c.project_id == project_id)
  with engine.connect() as conn:
    res = conn.execute(sql).mappings().all()
  return res


def get_models_by_label_id(label_id: int,
                           deleted: bool = None,
                           conn=None):
  cond = [model_info.c.label_id == label_id]
  if deleted is not None:
    cond.append(model_info.c.deleted == deleted)
  sql = (select(*basic_model_cols)
         .where(and_(*cond))
         .order_by(model_info.c.data_num.desc(),
                   model_info.c.update_time.desc()))
  with engine.connect() as conn:
    res = conn.execute(sql).mappings().all()
  return res


def create_new_model(label_id: int,
                     model_id: str,
                     status: ModelStatus = ModelStatus.free.value,
                     extra: dict = None,
                     iteration: int = 1,
                     conn=None):
  res = conn.execute(model_info
                     .insert()
                     .values({"label_id": label_id,
                              "model_uuid": model_id,
                              "extra": extra or dict(),
                              "status": status,
                              "iteration": iteration})
                     .returning(model_info.c.id,
                                model_info.c.model_uuid)).fetchone()._asdict()

  return res


engine = create_engine("sqlite:///test.db", echo=True, pool_size=10, max_overflow=10, pool_recycle=1800)

if __name__ == "__main__":

    metadata_obj.create_all(engine)
    # model_info.create(engine)
    # ins = user.insert().values(name="jack", token="Jack Jones")
    # ins = project.insert().values(name="faire_tale_label", user_id=1, file_path="abc")
    # ins = label.insert().values(name="jack label", project_id=1, file_path="abc")
    try:
        with engine.begin() as conn:
            user_id = conn.execute(user.insert().returning(project.c.id),
                                   {"name": "jack", "token": "_password"}).scalar()
            # project_id = conn.execute(project.insert().returning(project.c.id),
            #                           {"name": "faire_tale_label",
            #                            "user_id": user_id,
            #                            "file_path": "foo",
            #                            "config": {"type": "sentence_relation",
            #                                       "columns": ["premise", "hypothesis"]}}).scalar()
            # res = conn.execute(label_result
            #                    .insert()
            #                    .values({"name": "jack label",
            #                             "project_id": project_id,
            #                             "user_id": user_id,
            #                             "config": {
            #                                 'label_choice': ['entailment', 'neutral', 'contradiction'],
            #                                 'sentence_column_1': 'premise',
            #                                 'sentence_column_2': 'hypothesis'},
            #                             "iteration": 1,
            #                             "file_path": "foo"}))
    except Exception as e:
        print(e)
        print('init db wrong')

    # project_res = get_project_by_id(1)
    # pprint.pprint(project_res)
    # print(project_res['id'])

