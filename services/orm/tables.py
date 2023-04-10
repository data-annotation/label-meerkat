import json

from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
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
    Column("iteration", Integer, default=1),
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
    Column("extra", JSON, default=dict(), nullable=False,
           server_default=text("'{}'")),
    Column("iteration", Integer, default=1),
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


def get_project_by_id(project_id: int, conn=None):
    conn = conn or engine.connect()
    project_res = (conn.execute(select(project.c.id,
                                       project.c.name,
                                       project.c.file_path,
                                       project.c.config,
                                       project.c.create_time,
                                       project.c.update_time)
                                .where(project.c.id == project_id))
                   .fetchone()._asdict())
    return project_res if project_res else None


def get_label_by_id(label_id: int, project_id: int = None, conn=None):
    conn = conn or engine.connect()
    cond = [label_result.c.id == label_id]
    if project_id:
        cond.append(label_result.c.project_id == project_id)
    label_res = (conn.execute(select(label_result.c.id,
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
                              .where(and_(*cond)))
                 .fetchone()._asdict())
    return label_res if label_res else None


def get_labels_by_project_id(project_id: int, conn=None):
    conn = conn or engine.connect()
    sql = select(label_result.c.id,
                 label_result.c.name,
                 label_result.c.user_id,
                 label_result.c.project_id,
                 label_result.c.config,
                 label_result.c.create_time,
                 label_result.c.update_time).where(label_result.c.project_id == project_id)

    return conn.execute(sql).mappings().all()


engine = create_engine("sqlite:///test.db", echo=True)

if __name__ == "__main__":

    metadata_obj.create_all(engine)
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
        print('init db wrong')

    # project_res = get_project_by_id(1)
    # pprint.pprint(project_res)
    # print(project_res['id'])

