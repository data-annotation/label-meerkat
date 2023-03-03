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
from sqlalchemy import func

metadata_obj = MetaData()

project = Table(
    "project",
    metadata_obj,
    Column("id", Integer, Sequence("project_id_seq"), primary_key=True, comment="annoation project id"),
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
    Column("iteration", Integer, default=1),
    Column("last_model", String),
    Column("current_model", String),
    Column("file_path", String,  unique=True, nullable=False)
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


if __name__ == "__main__":
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///test.db", echo=True)
    metadata_obj.create_all(engine)
    # ins = user.insert().values(name="jack", token="Jack Jones")
    # ins = project.insert().values(name="faire_tale_label", user_id=1, file_path="abc")
    # ins = label.insert().values(name="jack label", project_id=1, file_path="abc")
    conn = engine.connect()
    conn.execute(user.insert(),
                 {"name": "jack", "token": "password"})
    conn.execute(project.insert(),
                 {"name": "faire_tale_label",
                  "user_id": 2,
                  "file_path": "foo",
                  "config": {"type": "sentence_relation",
                             "columns": ["premise", "hypothesis"]}})
    res = conn.execute(label_result
                       .insert()
                       .values({"name": "jack label",
                                "project_id": 3,
                                "user_id": 2,
                                "config": {
                                  'label_choice': ['entailment', 'neutral', 'contradiction'],
                                  'sentence_column_1': 'premise',
                                  'sentence_column_2': 'hypothesis'},
                                "iteration": 1,
                                "file_path": "foo"}))
    # conn.execute(ins)

