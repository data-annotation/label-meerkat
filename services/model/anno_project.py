from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Table
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
    Column("id", Integer, Sequence("project_id_seq"), primary_key=True),
    Column("name", String, index=True, unique=True, nullable=False),
    Column("create_time", DateTime(timezone=True), server_default=func.now()),
    Column("update_time", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
    Column("user_id", Integer, ForeignKey("user.id"), nullable=False),
    Column("file_path", String, index=True, unique=True, nullable=False)
)

user = Table(
    "user",
    metadata_obj,
    Column("id", Integer, Sequence("user_id_seq"), primary_key=True),
    Column("name", String, index=True, unique=True, nullable=False),
    Column("create_time", DateTime(timezone=True), server_default=func.now()),
    Column("update_time", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
    Column("token", String, index=True, unique=True, nullable=False),
)

label = Table(
    "label",
    metadata_obj,
    Column("id", Integer, Sequence("label_id_seq"), primary_key=True),
    Column("name", String, nullable=False),
    Column("create_time", DateTime(timezone=True), server_default=func.now()),
    Column("update_time", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
    Column("user_id", Integer, ForeignKey("user.id"), nullable=False),
    Column("project_id", Integer, ForeignKey("project.id"), nullable=False),
    Column("file_path", String, index=True, unique=True, nullable=False)
)

Index("ux_label_name_user_id", label.c.name, label.c.user_id, unique=True)

if __name__ == "__main__":
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///test.db", echo=True)
    metadata_obj.create_all(engine)
    # ins = user.insert().values(name="jack", token="Jack Jones")
    # ins = project.insert().values(name="faire_tale_label", user_id=1, file_path="abc")
    # ins = label.insert().values(name="jack label", project_id=1, file_path="abc")
    conn = engine.connect()
    conn.execute(user.insert(), {"name": "jack", "token": "password"})
    # conn.execute(project.insert(), {"name": "faire_tale_label", "user_id": 2, "file_path": "foo"})
    # conn.execute(label.insert(), {"name": "jack label", "project_id": 3, "user_id": 2, "file_path": "foo"})
    # conn.execute(ins)

