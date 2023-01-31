from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


#  = Table(
#     "author_publisher",
#     Base.metadata,
#     Column("author_id", Integer, ForeignKey("author.author_id")),
#     Column("publisher_id", Integer, ForeignKey("publisher.publisher_id")),
# )



# author_publisher = Table(
#     "author_publisher",
#     Base.metadata,
#     Column("author_id", Integer, ForeignKey("author.author_id")),
#     Column("publisher_id", Integer, ForeignKey("publisher.publisher_id")),
# )

class Project(Base):
    __tablename__ = "project"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    create_time = Column(DateTime)
    update_time = Column(DateTime)
    user_id = Column(Integer, ForeignKey("user.id"))


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    create_time = Column(DateTime)
    update_time = Column(DateTime)
    token = Column(String)


class Label(Base):
    __tablename__ = "label"
    id = Column(Integer, primary_key=True)
    version = Column(String)
    user_id = Column(Integer, ForeignKey("user.id"))
    project_id = Column(Integer, ForeignKey("project.id"))
    create_time = Column(DateTime)
    update_time = Column(DateTime)

