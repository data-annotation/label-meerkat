from fastapi import APIRouter
from sqlalchemy import select

from . import engine
from ..orm.tables import user

router = APIRouter(
    prefix="/users",
    tags=["user"],
    responses={404: {"description": "Not found"}},
)


@router.post("/")
def new_user(name: str, token: str = None):
    """
    create a new user

    """
    token = token or name
    with engine.begin() as conn:
        res = conn.execute(user
                           .insert()
                           .values({"name": name, "token": token})
                           .returning(user.c.id,
                                      user.c.name,
                                      user.c.token)).fetchone()._asdict()
    return res


@router.get("/")
def user_list():
    """
    get all user list

    """
    conn = engine.connect()
    users = conn.execute(select(user.c.name,
                                user.c.id,
                                user.c.create_time,
                                user.c.update_time,
                                user.c.token)).mappings().all()
    return users

