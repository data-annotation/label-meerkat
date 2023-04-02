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

    conn = engine.connect()
    conn.execute(user.insert(), {"name": name, "token": token})
    return True


@router.get("/")
def user_list():
    """
    get all user list

    """
    conn = engine.connect()
    users = conn.execute(select(user.c.name, user.c.create_time, user.c.update_time))
    res = []
    for name, create_time, update_time in users:
        res.append({'name': name,
                    'create_time': create_time,
                    'update_time': update_time})
    return res

