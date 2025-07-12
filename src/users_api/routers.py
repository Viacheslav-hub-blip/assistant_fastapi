from typing import Any

from fastapi import APIRouter

from src.users_api.services.user_service import UserService
from src.users_api.user_auth import UserAuth

router = APIRouter(
    prefix="/users_api",
    tags=["Users"],
)

# экземпляр класса
user_auth = UserAuth()


@router.post("/registrate")
async def register(email: str, login: str, password: str) -> dict[str, Any]:
    if UserService.check_user_exists(email):
        return {"status": 404, "message": "User already exists"}
    new_user_id = UserService.insert_user(email, login, password)
    token = user_auth.login_for_access_token(email, password)["token"]
    print("token", token)
    return {"status": 200, "user_id": new_user_id, "token": token}


@router.post("/login")
async def login(email: str, password: str) -> dict[str, Any]:
    # получаем токен и возращаем клиенту
    if not UserService.check_user_exists(email):
        return {"status": 404, "message": "Неправильный логин или пароль"}
    user = UserService.select_user_by_email(email)
    print("user", user)
    token = user_auth.login_for_access_token(email, password)
    return {"status": 200, "user_id": user.id, "token": token}


@router.post("/decode_token")
async def decode_token(token: str) -> dict[str, Any]:
    print("token", token)
    return user_auth.get_current_user_by_token(token)
