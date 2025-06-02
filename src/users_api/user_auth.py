from datetime import timedelta, datetime
from typing import Union, Any

import jwt
from pydantic import BaseModel

from src.users_api.config import SECRET_KEY
from src.users_api.services.user_service import UserService


class Token(BaseModel):
    access_token: str
    token_type: str
    access_token_expires: str


class UserDTO(BaseModel):
    id: int
    email: str
    login: str
    password: str


user_service = UserService()


class UserAuth:
    def create_access_token(self, data: dict, expires_delta: timedelta) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire})
        encode_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
        return encode_jwt

    def login_for_access_token(self, email: str, password: str) -> dict[str, Any]:
        user: UserDTO = self.validate_user(email, password)

        if not user:
            return {"status": 404, "error": "Incorrect email or password"}

        access_token_expires = timedelta(minutes=120)
        access_token = self.create_access_token(
            data={"email": user.email, "password": password}, expires_delta=access_token_expires
        )

        return {"status": 200, "token": Token(access_token=access_token, token_type="bearer",
                                              access_token_expires=str(access_token_expires))}

    def validate_user(self, email: str, password: str) -> Union[UserDTO, None]:
        user = user_service.select_user_by_email(email)

        if user and user.password == password:
            return UserDTO(id=user.id, email=user.email, login=user.login, password=password)
        return None

    def get_current_user_by_token(self, token: str) -> dict[str, Any]:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])

            email: str = payload.get("email")
            password: str = payload.get("password")
            exp: str = payload.get("exp")

            if email is None or password is None or exp is None:
                return {"status": 404, "error": "Incorrect email or password"}

            if datetime.fromtimestamp(float(exp)) - datetime.now() < timedelta(minutes=1):
                return {"status": 400, "error": "Timeout"}

            user: UserDTO = self.validate_user(email, password)

            if user is None:
                return {"status": 404, "error": "User not found"}
            return {"status": 200, "user": user}

        except jwt.ExpiredSignatureError:
            return {"status": 400, "error": "Failed decode token"}
