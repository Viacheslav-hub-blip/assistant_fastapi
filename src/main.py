from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.rag_agent_api.routers.routers import router
from src.users_api.routers import router as user_router

app = FastAPI()

app.include_router(router)
app.include_router(user_router)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# uvicorn src.main:app --use-colors --log-level debug --reload
