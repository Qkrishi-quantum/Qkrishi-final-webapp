from apis.base import api_router
from core.config import settings
from db.base import Base
from db.session import engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


def create_tables():
    Base.metadata.create_all(bind=engine)


def include_router(app):
    app.include_router(api_router)


def start_application():
    app = FastAPI()
    
    origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    create_tables()
    include_router(app)
    return app


app = start_application()
