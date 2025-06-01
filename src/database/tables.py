from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base

from src.database.connection import engine

Base = declarative_base()


class Users(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String)
    password = Column(String)

    def __repr__(self):
        return f"{self.id}, {self.email}, {self.password}"


class WorkSpace(Base):
    __tablename__ = 'workspace'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    name = Column(String)

    def __repr__(self):
        return f"{self.id}, {self.name}"


class Messages(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    workspace_id = Column(Integer, ForeignKey('workspace.id'))
    message = Column(String)
    message_type = Column(String)

    def __repr__(self):
        return f"{self.id}, {self.user_id}, {self.workspace_id}, {self.message_type}"


class Files(Base):
    __tablename__ = 'files'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    workspace_id = Column(Integer, ForeignKey('workspace.id'))
    file_name = Column(String)
    load_date = Column(String)
    summary_content = Column(String)

    def __repr__(self):
        return f"{self.user_id}, {self.workspace_id}, {self.load_date}, {self.summary_content}"


class Chunks(Base):
    __tablename__ = 'chunks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    workspace_id = Column(Integer, ForeignKey('workspace.id'))
    source_doc_id = Column(Integer, ForeignKey('files.id'))
    source_doc_name = Column(String)
    doc_number = Column(Integer)
    summary_content = Column(String)

    def __repr__(self):
        return f"{self.user_id}, {self.workspace_id}, {self.source_doc_name}, {self.summary_content}"


class WorkspacesMarket(Base):
    __tablename__ = "workspaces_market"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    source_workspace_id = Column(Integer, ForeignKey('workspace.id'))
    workspace_name = Column(String)
    workspace_description = Column(String)

    def __repr__(self):
        return f"{self.user_id}, {self.source_workspace_id}, {self.workspace_name}, {self.workspace_description}"


Base.metadata.create_all(engine)
