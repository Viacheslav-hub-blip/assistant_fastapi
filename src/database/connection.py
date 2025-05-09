from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from config import *

db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

engine = create_engine(db_url)

my_Session = sessionmaker(bind=engine)
session = my_Session()
print(session)