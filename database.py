import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      
    pool_recycle=300        
)

session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
base = declarative_base()

def get_db():
    db = session()
    try:
        yield db
    finally:
        db.close()
