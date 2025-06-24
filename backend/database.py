from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터베이스 URL 설정
DATABASE_URL = "mysql+pymysql://wmsuser:wmspassword123@127.0.0.1:3306/wms_ocr"

logger.info(f"데이터베이스 URL: {DATABASE_URL}")

try:
    # 엔진 생성
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=True,  # 디버깅을 위해 SQL 로그 활성화
        connect_args={
            "charset": "utf8mb4",
            "use_unicode": True
        }
    )
    
    # 연결 테스트 (SQLAlchemy 2.0 방식)
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        logger.info("데이터베이스 연결 성공!")
        
except Exception as e:
    logger.error(f"데이터베이스 연결 실패: {e}")
    raise

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스 생성
Base = declarative_base()

# 데이터베이스 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 연결 테스트 함수 (SQLAlchemy 2.0 방식)
def test_connection():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT VERSION()"))
            version = result.fetchone()
            logger.info(f"MySQL 버전: {version[0]}")
            return True
    except Exception as e:
        logger.error(f"연결 테스트 실패: {e}")
        return False