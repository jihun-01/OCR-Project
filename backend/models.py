from sqlalchemy import Column, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from database import Base

class OutboundOrder(Base):
    __tablename__ = "outbound_orders"
    
    id = Column(String(50), primary_key=True, index=True, comment="출고 ID")
    orderdate = Column(DateTime, nullable=False, comment="주문일")
    outbounddate = Column(DateTime, nullable=True, comment="출고일")
    
    # 고객 정보
    recipient = Column(String(100), nullable=False, comment="받는사람")
    phone = Column(String(20), nullable=False, comment="전화번호")
    address = Column(Text, nullable=False, comment="주소")
    
    # 운송 정보
    tracking_number = Column(String(100), nullable=True, comment="운송장번호")
    status = Column(String(50), nullable=False, default="주문 접수", comment="주문상태")
    
    # 상품 정보 (JSON으로 저장)
    items = Column(JSON, nullable=False, comment="주문상품목록")
    
    # 메타데이터
    created_at = Column(DateTime, default=func.now(), comment="생성일시")
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment="수정일시") 