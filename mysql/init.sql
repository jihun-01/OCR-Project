-- UTF-8 설정
SET NAMES utf8mb4;
SET CHARACTER SET utf8mb4;

-- 데이터베이스 생성 (UTF-8 명시)
CREATE DATABASE IF NOT EXISTS wms_ocr 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

USE wms_ocr;

-- 테이블 생성 (UTF-8 명시)
CREATE TABLE IF NOT EXISTS outbound_orders (
    id VARCHAR(50) PRIMARY KEY COMMENT '출고 ID',
    orderdate DATETIME NOT NULL COMMENT '주문일',
    outbounddate DATETIME NULL COMMENT '출고일',
    recipient VARCHAR(100) NOT NULL COMMENT '받는사람',
    phone VARCHAR(20) NOT NULL COMMENT '전화번호',
    address TEXT NOT NULL COMMENT '주소',
    tracking_number VARCHAR(100) NULL COMMENT '운송장번호',
    status VARCHAR(50) NOT NULL DEFAULT '주문 접수' COMMENT '주문상태',
    items JSON NOT NULL COMMENT '주문상품목록',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '생성일시',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정일시'
) ENGINE=InnoDB 
  DEFAULT CHARSET=utf8mb4 
  COLLATE=utf8mb4_unicode_ci;

-- 초기 데이터 삽입 (UTF-8로 저장)
INSERT INTO outbound_orders (id, orderdate, outbounddate, recipient, phone, address, tracking_number, status, items) VALUES
('SHP-20240601-001', '2025-06-15 10:00:00', '2025-06-16 14:00:00', '홍길동', '010-1234-5678', '서울시 강남구 역삼동 123-45, 무슨무슨 건물 101호', 'TRACK-001', '출고 준비', '[{"name": "마우스", "quantity": 2, "price": 15000}, {"name": "키보드", "quantity": 1, "price": 30000}]'),
('SHP-20240601-002', '2025-06-16 11:30:00', NULL, '김수', '010-2345-6789', '서울시 강남구 역삼동 123-45, 무슨무슨 건물 102호', '', '주문 접수', '[{"name": "USB 케이블", "quantity": 5, "price": 5000}]'),
('SHP-20240601-003', '2025-06-17 09:15:00', NULL, '이지현', '010-3456-7890', '서울시 강남구 역삼동 123-45, 무슨무슨 건물 103호', '', '출고 준비', '[{"name": "마우스", "quantity": 2, "price": 15000}, {"name": "키보드", "quantity": 1, "price": 30000}]'),
('SHP-20240601-004', '2025-06-18 13:45:00', '2025-06-19 16:20:00', '박윤호', '010-4567-8901', '서울시 강남구 역삼동 123-45, 무슨무슨 건물 104호', 'TRACK-002', '배송 중', '[{"name": "마우스", "quantity": 2, "price": 15000}, {"name": "키보드", "quantity": 1, "price": 30000}]'),
('SHP-20240601-005', '2025-06-19 08:30:00', NULL, '최다영', '010-5678-9012', '서울시 강남구 역삼동 123-45, 무슨무슨 건물 105호', '', '취소', '[{"name": "마우스", "quantity": 2, "price": 15000}, {"name": "키보드", "quantity": 1, "price": 30000}]');

-- 문자셋 확인 쿼리
SHOW VARIABLES LIKE 'character%';
SHOW VARIABLES LIKE 'collation%';