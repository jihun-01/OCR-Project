from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from datetime import datetime
import easyocr
import numpy as np
from PIL import Image
import io
import logging
import re
import cv2
import torch
from typing import List, Optional
import os
import dotenv

dotenv.load_dotenv()
FRONT_URL = os.getenv("FRONT_URL")

# 데이터베이스 관련 import 추가
from database import engine, get_db, Base
from models import OutboundOrder

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EasyOCR reader 전역 변수
reader = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시
    global reader
    logger.info("EasyOCR 초기화 중...")
    
    # 데이터베이스 테이블 생성
    Base.metadata.create_all(bind=engine)
    logger.info("데이터베이스 테이블 생성 완료")
    
    # GPU 사용 가능 여부 확인
    gpu_available = torch.cuda.is_available()
    logger.info(f"GPU 사용 가능: {gpu_available}")
    
    reader = easyocr.Reader(
        ['ko', 'en'], 
        gpu=True,  # GPU 사용
        download_enabled=True,
        model_storage_directory='./models',  # 모델 캐시 디렉토리
        user_network_directory='./models'    # 사용자 네트워크 디렉토리
    )
    logger.info("EasyOCR 초기화 완료")
    yield
    # 종료 시 (필요하면 정리 작업 추가)

app = FastAPI(
    title="OCR API", 
    description="EasyOCR을 사용한 텍스트 추출 API",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (개발 환경용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def optimize_image_for_ocr(image_array, max_width=1200, max_height=1200):
    """
    OCR 성능을 위해 이미지를 최적화합니다.
    """
    height, width = image_array.shape[:2]
    
    # 이미지가 너무 크면 리사이즈 (메모리 절약 및 속도 향상)
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"이미지 리사이즈: {width}x{height} -> {new_width}x{new_height}")
    
    return image_array

def optimize_image_for_delivery(image_array, min_width=800, max_width=3000, min_height=800, max_height=3000):
    """
    운송장 OCR을 위해 이미지를 최적화합니다.
    운송장의 작은 텍스트들을 고려하여 적절한 크기로 조정합니다.
    """
    height, width = image_array.shape[:2]
    
    # 너무 작은 이미지는 확대 (운송장의 작은 텍스트 인식을 위해)
    if width < min_width or height < min_height:
        scale = max(min_width / width, min_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        logger.info(f"운송장 이미지 확대: {width}x{height} -> {new_width}x{new_height}")
    
    # 너무 큰 이미지는 축소 (메모리 절약, 하지만 너무 작게 하지 않음)
    elif width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"운송장 이미지 축소: {width}x{height} -> {new_width}x{new_height}")
    
    return image_array

def fast_preprocessing(image_array):
    """
    빠른 전처리 (기본적인 노이즈 제거만)
    """
    # 그레이스케일 변환
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # 간단한 노이즈 제거 (가우시안 블러)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 적응형 이진화
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, 11  # 블록 크기와 C값 조정으로 속도 향상
    )
    
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

def delivery_preprocessing(image_array):
    """
    운송장 전용 전처리 (인식률 향상을 위해 더 정교한 처리)
    """
    # 그레이스케일 변환
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # 노이즈 제거 (미디언 블러 - 운송장의 선명한 텍스트를 위해)
    denoised = cv2.medianBlur(gray, 3)
    
    # 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 적응형 이진화 (운송장의 다양한 배경을 위해)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 31, 15  # 원래 값으로 복원
    )
    
    # 모폴로지 연산으로 텍스트 선명도 향상
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

def clean_address(address_text):
    """
    주소에서 지역명(시/도) 이전의 불필요한 내용을 제거합니다.
    """
    if not address_text:
        return address_text
    
    # 주요 지역명 (시/도 단위) - 긴 이름부터 정렬하여 우선 매칭
    region_names = [
        '서울특별시', '부산광역시', '대구광역시', '인천광역시', 
        '광주광역시', '대전광역시', '울산광역시', '세종특별자치시',
        '강원특별자치도', '제주특별자치도',
        '경기도', '강원도', '충청북도', '충청남도', 
        '전라북도', '전라남도', '경상북도', '경상남도', '제주도',
        '서울시', '부산시', '대구시', '인천시', '광주시', 
        '대전시', '울산시', '세종시',
        '서울', '부산', '대구', '인천', '광주', 
        '대전', '울산', '세종', '경기', '강원', 
        '충북', '충남', '전북', '전남', '경북', '경남', '제주'
    ]
    
    # 지역명을 찾아서 그 위치부터 주소 시작
    for region in region_names:
        if region in address_text:
            # 지역명이 나타나는 첫 번째 위치 찾기
            region_index = address_text.find(region)
            if region_index != -1:
                # 지역명부터 시작하는 주소 반환
                cleaned_address = address_text[region_index:].strip()
                if cleaned_address != address_text:
                    logger.info(f"주소 정리: '{address_text}' -> '{cleaned_address}'")
                return cleaned_address
    
    # 지역명이 없으면 원본 반환
    return address_text

def extract_complete_address(all_texts, address_candidates):
    """
    여러 텍스트를 결합하여 완전한 주소(도로명주소 + 상세주소)를 추출합니다.
    """
    if not address_candidates:
        return None
    
    # 가장 좋은 주소 후보를 기본으로 선택
    best_address = max(address_candidates, 
                      key=lambda x: (x['keyword_count'], x['length'], x['confidence']))
    base_address = best_address['text']
    
    # 상세주소 키워드들
    detail_keywords = ['동', '호', '층', '번지', '번길', '가', '단지', '아파트', 'APT', '빌딩', '빌라']
    
    # 상세주소를 찾아서 결합
    found_details = []
    
    logger.info(f"기본 주소: {base_address}")
    logger.info(f"전체 텍스트 개수: {len(all_texts)}")
    
    # 기본 주소에 이미 포함된 건물명들 추출
    base_building_names = []
    for bld_keyword in ['아파트', 'APT', '빌딩', '빌라', '단지', '타워', '플라자']:
        bld_pattern = rf'[가-힣a-zA-Z0-9\s]*{bld_keyword}'
        existing_matches = re.findall(bld_pattern, base_address)
        for match in existing_matches:
            clean_match = match.strip()
            if clean_match and len(clean_match) >= 2:
                base_building_names.append(clean_match)
    
    logger.info(f"기본 주소에 포함된 건물명들: {base_building_names}")
    
    for text_info in all_texts:
        text = text_info['text']
        confidence = text_info['confidence']
        
        # 신뢰도가 낮은 텍스트는 제외
        if confidence < 0.3:
            continue
            
        logger.info(f"검사 중인 텍스트: '{text}' (신뢰도: {confidence})")
        
        # 이미 기본 주소에 포함된 텍스트는 건너뛰기
        if text in base_address:
            logger.info(f"이미 포함된 텍스트 건너뛰기: {text}")
            continue
        
        # 상세주소 패턴 찾기
        for keyword in detail_keywords:
            if keyword in text:
                logger.info(f"키워드 '{keyword}' 발견 in '{text}'")
                
                if keyword in ['동', '호', '층']:
                    # 숫자 + 동/호/층 패턴 (더 유연하게)
                    patterns = [
                        rf'\d+\s*{keyword}',      # 1405호
                        rf'\d+[-]\d+\s*{keyword}', # 101-1405호
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        if matches:
                            for match in matches:
                                clean_match = match.strip()
                                if clean_match not in found_details:
                                    found_details.append(clean_match)
                                    logger.info(f"상세주소 발견: {clean_match}")
                
                elif keyword in ['번지', '번길']:
                    # 번지/번길 패턴
                    pattern = rf'\d+[-]?\d*\s*{keyword}'
                    matches = re.findall(pattern, text)
                    if matches:
                        for match in matches:
                            clean_match = match.strip()
                            if clean_match not in found_details:
                                found_details.append(clean_match)
                                logger.info(f"상세주소 발견: {clean_match}")
                
                elif keyword in ['가']:
                    # 가 패턴 (예: 1가, 2가)
                    pattern = rf'\d+\s*{keyword}'
                    matches = re.findall(pattern, text)
                    if matches:
                        for match in matches:
                            clean_match = match.strip()
                            if clean_match not in found_details:
                                found_details.append(clean_match)
                                logger.info(f"상세주소 발견: {clean_match}")
                
                elif keyword in ['단지', '아파트', 'APT', '빌딩', '빌라']:
                    # 기본 주소에 이미 건물명이 있으면 추가하지 않음
                    if base_building_names:
                        logger.info(f"기본 주소에 이미 건물명이 있어 '{text}' 건너뛰기: {base_building_names}")
                        continue
                    
                    # 단지/아파트/빌딩 이름 패턴
                    pattern = rf'[가-힣a-zA-Z0-9\s]*{keyword}'
                    matches = re.findall(pattern, text)
                    if matches:
                        for match in matches:
                            clean_match = match.strip()
                            
                            # 중복 체크: 이미 found_details에 유사한 건물명이 있는지 확인
                            should_add = True
                            for existing_detail in found_details:
                                if any(bld in existing_detail for bld in ['아파트', 'APT', '빌딩', '빌라', '단지']):
                                    if (clean_match == existing_detail or 
                                        clean_match in existing_detail or 
                                        existing_detail in clean_match):
                                        should_add = False
                                        logger.info(f"중복 건물명 제외: '{clean_match}' (이미 추가된: '{existing_detail}')")
                                        break
                            
                            # 너무 긴 매치는 제외 (오인식 방지)
                            if should_add and 2 <= len(clean_match) <= 20:
                                found_details.append(clean_match)
                                logger.info(f"건물명 발견: {clean_match}")
        
        # 텍스트 전체가 호수인 경우도 확인 (예: "1405호")
        if re.match(r'^\d+호$', text.strip()):
            clean_match = text.strip()
            if clean_match not in found_details:
                found_details.append(clean_match)
                logger.info(f"호수 텍스트 발견: {clean_match}")
        
        # 텍스트 전체가 동수인 경우도 확인 (예: "305동")
        if re.match(r'^\d+동$', text.strip()):
            clean_match = text.strip()
            if clean_match not in found_details:
                found_details.append(clean_match)
                logger.info(f"동수 텍스트 발견: {clean_match}")
    
    logger.info(f"발견된 상세주소들: {found_details}")
    
    # 기본 주소에서 동/호수만 제거 (시/군/구는 그대로 유지)
    clean_base_address = base_address
    
    # 기본 주소에서 동/호수 패턴만 제거
    dong_ho_patterns = [
        r'\s*\d+동\s*',     # 305동
        r'\s*\d+호\s*',     # 1405호
        r'\s*\d+층\s*',     # 15층
        r'\s*\d+[-]\d+호\s*' # 101-1405호
    ]
    
    for pattern in dong_ho_patterns:
        matches = re.findall(pattern, clean_base_address)
        if matches:
            for match in matches:
                # 매치된 동/호수를 found_details에 추가 (아직 없다면)
                clean_match = match.strip()
                if clean_match and clean_match not in found_details:
                    found_details.append(clean_match)
                    logger.info(f"기본 주소에서 추출한 상세주소: {clean_match}")
                
                # 기본 주소에서 제거
                clean_base_address = clean_base_address.replace(match, ' ')
    
    # 공백 정리
    clean_base_address = re.sub(r'\s+', ' ', clean_base_address).strip()
    clean_base_address = clean_base_address.replace(' ,', ',').replace('  ', ' ')
    
    logger.info(f"정리된 기본 주소: {clean_base_address}")
    
    # 발견된 상세주소들을 적절한 순서로 정렬
    if found_details:
        # 상세주소를 카테고리별로 분류
        building_info = []  # 아파트, 빌딩명 등
        detail_info = []    # 동, 호, 층
        
        detail_order = ['동', '호', '층']
        
        for detail in found_details:
            if any(keyword in detail for keyword in ['아파트', 'APT', '빌딩', '빌라', '단지']):
                # 기본 주소에 이미 건물명이 있으면 추가하지 않음
                if not base_building_names:
                    building_info.append(detail)
                else:
                    logger.info(f"기본 주소에 건물명이 있어 '{detail}' 제외")
            else:
                if detail not in detail_info:
                    detail_info.append(detail)
        
        # 동, 호, 층 순서로 정렬
        sorted_detail_info = []
        for order_keyword in detail_order:
            for detail in detail_info:
                if order_keyword in detail and detail not in sorted_detail_info:
                    sorted_detail_info.append(detail)
        
        # 나머지 상세정보 추가
        for detail in detail_info:
            if detail not in sorted_detail_info:
                sorted_detail_info.append(detail)
        
        # 최종 주소 구성: 기본주소(시/군/구 포함) + 건물정보 + 상세정보(동,호,층)
        address_parts = [clean_base_address]
        
        if building_info:
            address_parts.extend(building_info)
        
        if sorted_detail_info:
            address_parts.extend(sorted_detail_info)
        
        complete_address = ' '.join(address_parts)
        logger.info(f"완전한 주소 구성: {clean_base_address} + 건물정보{building_info} + 상세정보{sorted_detail_info}")
    else:
        complete_address = clean_base_address
    
    return complete_address.strip()

@app.get("/")
async def root():
    return {"message": "OCR API 서버가 실행 중입니다"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "reader_loaded": reader is not None}

@app.post("/ocr")
async def extract_text(image: UploadFile = File(...)):
    """
    이미지에서 텍스트를 추출합니다.
    
    - **image**: 업로드할 이미지 파일 (JPG, PNG 등)
    """
    try:
        # 파일 타입 검증
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
        
        # 이미지 읽기
        image_data = await image.read()
        
        # PIL Image로 변환
        pil_image = Image.open(io.BytesIO(image_data))
        
        # RGB로 변환 (RGBA나 다른 모드일 경우)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # numpy 배열로 변환
        image_array = np.array(pil_image)
        
        # 이미지 최적화
        image_array = optimize_image_for_ocr(image_array)
        
        logger.info(f"이미지 크기: {image_array.shape}")
        
        # OCR 실행 (빠른 모드)
        results = reader.readtext(
            image_array,
            paragraph=False,  # 단락 모드 비활성화로 속도 향상
            detail=1,         # 상세 정보 포함
            batch_size=1      # 배치 크기 1로 설정
        )
        
        # 결과 처리
        extracted_texts = []
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # 신뢰도 30% 이상
                extracted_texts.append({
                    'text': text.strip(),
                    'confidence': round(confidence, 3),
                    'bbox': [[int(point[0]), int(point[1])] for point in bbox]
                })
        
        logger.info(f"추출된 텍스트 개수: {len(extracted_texts)}")
        
        return JSONResponse(content={
            'success': True,
            'total_texts': len(extracted_texts),
            'texts': extracted_texts,
            'message': f'{len(extracted_texts)}개의 텍스트가 추출되었습니다'
        })
        
    except Exception as e:
        logger.error(f"OCR 처리 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR 처리 실패: {str(e)}")

@app.post("/ocr/delivery")
async def extract_delivery_info(image: UploadFile = File(...)):
    """
    운송장에서 특정 정보를 추출합니다.
    
    - **image**: 운송장 이미지 파일
    """
    try:
        # 이미지 읽기 및 PIL → numpy 변환
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        image_array = np.array(pil_image)

        # 이미지 최적화
        image_array = optimize_image_for_delivery(image_array)

        # 운송장 전용 전처리 적용
        processed_img = delivery_preprocessing(image_array)

        # OCR 실행 (운송장용 최적화)
        results = reader.readtext(
            processed_img,
            paragraph=False,
            detail=1,
            batch_size=1
        )
        
        # 운송장 관련 정보 추출
        delivery_info = {
            'tracking_number': None,
            'recipient': None,
            'phone': None,
            'address': None,
            'all_texts': []
        }
        
        # 주소 후보들을 저장할 리스트
        address_candidates = []
        tracking_number_candidates = []
        
        for (bbox, text, confidence) in results:
            if confidence > 0.3:
                clean_text = text.strip()
                
                delivery_info['all_texts'].append({
                    'text': clean_text,
                    'confidence': round(confidence, 3)
                })
                
                # 운송장 번호 패턴을 모든 텍스트에서 찾기 (키워드 관계없이)
                # 하이픈이 포함된 패턴 먼저 찾기
                hyphen_patterns = [
                    r'\d{4}-\d{4}-\d{4}',  # 0000-0000-0000
                ]
                
                # 운송장 번호 키워드 (우선순위 부여용)
                tracking_keywords = ['운송장번호', '운송장 번호', '송장번호', '송장 번호', '배송번호', '배송 번호', '운송번호', '운송 번호']
                has_keyword = any(keyword in clean_text for keyword in tracking_keywords)
                
                # 하이픈 패턴 찾기
                for pattern in hyphen_patterns:
                    matches = re.findall(pattern, clean_text)
                    if matches:
                        for match in matches:
                            # 0으로 시작하지 않는 운송장 번호만 선택
                            if not match.startswith('0'):
                                tracking_number_candidates.append({
                                    'text': match,
                                    'confidence': confidence,
                                    'length': len(match),
                                    'type': 'hyphen_pattern',
                                    'has_keyword': has_keyword  # 키워드 포함 여부로 우선순위 결정
                                })
                                logger.info(f"운송장 번호 후보 발견 (하이픈 패턴): {match} {'(키워드 포함)' if has_keyword else ''}")
                
                # 하이픈 없는 긴 숫자 패턴도 찾기 (10자리 이상)
                long_number_patterns = [
                    r'\d{12,}',  # 12자리 이상
                    r'\d{11}',   # 11자리
                    r'\d{10}',   # 10자리
                ]
                
                for pattern in long_number_patterns:
                    matches = re.findall(pattern, clean_text)
                    if matches:
                        for match in matches:
                            # 0으로 시작하지 않는 번호만 선택 (전화번호, 우편번호 등 제외)
                            if not match.startswith('0'):
                                tracking_number_candidates.append({
                                    'text': match,
                                    'confidence': confidence,
                                    'length': len(match),
                                    'type': 'number_only',
                                    'has_keyword': has_keyword
                                })
                                logger.info(f"운송장 번호 후보 발견 (숫자만): {match} {'(키워드 포함)' if has_keyword else ''}")
                
                # 전화번호 패턴
                phone_pattern = r'0\d{1,2}-?\d{3,4}-?\d{4}'
                if re.search(phone_pattern, clean_text):
                    delivery_info['phone'] = clean_text
                
                # 주소 추출 - 여러 키워드가 포함된 텍스트 우선 선택
                address_keywords = [
                    # 행정구역
                    '도', '시', '구', '군', '읍', '면', '동', '리',
                    # 도로명
                    '로', '길', '번길', '가길',
                    # 번지/번호
                    '번지', '번', '호', '층', '가',
                    # 건물유형
                    '아파트', 'APT', '빌라', '빌딩', '오피스텔', '단지', '타워', '플라자',
                    # 지역명
                    '경기', '서울', '인천', '부산', '대전', '대구', '광주', '울산', '세종',
                    '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'
                ]
                
                # 텍스트에 포함된 주소 키워드 개수 계산
                keyword_count = sum(1 for keyword in address_keywords if keyword in clean_text)
                
                # 주소 키워드가 2개 이상 포함된 텍스트를 후보로 저장
                if keyword_count >= 2:
                    address_candidates.append({
                        'text': clean_text,
                        'confidence': confidence,
                        'length': len(clean_text),
                        'keyword_count': keyword_count
                    })
                    logger.info(f"주소 후보 발견 (키워드 {keyword_count}개): {clean_text}")
                
                # 키워드가 1개만 있는 경우도 후보로 저장 (우선순위 낮음)
                elif keyword_count == 1:
                    address_candidates.append({
                        'text': clean_text,
                        'confidence': confidence,
                        'length': len(clean_text),
                        'keyword_count': keyword_count
                    })
                    logger.info(f"주소 후보 발견 (키워드 1개): {clean_text}")
        
        # 운송장 번호 후보들 중에서 최적의 번호 선택
        if tracking_number_candidates:
            # 키워드 포함 > 하이픈 패턴 > 길이 > 신뢰도 순으로 우선순위 결정
            best_tracking = max(tracking_number_candidates, 
                              key=lambda x: (x['has_keyword'], x['type'] == 'hyphen_pattern', x['length'], x['confidence']))
            delivery_info['tracking_number'] = best_tracking['text']
            keyword_status = "(키워드 포함)" if best_tracking['has_keyword'] else "(패턴 기반)"
            logger.info(f"선택된 운송장 번호 ({best_tracking['type']}) {keyword_status}: {best_tracking['text']}")
        else:
            logger.warning("운송장 번호를 찾을 수 없습니다.")
        
        # 주소 후보들 중에서 최적의 주소 선택 및 완전한 주소 구성
        if address_candidates:
            # 완전한 주소 추출 (도로명주소 + 상세주소)
            complete_address = extract_complete_address(delivery_info['all_texts'], address_candidates)
            
            if complete_address:
                # 주소 후처리 적용 (지역명 이전 내용 제거)
                cleaned_address = clean_address(complete_address)
                delivery_info['address'] = cleaned_address
                
                logger.info(f"완전한 주소 추출: {complete_address}")
                if complete_address != cleaned_address:
                    logger.info(f"정리된 주소: {cleaned_address}")
            else:
                logger.warning("완전한 주소를 구성할 수 없습니다.")
        else:
            logger.warning("주소를 찾을 수 없습니다.")
        
        return JSONResponse(content={
            'success': True,
            'delivery_info': delivery_info,
            'message': '운송장 정보 추출 완료'
        })
        
    except Exception as e:
        logger.error(f"운송장 처리 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"운송장 처리 실패: {str(e)}")



def normalize_address(address):
    """주소 정규화 함수 (공백, 특수문자 제거)"""
    if not address:
        return ""
    # 공백, 쉼표, 특수문자 제거 후 소문자로 변환
    normalized = re.sub(r'[,\s\-\|]+', '', address.lower())
    return normalized

def calculate_similarity(str1, str2):
    """두 문자열의 유사도 계산"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, str1, str2).ratio()

@app.get("/orders/{order_id}/verify-address")
async def verify_address(order_id: str, ocr_address: str, db: Session = Depends(get_db)):
    """주문번호의 주소와 OCR 주소를 비교하여 검증"""
    try:
        # 데이터베이스에서 주문 찾기
        order = db.query(OutboundOrder).filter(OutboundOrder.id == order_id).first()
        
        if not order:
            return {
                "success": False,
                "status": "order_not_found",
                "message": "주문번호를 찾을 수 없습니다."
            }
        
        db_address = order.address
        normalized_db_address = normalize_address(db_address)
        normalized_ocr_address = normalize_address(ocr_address)
        
        logger.info(f"주소 비교 - DB: '{db_address}' -> '{normalized_db_address}'")
        logger.info(f"주소 비교 - OCR: '{ocr_address}' -> '{normalized_ocr_address}'")
        
        # 주소 일치 여부 확인 (80% 이상 일치하면 같은 주소로 판단)
        similarity = calculate_similarity(normalized_db_address, normalized_ocr_address)
        is_address_match = similarity >= 0.8  # 80% 이상 일치
        
        logger.info(f"주소 유사도: {similarity:.2f}, 일치 여부: {is_address_match}")
        
        # 상태별 메시지 결정
        if is_address_match:
            # 주소가 일치하는 경우 - 주문 상태 확인
            if order.status == "취소":
                return {
                    "success": True,
                    "status": "canceled",
                    "message": "취소된 주문입니다. 출고할 수 없습니다.",
                    "order_status": order.status,
                    "db_address": db_address,
                    "ocr_address": ocr_address,
                    "similarity": round(similarity, 2)
                }
            elif order.status in ["출고 완료", "배송 중", "배송 완료"]:
                return {
                    "success": True,
                    "status": "already_shipped",
                    "message": "이미 발송된 주소입니다 주문을 확인해주세요",
                    "order_status": order.status,
                    "db_address": db_address,
                    "ocr_address": ocr_address,
                    "similarity": round(similarity, 2)
                }
            else:
                return {
                    "success": True,
                    "status": "ready_to_ship",
                    "message": "정상적으로 출고가 가능합니다.",
                    "order_status": order.status,
                    "db_address": db_address,
                    "ocr_address": ocr_address,
                    "similarity": round(similarity, 2)
                }
        else:
            # 주소가 일치하지 않는 경우
            return {
                "success": True,
                "status": "address_mismatch",
                "message": "입력된 주소와 주문번호가 일치하지 않습니다. 주소를 확인해주세요",
                "order_status": order.status,
                "db_address": db_address,
                "ocr_address": ocr_address,
                "similarity": round(similarity, 2)
            }
            
    except Exception as e:
        logger.error(f"주소 검증 중 오류: {e}")
        return {
            "success": False,
            "status": "error",
            "message": f"주소 검증 중 오류가 발생했습니다: {str(e)}"
        }

@app.patch("/orders/{order_id}/outbound")
async def outbound_order(order_id: str, tracking_number: str = None, db: Session = Depends(get_db)):
    """출고 처리 - OCR로 인식된 운송장 번호 업데이트"""
    try:
        # 데이터베이스에서 주문 찾기
        order = db.query(OutboundOrder).filter(OutboundOrder.id == order_id).first()
        
        if not order:
            return {
                "success": False,
                "status": "order_not_found",
                "message": "주문번호를 찾을 수 없습니다."
            }
        
        # 출고 상태 업데이트
        order.status = "출고 완료"
        order.outbounddate = datetime.now()
        
        # OCR로 인식된 운송장 번호가 있으면 업데이트
        if tracking_number and tracking_number.strip() and tracking_number != '운송장을 추출할 수 없습니다.':
            order.tracking_number = tracking_number.strip()
            logger.info(f"운송장 번호 업데이트: {order_id} -> {tracking_number}")
        else:
            # 운송장 번호가 없으면 기본값 설정
            order.tracking_number = f"AUTO-{order_id[-3:]}"
            logger.info(f"자동 운송장 번호 생성: {order_id} -> AUTO-{order_id[-3:]}")
        
        # 데이터베이스에 저장
        db.commit()
        
        return {
            "success": True,
            "status": "success",
            "message": "출고 처리가 완료되었습니다.",
            "updated_order": {
                "id": order.id,
                "status": order.status,
                "outbounddate": order.outbounddate.strftime("%Y-%m-%d %H:%M:%S"),
                "tracking_number": order.tracking_number
            }
        }
        
    except Exception as e:
        logger.error(f"출고 처리 중 오류: {e}")
        db.rollback()
        return {
            "success": False,
            "status": "error",
            "message": f"출고 처리 중 오류가 발생했습니다: {str(e)}"
        }

# 주문 목록 조회 API 추가
@app.get("/orders")
async def get_orders(
    status: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """주문 목록 조회 (필터링 지원)"""
    try:
        # 기본 쿼리
        query = db.query(OutboundOrder)
        
        # 상태 필터링
        if status and status != "전체":
            query = query.filter(OutboundOrder.status == status)
        
        # 검색 필터링 (주문번호 또는 수령인)
        if search:
            query = query.filter(
                (OutboundOrder.id.contains(search)) |
                (OutboundOrder.recipient.contains(search))
            )
        
        # 주문일 기준 내림차순 정렬
        orders = query.order_by(OutboundOrder.orderdate.desc()).all()
        
        # 응답 데이터 구성
        order_list = []
        for order in orders:
            order_list.append({
                "id": order.id,
                "orderdate": order.orderdate.strftime("%Y-%m-%d") if order.orderdate else "",
                "outbounddate": order.outbounddate.strftime("%Y-%m-%d") if order.outbounddate else "",
                "recipient": order.recipient,
                "phone": order.phone,
                "address": order.address,
                "trackingNumber": order.tracking_number or "",
                "status": order.status,
                "items": order.items if order.items else []
            })
        
        return {
            "success": True,
            "orders": order_list,
            "total": len(order_list),
            "message": f"{len(order_list)}개의 주문을 조회했습니다."
        }
        
    except Exception as e:
        logger.error(f"주문 목록 조회 중 오류: {e}")
        return {
            "success": False,
            "orders": [],
            "total": 0,
            "message": f"주문 목록 조회 중 오류가 발생했습니다: {str(e)}"
        }

# 특정 주문 상세 조회 API 추가
@app.get("/orders/{order_id}")
async def get_order_detail(order_id: str, db: Session = Depends(get_db)):
    """특정 주문 상세 조회"""
    try:
        order = db.query(OutboundOrder).filter(OutboundOrder.id == order_id).first()
        
        if not order:
            return {
                "success": False,
                "order": None,
                "message": "주문번호를 찾을 수 없습니다."
            }
        
        order_detail = {
            "id": order.id,
            "orderdate": order.orderdate.strftime("%Y-%m-%d %H:%M:%S") if order.orderdate else "",
            "outbounddate": order.outbounddate.strftime("%Y-%m-%d %H:%M:%S") if order.outbounddate else "",
            "recipient": order.recipient,
            "phone": order.phone,
            "address": order.address,
            "trackingNumber": order.tracking_number or "",
            "status": order.status,
            "items": order.items if order.items else []
        }
        
        return {
            "success": True,
            "order": order_detail,
            "message": "주문 상세 정보를 조회했습니다."
        }
        
    except Exception as e:
        logger.error(f"주문 상세 조회 중 오류: {e}")
        return {
            "success": False,
            "order": None,
            "message": f"주문 상세 조회 중 오류가 발생했습니다: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile="../frontend/mkcert/key.pem",
        ssl_certfile="../frontend/mkcert/cert.pem"
    )