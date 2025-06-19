from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import easyocr
import numpy as np
from PIL import Image
import io
import logging
import re
import cv2
import torch

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
    allow_origins=["https://192.168.45.251:3000"],
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

def optimize_image_for_delivery(image_array, min_width=800, max_width=2000, min_height=800, max_height=2000):
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
    
    # 너무 큰 이미지는 축소 (메모리 절약)
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
                            # 전화번호나 기타 숫자가 아닌지 확인
                            if not re.match(r'^0\d+', match):  # 0으로 시작하는 전화번호 제외
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
                address_keywords = ['시', '구', '동', '로', '길', '번길', '번지', '호', '층','경기','서울','인천','부산',
                                    '대전','대구','광주','울산','세종','강원','충청','전라','경상','제주']
                
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
        
        # 주소 후보들 중에서 최적의 주소 선택
        if address_candidates:
            # 키워드 개수 > 길이 > 신뢰도 순으로 우선순위 결정
            best_address = max(address_candidates, 
                             key=lambda x: (x['keyword_count'], x['length'], x['confidence']))
            delivery_info['address'] = best_address['text']
            logger.info(f"선택된 주소 (키워드 {best_address['keyword_count']}개): {best_address['text']}")
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