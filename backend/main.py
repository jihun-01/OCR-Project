from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import easyocr
import numpy as np
from PIL import Image
import io
import logging

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
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
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
    allow_origins=["http://localhost:3000"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        logger.info(f"이미지 크기: {image_array.shape}")
        
        # OCR 실행
        results = reader.readtext(image_array)
        
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
        # 기본 OCR 실행
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        image_array = np.array(pil_image)
        results = reader.readtext(image_array)
        
        # 운송장 관련 정보 추출
        delivery_info = {
            'tracking_number': None,
            'recipient': None,
            'phone': None,
            'address': None,
            'all_texts': []
        }
        
        for (bbox, text, confidence) in results:
            if confidence > 0.3:
                clean_text = text.strip()
                delivery_info['all_texts'].append({
                    'text': clean_text,
                    'confidence': round(confidence, 3)
                })
                
                # 운송장 번호 패턴 (숫자 10자리 이상)
                import re
                if re.search(r'\d{10,}', clean_text):
                    delivery_info['tracking_number'] = clean_text
                
                # 전화번호 패턴
                phone_pattern = r'0\d{1,2}-?\d{3,4}-?\d{4}'
                if re.search(phone_pattern, clean_text):
                    delivery_info['phone'] = clean_text
                
                # 주소 키워드
                if any(keyword in clean_text for keyword in ['시', '구', '동', '로', '길']):
                    delivery_info['address'] = clean_text
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)