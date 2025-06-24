import React, { useRef, useState, useEffect } from 'react';
import qrImage from '../../assets/upload.png';
import { Navigate, useNavigate } from 'react-router-dom';
import { useOrder } from '../../context/OrderContext';

// API URL 동적 설정 - 현재 접속한 기기의 IP 사용
const getApiUrl = () => {
  const hostname = window.location.hostname;
  const port = 8000;
  
  // localhost인 경우 (PC에서 개발 시)
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'https://localhost:8000';
  }
  
  // 실제 IP인 경우 (모바일에서 접속 시)
  return `https://${hostname}:${port}`;
};

const API_BASE_URL = getApiUrl();

const OutboundOCR = ({ onImageSelect }) => {
  const fileInputRef = useRef(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState('');
  const { orderId } = useOrder();
  const navigate = useNavigate();

  // 서버 상태 확인 함수
  const checkServerStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const result = await response.json();
      console.log('서버 상태:', result);
    } catch (err) {
      console.warn('서버 상태 확인 실패:', err);
    }
  };

  // useEffect를 조건문 밖으로 이동
  useEffect(() => {
    checkServerStatus();
  }, []);

  // location.state가 존재하지 않으면 404 페이지로 이동
  if (!orderId) {
    return <Navigate to="/404" replace />;
  }

  const decodedText = orderId;

  // OCR API 호출 함수 (FastAPI)
  const processImageWithOCR = async (imageFile) => {
    setIsProcessing(true);
    setError(null);
    setProgress('이미지 업로드 중...');
    
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      
      setProgress('텍스트 추출 중...');
      
      console.log('요청 URL:', `${API_BASE_URL}/ocr/delivery`);
      console.log('FormData 내용:', formData);
      
      const response = await fetch(`${API_BASE_URL}/ocr/delivery`, {
        method: 'POST',
        body: formData,
      });
      
      console.log('응답 상태:', response.status);
      console.log('응답 헤더:', response.headers);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('서버 응답 에러:', errorText);
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }
      
      const result = await response.json();
      console.log('OCR 결과:', result);
      
      setProgress('결과 처리 중...');
      
      if (result.success) {
        // OCR 결과를 결과 페이지로 전달
        navigate('/result', {
          state: {
            orderNumber: decodedText,
            ocrResults: result.delivery_info.all_texts,
            deliveryInfo: result.delivery_info,
            originalImage: URL.createObjectURL(imageFile),
            message: result.message
          }
        });
      } else {
        setError('텍스트 추출에 실패했습니다: ' + result.message);
      }
    } catch (err) {
      console.error('상세 에러 정보:', err);
      console.error('에러 타입:', err.name);
      console.error('에러 메시지:', err.message);
      console.error('에러 스택:', err.stack);
      
      // 더 구체적인 에러 메시지 설정
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        setError('네트워크 연결에 실패했습니다. 서버가 실행 중인지 확인하세요.');
      } else if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
        setError('서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인하세요.');
      } else if (err.message.includes('CORS')) {
        setError('CORS 오류가 발생했습니다. 서버 설정을 확인하세요.');
      } else if (err.message.includes('SSL') || err.message.includes('certificate')) {
        setError('SSL 인증서 오류가 발생했습니다. 브라우저에서 인증서를 신뢰하도록 설정하세요.');
      } else {
        setError('OCR 처리 중 오류가 발생했습니다: ' + err.message);
      }
    } finally {
      setIsProcessing(false);
      setProgress('');
    }
  };

  // 버튼 클릭 시 input 클릭 트리거
  const handleButtonClick = () => {
    if (!isProcessing) {
      fileInputRef.current.click();
    }
  };

  // 파일 선택 시 OCR 처리
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // 파일 크기 확인 (10MB 제한)
      if (file.size > 10 * 1024 * 1024) {
        setError('파일 크기가 너무 큽니다. 10MB 이하의 이미지를 업로드하세요.');
        return;
      }
      
      processImageWithOCR(file);
      
      if (onImageSelect) {
        onImageSelect(file);
      }
    }
  };

  return (
    <div className="dark:bg-gray-800 max-w-screen-sm mx-auto flex flex-col min-h-[calc(100vh-56px)] bg-gray-100 px-4 py-8">
      <div className="w-full h-full dark:bg-gray-600 bg-white rounded-2xl shadow-lg p-10 flex flex-col items-center">
        <div className="dark:text-white text-gray-700 w-full bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col text-2xl items-center justify-center mb-8">
          주문번호 : {decodedText}
        </div>
        
        {/* 에러 메시지 표시 */}
        {error && (
          <div className="dark:bg-gray-800 w-full bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <div className="flex">
              <div className="py-1">
                <svg className="fill-current h-6 w-6 text-red-500 mr-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                  <path d="M2.93 17.07A10 10 0 1 1 17.07 2.93 10 10 0 0 1 2.93 17.07zm12.73-1.41A8 8 0 1 0 4.34 4.34a8 8 0 0 0 11.32 11.32zM9 11V9h2v6H9v-4zm0-6h2v2H9V5z"/>
                </svg>
              </div>
              <div className="overflow-hidden">
                <p className="dark:text-white font-bold">오류 발생</p>
                <p className="dark:text-white text-sm overflow-hidden">{error}</p>
              </div>
            </div>
          </div>
        )}
        
        {/* 진행 상태 표시 */}
        {progress && (
          <div className="dark:bg-gray-800 w-full bg-blue-100 border border-blue-400 text-blue-700 px-4 py-3 rounded mb-4">
            <p className="dark:text-white text-center">{progress}</p>
          </div>
        )}
        
        <p className="dark:text-white text-gray-700 text-lg font-semibold mb-8 mt-2 text-center">
          {isProcessing ? 'AI가 운송장을 분석하는 중...' : '운송장을 업로드 해 주세요'}
        </p>
        
        <button
          className={`dark:bg-gray-500 bg-gray-100 rounded-2xl p-8 shadow-md transition flex flex-col items-center ${
            isProcessing ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-200'
          }`}
          onClick={handleButtonClick}
          type="button"
          disabled={isProcessing}
        >
          {isProcessing ? (
            <div className="w-96 h-96 flex flex-col items-center justify-center">
              <div className="animate-spin rounded-full h-32 w-32 border-b-4 border-blue-600 mb-4"></div>
              <p className="dark:text-white text-gray-600 text-center">{progress}</p>
            </div>
          ) : (
            <img src={qrImage} alt="이미지 업로드" className="w-96 h-96 object-contain" />
          )}
        </button>
        
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
          disabled={isProcessing}
        />
        
        <span className="dark:text-white text-sm text-gray-400 mt-6">
          {isProcessing ? 'AI 분석 중입니다...' : '업로드 버튼을 눌러 운송장을 스캔하세요'}
        </span>
        
        <span className="text-xs text-gray-300 mt-2">
          지원 형식: JPG, PNG (최대 10MB)
        </span>
      </div>
    </div>
  );
};

export default OutboundOCR;