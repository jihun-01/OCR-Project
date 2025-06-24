import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useOrder } from '../../context/OrderContext';
import { Navigate } from 'react-router-dom';

const BACKEND_API_URL = process.env.REACT_APP_BACKEND_API_URL;


const OutboundResult = () => {
  const { orderId } = useOrder();
  const location = useLocation();
  const navigate = useNavigate();
  
  // OCR에서 추출된 주소 데이터 가져오기
  const { deliveryInfo } = location.state || {};
  const extractedAddress = deliveryInfo?.address || '주소를 추출할 수 없습니다.';
  const extractedDeliveryNumber = deliveryInfo?.tracking_number || '운송장을 추출할 수 없습니다.';

  // 상태 관리
  const [verificationResult, setVerificationResult] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);



  // 주소 검증 함수
  const verifyAddress = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(
        `${BACKEND_API_URL}/orders/${encodeURIComponent(orderId)}/verify-address?ocr_address=${encodeURIComponent(extractedAddress)}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('주소 검증 결과:', result);
      setVerificationResult(result);

    } catch (err) {
      console.error('주소 검증 오류:', err);
      setError('주소 검증 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  // 컴포넌트 마운트 시 주소 검증 실행
  useEffect(() => {
    if (orderId && extractedAddress && extractedAddress !== '주소를 추출할 수 없습니다.') {
      verifyAddress();
    } else {
      setIsLoading(false);
      setError('주문번호 또는 주소 정보가 없습니다.');
    }
  }, [orderId, extractedAddress]);

  if (!orderId) {
    return <Navigate to="/404" replace />;
  }
  
  // 메시지 스타일 결정
  const getMessageStyle = () => {
    if (!verificationResult) return "bg-gray-100 dark:bg-gray-700";
    
    switch (verificationResult.status) {
      case "ready_to_ship":
        return "bg-green-100 dark:bg-green-900";
      case "already_shipped":
        return "bg-yellow-100 dark:bg-yellow-900";
      case "address_mismatch":
        return "bg-red-100 dark:bg-red-900";
      case "order_mismatch":
        return "bg-red-100 dark:bg-red-900";
      case "canceled":
        return "bg-red-100 dark:bg-red-900";
      default:
        return "bg-gray-100 dark:bg-gray-700";
    }
  };

  // 메시지 텍스트 결정
  const getMessage = () => {
    if (isLoading) return "주소를 검증하는 중입니다...";
    if (error) return error;
    if (!verificationResult) return "주소 검증을 실행할 수 없습니다.";
    
    return verificationResult.message;
  };

  // 출고 확정 버튼 활성화 여부
  const isShipButtonEnabled = () => {
    return verificationResult && verificationResult.status === "ready_to_ship";
  };

  // 출고 확정 처리
  const handleShipConfirm = async () => {
    if (!isShipButtonEnabled()) return;

    try {
      setIsLoading(true);
      
      // 출고 확정 API 호출 (운송장 번호 포함)
      const response = await fetch(
        `${BACKEND_API_URL}/orders/${encodeURIComponent(orderId)}/outbound?tracking_number=${encodeURIComponent(extractedDeliveryNumber)}`,
        {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('출고 확정 결과:', result);
      
      if (result.success) {
        alert('출고 처리가 완료되었습니다.');
        navigate('/');
      } else {
        throw new Error(result.message || '출고 처리에 실패했습니다.');
      }
      
    } catch (err) {
      console.error('출고 확정 오류:', err);
      setError('출고 확정 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="dark:bg-gray-800 max-w-screen-sm mx-auto flex flex-col min-h-[calc(100vh-56px)] bg-gray-100 px-4 py-8">
      <div className="w-full h-full dark:bg-gray-600 bg-white rounded-2xl shadow-lg p-10 flex flex-col items-center">
        <div className="w-full h-12 dark:bg-gray-700 dark:text-white bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col text-2xl items-start justify-center mb-8">
          주문번호 : {orderId}
        </div>
        
        <div className="w-full h-12 dark:bg-gray-700 dark:text-white bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col text-base items-start justify-center mb-8">
          운송장 번호 : {extractedDeliveryNumber}
        </div>
        
        <div className="w-full h-12 dark:bg-gray-700 dark:text-white bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col text-base items-start justify-center mb-8">
          수령인 : {deliveryInfo?.recipient || '정보 없음'}
        </div>
        
        <div className="w-full h-32 dark:bg-gray-700 dark:text-white bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col text-base items-start justify-start flex-wrap mb-8">
          <div className="font-semibold mb-2">OCR 추출 주소:</div>
          <div>{extractedAddress}</div>
        </div>

        {/* DB 주소 표시 (검증 결과가 있을 때만) */}
        {verificationResult && verificationResult.db_address && (
          <div className="w-full h-32 dark:bg-gray-700 dark:text-white bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col text-base items-start justify-start flex-wrap mb-8">
            <div className="font-semibold mb-2">DB 등록 주소:</div>
            <div>{verificationResult.db_address}</div>
            {verificationResult.similarity && (
              <div className="text-sm text-gray-500 mt-2">
                유사도: {(verificationResult.similarity * 100).toFixed(1)}%
              </div>
            )}
          </div>
        )}
        
        {/* 검증 결과 메시지 */}
        <div className={`w-full h-28 dark:text-white ${getMessageStyle()} rounded-2xl p-4 shadow-md transition flex flex-col text-xl items-center justify-center mb-8`}>
          {getMessage()}
        </div>
        
        {/* 버튼들 */}
        <div className="flex flex-row justify-between w-80">
          <button 
            onClick={() => navigate('/')}
            className="w-36 h-16 bg-gray-100 rounded-full p-4 shadow-md hover:bg-gray-200 transition flex flex-col text-2xl items-center justify-center mb-8"
          >
            취소
          </button>
          <button 
            onClick={handleShipConfirm}
            disabled={!isShipButtonEnabled()}
            className={`w-36 h-16 rounded-full p-4 shadow-md transition flex flex-col text-2xl items-center justify-center mb-8 ${
              isShipButtonEnabled()
                ? 'bg-blue-500 text-white hover:bg-blue-600'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}
          >
            출고 확정
          </button>
        </div>
      </div>
    </div>
  );
};

export default OutboundResult;
