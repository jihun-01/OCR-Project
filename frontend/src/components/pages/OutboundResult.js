import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useOrder } from '../../context/OrderContext';
import { mockData } from './WmsOutboundstate';

const OutboundResult = () => {
  const { orderId } = useOrder();
  const location = useLocation();
  const navigate = useNavigate();
  
  // OCR에서 추출된 주소 데이터 가져오기
  const { deliveryInfo } = location.state || {};
  const extractedAddress = deliveryInfo?.address || '주소를 추출할 수 없습니다.';
  const extractedDeliveryNumber = deliveryInfo?.tracking_number || '운송장을 추출할 수 없습니다.';

  return (
    <div className="dark:bg-gray-800 max-w-screen-sm mx-auto flex flex-col min-h-[calc(100vh-56px)] bg-gray-100 px-4 py-8">
      <div className="w-full h-full dark:bg-gray-600 bg-white rounded-2xl shadow-lg p-10 flex flex-col items-center">
        <div className="w-full dark:bg-gray-700 dark:text-white bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-2xl items-start justify-center mb-8">
          주문번호 : {orderId}
        </div>
        <div className="w-full dark:bg-gray-700 dark:text-white bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-base items-start justify-center mb-8">
          운송장 번호 : {extractedDeliveryNumber}
        </div>
        <div className="w-full dark:bg-gray-700 dark:text-white bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-base items-start justify-center mb-8">
          수령인 :
        </div>
        <div className="w-full h-32 dark:bg-gray-700 dark:text-white bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-2 items-start justify-start flex-wrap mb-8">
          주소 : {extractedAddress}
        </div> 
        <div className="w-full h-36 dark:bg-green-900 dark:text-white bg-green-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-2xl items-center justify-center mb-8">
          정상적으로 출고가 가능합니다.
        </div>
        <div className="flex flex-row justify-between w-80">
          <button 
            onClick={() => navigate('/')}
            className="w-36 h-16 bg-gray-100 rounded-full p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-2xl items-center justify-center mb-8"
          >
            취소
          </button>
          <button 
            onClick={() => navigate('/')}
            className="w-36 h-16 bg-blue-500 text-white rounded-full p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-2xl items-center justify-center mb-8"
          >
            출고 확정
          </button>
        </div>
      </div>
    </div>
  );
};

export default OutboundResult;
