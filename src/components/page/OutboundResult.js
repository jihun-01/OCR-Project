import React, { useRef, useState } from 'react';


const OutboundResult = () => {
 
  return (
    <div className="max-w-screen-sm mx-auto flex flex-col min-h-[calc(100vh-56px)] bg-gray-100 px-4 py-8">
      <div className="w-full h-full bg-white rounded-2xl shadow-lg p-10 flex flex-col items-center">
        <div className="w-full bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-2xl items-start justify-center mb-8">
          주문번호 :
        </div>
        <div className="w-full bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-2xl items-start justify-center mb-8">
          수령인 :
        </div>
        <div className="w-full h-36 bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-2xl items-start justify-start flex-wrap mb-8">
          주소 :
        </div>
        <div className="w-full h-64 bg-green-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-2xl items-center justify-center mb-8">
          정상적으로 출고가 가능합니다.
        </div>
        <div className="flex flex-row justify-between w-80">
          <button className="w-36 h-16 bg-gray-100 rounded-full p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-2xl items-center justify-center mb-8">
            취소
          </button>
          <button className="w-36 h-16 bg-blue-500 text-white rounded-full p-4 shadow-md hover:bg-gray-200 transition flex flex-col  text-2xl items-center justify-center mb-8">
            출고 확정
          </button>
        </div>
      </div>
    </div>
  );
};

export default OutboundResult;
