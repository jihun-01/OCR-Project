import React, { useRef } from 'react';
import qrImage from '../../assets/upload.png';
import { useLocation } from 'react-router-dom';


const OutboundOCR = ({ onImageSelect }) => {
  const fileInputRef = useRef(null);
  const { decodedText } = useLocation().state;

  // 버튼 클릭 시 input 클릭 트리거
  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  // 파일 선택(또는 촬영) 시 콜백
  const handleFileChange = (e) => {
    if (onImageSelect) {
      onImageSelect(e.target.files[0]);
    }
  };

  return (
    <div className="max-w-screen-sm mx-auto flex flex-col min-h-[calc(100vh-56px)] bg-gray-100 px-4 py-8">
      <div className="w-full h-full bg-white rounded-2xl shadow-lg p-10 flex flex-col items-center">
        <div className="w-full bg-gray-100 rounded-2xl p-4 shadow-md hover:bg-gray-200 transition flex flex-col items-center text-2xl items-center justify-center mb-8">
          주문번호 : {decodedText}
        </div>
        <p className="text-gray-700 text-lg font-semibold mb-8 mt-2 text-center">
          운송장을 업로드 해 주세요
        </p>
        <button
          className="bg-gray-100 rounded-2xl p-8 shadow-md hover:bg-gray-200 transition flex flex-col items-center"
          onClick={handleButtonClick}
          type="button"
        >
          <img src={qrImage} alt="QR 스캔" className="w-96 h-96 object-contain" />
        </button>
        <input
          type="file"
          accept="image/*"
          capture="environment"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
        />
        <span className="text-sm text-gray-400 mt-6">업로드 버튼을 눌러 스캔하세요</span>
      </div>
    </div>
  );
};

export default OutboundOCR;