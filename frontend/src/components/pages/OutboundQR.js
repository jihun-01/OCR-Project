import React, { useRef, useState } from 'react';
import qrImage from '../../assets/QR.png';
import { Html5Qrcode } from 'html5-qrcode';
import { useNavigate } from 'react-router-dom';
import { useOrder } from '../../context/OrderContext';

const OutboundQR = ({ onScanResult }) => {
  const [scanning, setScanning] = useState(false);
  const qrRegionId = 'qr-reader-region';
  const html5QrCodeRef = useRef(null);
  const navigate = useNavigate();
  const { setOrderId } = useOrder();
  // QR 버튼 클릭 시 카메라 스캔 시작
  const handleScanClick = () => {
    setScanning(true);
    setTimeout(() => {
      if (!html5QrCodeRef.current) {
        html5QrCodeRef.current = new Html5Qrcode(qrRegionId);
      }
      html5QrCodeRef.current.start(
        { facingMode: 'environment' },
        {
          fps: 10,
          qrbox: 250,
        },
        (decodedText) => {
          setScanning(false);
          html5QrCodeRef.current.stop();
          setOrderId(decodedText);
          navigate('/ocr');
        },
        (errorMessage) => {
          // 인식 실패시 무시
        }
      );
    }, 100); // region 렌더링 대기
  };

  // 스캔 취소
  const handleCancel = () => {
    setScanning(false);
    if (html5QrCodeRef.current) {
      html5QrCodeRef.current.stop();
    }
  };

  return (
    <div className="dark:bg-gray-800 max-w-screen-sm mx-auto flex flex-col min-h-[calc(100vh-56px)] bg-gray-100 px-4 py-8">
      <div className="w-full h-full dark:bg-gray-600 bg-white rounded-2xl shadow-lg p-10 flex flex-col items-center">
        <p className="dark:text-white text-gray-700 text-lg font-semibold mb-8 mt-2 text-center">
          주문번호를 스캔해 주세요
        </p>
        {!scanning ? (
          <button
            className="dark:bg-gray-500 bg-gray-100 rounded-2xl p-8 shadow-md hover:bg-gray-200 transition flex flex-col items-center"
            onClick={handleScanClick}
            type="button"
          >
            <img src={qrImage} alt="QR 스캔" className="w-96 h-96 object-contain" />
          </button>
        ) : (
          <div className="dark:bg-gray-800 flex flex-col items-center">
            <div id={qrRegionId} className="dark:bg-gray-800 w-80 h-80 bg-black rounded-lg" />
            <button
              className="dark:bg-gray-800 mt-4 px-4 py-2 bg-red-500 text-white rounded"
              onClick={handleCancel}
              type="button"
            >
              취소
            </button>
          </div>
        )}
        <span className="dark:text-white text-sm text-gray-400 mt-6">QR코드를 눌러 스캔하세요</span>
      </div>
    </div>
  );
};

export default OutboundQR;
