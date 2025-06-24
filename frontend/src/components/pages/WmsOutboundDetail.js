import React, { useState, useEffect } from "react";
import { useParams, Link } from 'react-router-dom';

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

const WmsOutboundDetail = ({ id: propId, order: propOrder, onClose, onUpdate }) => {
  const params = useParams();
  const id = propId || params.id;
  
  // 상태 관리
  const [order, setOrder] = useState(propOrder || null);
  const [loading, setLoading] = useState(!propOrder);
  const [error, setError] = useState(null);

  // API에서 주문 상세 정보 가져오기
  const fetchOrderDetail = async () => {
    if (!id) return;
    
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/orders/${id}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setOrder(result.order);
      } else {
        throw new Error(result.message || '주문 정보를 가져올 수 없습니다.');
      }
      
    } catch (err) {
      console.error('주문 상세 조회 오류:', err);
      setError('주문 정보를 불러오는 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  // 컴포넌트 마운트 시 또는 id 변경 시 데이터 로드
  useEffect(() => {
    if (!propOrder && id) {
      fetchOrderDetail();
    }
  }, [id, propOrder]);

  // 날짜 포맷 함수
  const formatDate = (dateStr) => {
    if (!dateStr) return '-';
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('ko-KR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return dateStr;
    }
  };

  // 로딩 상태
  if (loading) {
    return (
      <div className="dark:bg-gray-800 pt-16 min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4 font-sans">
        <div className="max-w-4xl mx-auto text-center">
          <div className="dark:bg-gray-800 bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 p-12">
            <div className="flex justify-center items-center h-32">
              <div className="text-lg text-gray-500 dark:text-gray-400">주문 정보를 불러오는 중...</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 에러 상태
  if (error) {
    return (
      <div className="dark:bg-gray-800 pt-16 min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4 font-sans">
        <div className="max-w-4xl mx-auto text-center">
          <div className="dark:bg-gray-800 bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 p-12">
            <h2 className="dark:text-white text-2xl font-bold text-red-600 mb-4">오류 발생</h2>
            <p className="dark:text-white text-gray-600 mb-6">{error}</p>
            <div className="space-x-4">
              <button 
                onClick={fetchOrderDetail}
                className="bg-blue-600 text-white px-6 py-3 rounded-xl hover:bg-blue-700 transition-colors"
              >
                다시 시도
              </button>
              {onClose ? (
                <button onClick={onClose} className="bg-gray-600 text-white px-6 py-3 rounded-xl hover:bg-gray-700 transition-colors">닫기</button>
              ) : (
                <Link 
                  to="/wms" 
                  className="inline-block bg-gray-600 text-white px-6 py-3 rounded-xl hover:bg-gray-700 transition-colors"
                >
                  출고 현황으로 돌아가기
                </Link>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 주문을 찾지 못한 경우
  if (!order) {
    return (
      <div className="dark:bg-gray-800 pt-16 min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4 font-sans">
        <div className="max-w-4xl mx-auto text-center">
          <div className="dark:bg-gray-800 bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 p-12">
            <h2 className="dark:text-white text-2xl font-bold text-red-600 mb-4">주문을 찾을 수 없습니다</h2>
            <p className="dark:text-white text-gray-600 mb-6">주문번호 '{id}'에 해당하는 정보가 없습니다.</p>
            {onClose ? (
              <button onClick={onClose} className="inline-block bg-blue-600 text-white px-6 py-3 rounded-xl hover:bg-blue-700 transition-colors">닫기</button>
            ) : (
              <Link 
                to="/wms" 
                className="inline-block bg-blue-600 text-white px-6 py-3 rounded-xl hover:bg-blue-700 transition-colors"
              >
                출고 현황으로 돌아가기
              </Link>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dark:bg-gray-800 pt-16 min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4 font-sans">
      <div className="dark:bg-gray-800 max-w-6xl mx-auto">
        {/* 헤더 및 뒤로가기/닫기 버튼 */}
        <div className="mb-8 flex items-center justify-between">
          <h1 className="dark:text-white mt-4 text-3xl font-bold text-gray-900">주문 상세 정보</h1>
          {onClose ? (
            <button onClick={onClose} className="mt-4 bg-blue-600 text-white px-6 py-3 rounded-xl hover:bg-blue-700 transition-colors">닫기</button>
          ) : (
            <Link 
              to="/wms" 
              className="mt-4 bg-blue-600 text-white px-6 py-3 rounded-xl hover:bg-blue-700 transition-colors"
            >
              출고 현황으로 돌아가기
            </Link>
          )}
        </div>

        {/* 주문 상세 정보 */}
        <div className="dark:bg-gray-800 bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 overflow-hidden">
          <div className="p-8 lg:p-12 grid lg:grid-cols-2 gap-12">
            {/* 주문 기본 정보 */}
            <div className="space-y-6">
              <h2 className="dark:text-white text-2xl font-bold text-gray-900 mb-6 border-b border-gray-200 pb-3">주문 정보</h2>
              <div className="space-y-4">
                <div className="flex justify-between">
                  <span className="dark:text-white font-semibold text-gray-700">주문번호:</span>
                  <span className="text-blue-600 font-bold">{order.id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="dark:text-white font-semibold text-gray-700">고객명:</span>
                  <span className="dark:text-white">{order.recipient}</span>
                </div>
                <div className="flex justify-between">
                  <span className="dark:text-white font-semibold text-gray-700">전화번호:</span>
                  <span className="dark:text-white">{order.phone}</span>
                </div>
                <div className="flex justify-between items-start">
                  <span className="dark:text-white font-semibold text-gray-700 text-nowrap">주소:</span>
                  <span className="dark:text-white text-right max-w-xs whitespace-break-spaces">{order.address}</span>
                </div>
                <div className="flex justify-between">
                  <span className="dark:text-white font-semibold text-gray-700">운송장 번호:</span>
                  <span className="font-mono bg-gray-100 dark:bg-gray-700 dark:text-white px-2 py-1 rounded">
                    {order.tracking_number || '-'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="dark:text-white font-semibold text-gray-700">주문일자:</span>
                  <span className="dark:text-white">{formatDate(order.orderdate)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="dark:text-white font-semibold text-gray-700">출고일자:</span>
                  <span className="dark:text-white">{formatDate(order.outbounddate)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="dark:text-white font-semibold text-gray-700">출고 상태:</span>
                  <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                    order.status === '출고 완료' ? 'bg-green-100 text-green-800' :
                    order.status === '출고 준비' ? 'bg-yellow-100 text-yellow-800' :
                    order.status === '배송 중' ? 'bg-blue-100 text-blue-800' :
                    order.status === '배송 완료' ? 'bg-purple-100 text-purple-800' :
                    order.status === '취소' ? 'bg-red-100 text-red-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {order.status}
                  </span>
                </div>
              </div>
            </div>

            {/* 주문 상품 정보 */}
            <div className="space-y-6">
              <h2 className="dark:text-white text-2xl font-bold text-gray-900 mb-6 border-b border-gray-200 pb-3">주문 상품</h2>
              <div className="dark:bg-gray-700 bg-gray-50 rounded-xl p-6 space-y-4">
                {order.items && order.items.length > 0 ? (
                  <>
                    {order.items.map((item, index) => (
                      <div key={index} className="flex justify-between items-center border-b border-gray-200 dark:border-gray-600 pb-4 last:border-b-0 last:pb-0">
                        <div>
                          <span className="dark:text-white font-semibold text-gray-900">{item.name}</span>
                          <span className="dark:text-white text-gray-600 ml-2">x {item.quantity}개</span>
                        </div>
                        <div className="text-right">
                          <div className="dark:text-white font-semibold text-gray-900">
                            {(item.quantity * item.price).toLocaleString()}원
                          </div>
                          <div className="dark:text-white text-sm text-gray-500">
                            단가: {item.price.toLocaleString()}원
                          </div>
                        </div>
                      </div>
                    ))}
                    <div className="flex justify-between items-center border-t border-gray-300 dark:border-gray-600 pt-4 font-bold text-lg">
                      <span className="dark:text-white text-blue-700">총 금액</span>
                      <span className="dark:text-white text-blue-700">
                        {order.items.reduce((sum, item) => sum + item.quantity * item.price, 0).toLocaleString()}원
                      </span>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                    상품 정보가 없습니다.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* 출고 진행 상태 */}
          <div className="px-8 pb-12">
            <h2 className="dark:text-white text-2xl font-bold text-gray-900 mb-6 border-b border-gray-200 pb-3">출고 진행 상태</h2>
            <div className="flex h-9 items-center justify-center gap-4 overflow-x-auto">
              {["주문 접수", "출고 준비", "출고 완료", "배송 중", "배송 완료"].map((step, idx) => {
                const stepOrder = ["주문 접수", "출고 준비", "출고 완료", "배송 중", "배송 완료"];
                const currentStepIndex = stepOrder.indexOf(order.status);
                const isCompleted = idx < currentStepIndex;
                const isCurrent = idx === currentStepIndex;

                return (
                  <div key={idx} className="flex items-center gap-2">
                    <div className={`w-6 h-6 rounded-full ring-4 flex items-center justify-center ${
                      isCurrent ? 'bg-blue-600 ring-blue-200' : 
                      isCompleted ? 'bg-green-500 ring-green-200' : 
                      order.status === '취소' ? 'bg-red-500 ring-red-200' :
                      'bg-gray-200 ring-gray-100'
                    }`}>
                      {isCompleted && (
                        <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                      {isCurrent && <div className="w-2 h-2 bg-white rounded-full"></div>}
                    </div>
                    <span className={`dark:text-white text-sm whitespace-nowrap ${
                      isCurrent ? 'text-blue-600 font-bold' : 
                      isCompleted ? 'text-green-600 font-semibold' : 
                      'text-gray-600'
                    }`}>
                      {step}
                    </span>
                    {idx < 4 && <div className={`w-8 h-0.5 ${
                      idx < currentStepIndex ? 'bg-green-400' : 'bg-gray-300'
                    }`}></div>}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WmsOutboundDetail; 