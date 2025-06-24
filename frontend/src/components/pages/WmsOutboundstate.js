import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import WmsOutboundDetail from "./WmsOutboundDetail";

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

const Modal = ({ children, onClose }) => (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
    <div className="relative w-full max-w-4xl mx-auto">
      <div className="absolute top-0 right-0 mt-4 mr-4">
        <button onClick={onClose} className="text-3xl text-gray-500 hover:text-gray-800 dark:hover:text-white">&times;</button>
      </div>
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-auto max-h-[90vh]">
        {children}
      </div>
    </div>
  </div>
);

const WmsOutboundstate = () => {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("전체");
  const [selectedId, setSelectedId] = useState(null);
  const [orders, setOrders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // 주문 목록 조회 함수
  const fetchOrders = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const params = new URLSearchParams();
      if (filter !== "전체") params.append("status", filter);
      if (search.trim()) params.append("search", search.trim());
      
      const response = await fetch(
        `${API_BASE_URL}/orders?${params.toString()}`,
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
      
      if (result.success) {
        setOrders(result.orders);
      } else {
        throw new Error(result.message || '주문 목록을 가져올 수 없습니다.');
      }
      
    } catch (err) {
      console.error('주문 목록 조회 오류:', err);
      setError('주문 목록을 불러오는 중 오류가 발생했습니다.');
      setOrders([]);
    } finally {
      setLoading(false);
    }
  };

  // 컴포넌트 마운트 시 데이터 로드
  useEffect(() => {
    fetchOrders();
  }, []);

  // 필터나 검색어 변경 시 데이터 다시 로드
  useEffect(() => {
    const delayedSearch = setTimeout(() => {
      fetchOrders();
    }, 300); // 300ms 디바운스

    return () => clearTimeout(delayedSearch);
  }, [filter, search]);

  // 로딩 상태
  if (loading) {
    return (
      <div className="p-6">
        <h2 className="dark:text-white text-2xl font-bold mb-4">출고 현황</h2>
        <div className="flex justify-center items-center h-64">
          <div className="text-lg text-gray-500 dark:text-gray-400">데이터를 불러오는 중...</div>
        </div>
      </div>
    );
  }

  // 에러 상태
  if (error) {
    return (
      <div className="p-6">
        <h2 className="dark:text-white text-2xl font-bold mb-4">출고 현황</h2>
        <div className="flex justify-center items-center h-64">
          <div className="text-lg text-red-500">{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <h2 className="dark:text-white text-2xl font-bold mb-4">출고 현황</h2>
      
      {/* 검색 및 필터 */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 gap-2">
        <input
          type="text"
          placeholder="출고 번호 또는 수령인 검색"
          className="border border-gray-300 rounded px-3 py-2 w-full sm:w-64"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          className="border border-gray-300 rounded px-3 py-2 w-full sm:w-48"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="전체">전체</option>
          <option value="주문 접수">주문 접수</option>
          <option value="출고 준비">출고 준비</option>
          <option value="출고 완료">출고 완료</option>
          <option value="배송 중">배송 중</option>
          <option value="배송 완료">배송 완료</option>
          <option value="취소">취소</option>
        </select>
      </div>

      {/* 결과 개수 표시 */}
      <div className="mb-4 text-sm text-gray-600 dark:text-gray-400">
        총 {orders.length}개의 주문
      </div>
      
      {/* 테이블 */}
      <div className="overflow-x-auto">
        <table className="min-w-full border border-gray-200">
          <thead className="bg-gray-100">
            <tr>
              <th className="whitespace-nowrap text-left px-4 py-2 border">주문 번호</th>
              <th className="whitespace-nowrap text-left px-4 py-2 border">주문자</th>
              <th className="whitespace-nowrap text-left px-4 py-2 border">주문 수량</th>
              <th className="whitespace-nowrap text-left px-4 py-2 border">출고 상태</th>
              <th className="whitespace-nowrap text-left px-4 py-2 border">운송장 번호</th>
              <th className="whitespace-nowrap text-left px-4 py-2 border">주문 일시</th>
              <th className="whitespace-nowrap text-left px-4 py-2 border">출고 일자</th>
            </tr>
          </thead>
          <tbody>
            {orders.map((order) => (
              <tr key={order.id} className="hover:bg-gray-50 dark:text-white">
                <td className="whitespace-nowrap px-4 py-2 border">
                  <button
                    className="text-blue-600 hover:underline dark:text-blue-400"
                    onClick={() => setSelectedId(order.id)}
                  >
                    {order.id}
                  </button>
                </td>
                <td className="whitespace-nowrap px-4 py-2 border">{order.recipient}</td>
                <td className="whitespace-nowrap px-4 py-2 border">
                  {order.items ? order.items.length : 0}
                </td>
                <td className="whitespace-nowrap px-4 py-2 border">
                  <span className={`px-2 py-1 rounded text-xs ${
                    order.status === '출고 완료' ? 'bg-green-100 text-green-800' :
                    order.status === '배송 중' ? 'bg-blue-100 text-blue-800' :
                    order.status === '배송 완료' ? 'bg-gray-100 text-gray-800' :
                    order.status === '취소' ? 'bg-red-100 text-red-800' :
                    'bg-yellow-100 text-yellow-800'
                  }`}>
                    {order.status}
                  </span>
                </td>
                <td className="whitespace-nowrap px-4 py-2 border">{order.trackingNumber}</td>
                <td className="whitespace-nowrap px-4 py-2 border">{order.orderdate}</td>
                <td className="whitespace-nowrap px-4 py-2 border">{order.outbounddate}</td>
              </tr>
            ))}
            {orders.length === 0 && (
              <tr>
                <td colSpan={7} className="text-center py-4 text-gray-500 dark:text-white">
                  일치하는 출고 내역이 없습니다.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      
      {/* 모달로 상세 정보 표시 */}
      {selectedId && (
        <Modal onClose={() => setSelectedId(null)}>
          <WmsOutboundDetail 
            id={selectedId} 
            onClose={() => setSelectedId(null)}
            onUpdate={fetchOrders} // 데이터 업데이트 시 목록 새로고침
          />
        </Modal>
      )}
    </div>
  );
};

export default WmsOutboundstate;