import React, { useState } from "react";
import { Link } from "react-router-dom";
import WmsOutboundDetail from "./WmsOutboundDetail";

const mockData = [
  {
    id: "SHP-20240601-001",
    orderdate: "2025-06-15",
    outbounddate: "2025-06-16",
    recipient: "홍길동",
    phone: "010-1234-5678",
    address: "서울시 강남구 역삼동 123-45, 무슨무슨 건물 101호",
    trackingNumber: "TRACK-001",
    status: "출고 완료",
    items: [
      { name: "마우스", quantity: 2, price: 15000 },
      { name: "키보드", quantity: 1, price: 30000 },
    ],
  },
  {
    id: "SHP-20240601-002",
    orderdate: "2025-06-16",
    outbounddate: "",
    recipient: "김수",
    phone: "010-2345-6789",
    address: "서울시 강남구 역삼동 123-45, 무슨무슨 건물 102호",
    trackingNumber: "",
    status: "주문 접수",
    items: [
      { name: "USB 케이블", quantity: 5, price: 5000 },
    ],
  },
  {
    id: "SHP-20240601-003",
    orderdate: "2025-06-17",
    outbounddate: "",
    recipient: "이지현",
    phone: "010-3456-7890",
    address: "서울시 강남구 역삼동 123-45, 무슨무슨 건물 103호",
    trackingNumber: "",
    status: "출고 준비",
    items: [
      { name: "마우스", quantity: 2, price: 15000 },
      { name: "키보드", quantity: 1, price: 30000 },
    ],
  },
  {
    id: "SHP-20240601-004",
    orderdate: "2025-06-18",
    outbounddate: "2025-06-19",
    recipient: "박윤호",
    phone: "010-4567-8901",
    address: "서울시 강남구 역삼동 123-45, 무슨무슨 건물 104호",
    trackingNumber: "TRACK-002",
    status: "배송 중",
    items: [
      { name: "마우스", quantity: 2, price: 15000 },
      { name: "키보드", quantity: 1, price: 30000 },
    ],
  },
  {
    id: "SHP-20240601-005",
    orderdate: "2025-06-19",
    outbounddate: "",
    recipient: "최다영",
    phone: "010-5678-9012",
    address: "서울시 강남구 역삼동 123-45, 무슨무슨 건물 105호",
    trackingNumber: "",
    status: "취소",
    items: [
      { name: "마우스", quantity: 2, price: 15000 },
      { name: "키보드", quantity: 1, price: 30000 },
    ],
  },
];

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

  const filteredData = mockData.filter((shipment) => {
    const matchSearch = shipment.id.includes(search) || shipment.recipient.includes(search);
    const matchFilter = filter === "전체" || shipment.status === filter;
    return matchSearch && matchFilter;
  });

  return (
    <div className="p-6">
      <h2 className="dark:text-white text-2xl font-bold mb-4">출고 현황</h2>
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
            {filteredData.map((shipment) => (
              <tr key={shipment.id} className="hover:bg-gray-50 dark:text-white">
                <td className="whitespace-nowrap px-4 py-2 border">
                  <button
                    className="text-blue-600 hover:underline dark:text-blue-400"
                    onClick={() => setSelectedId(shipment.id)}
                  >
                    {shipment.id}
                  </button>
                </td>
                <td className="whitespace-nowrap px-4 py-2 border">{shipment.recipient}</td>
                <td className="whitespace-nowrap px-4 py-2 border">{shipment.items.length}</td>
                <td className="whitespace-nowrap px-4 py-2 border">{shipment.status}</td>
                <td className="whitespace-nowrap px-4 py-2 border">{shipment.trackingNumber}</td>
                <td className="whitespace-nowrap px-4 py-2 border">{shipment.orderdate}</td>
                <td className="whitespace-nowrap px-4 py-2 border">{shipment.outbounddate}</td>
              </tr>
            ))}
            {filteredData.length === 0 && (
              <tr>
                <td colSpan={5} className="text-center py-4 text-gray-500 dark:text-white">
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
          <WmsOutboundDetail id={selectedId} onClose={() => setSelectedId(null)} />
        </Modal>
      )}
    </div>
  );
};

export {WmsOutboundstate, mockData};