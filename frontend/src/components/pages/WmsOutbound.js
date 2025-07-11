import React from 'react';
import WmsOutboundstate from './WmsOutboundstate';
import { Routes, Route } from 'react-router-dom';



const wms_outbound_menu = [
    { title: '출고현황', link: '/wms' },
];

const WmsOutbound = () => {
    return (
        <div className="h-screen w-full bg-white dark:bg-[#1E2028]">


            {/* 사이드바 */}
            <div className="hidden sm:block fixed left-0 top-16 h-[calc(100vh-4rem)] w-48 bg-gray-50 dark:bg-[#252731] border-r border-gray-200 dark:border-gray-700">
                <nav className="p-4 space-y-2">
                    {wms_outbound_menu.map((menu, index) => (
                        <a href={menu.link} className="flex items-center space-x-3 p-2 rounded-lg text-black dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700">
                            <span>{menu.title}</span>
                        </a>
                    ))}
                </nav>


            </div>

            {/* 메인 컨텐츠 영역 */}
            <div className="sm:ml-64 pt-16 h-screen">
                {/* 여기에 메인 컨텐츠를 추가하세요 */}
                <Routes>
                    <Route path="/" element={<WmsOutboundstate />} />
                </Routes>
            </div>
        </div>
    );
};

export default WmsOutbound;