import React from 'react';
import logo from '../../assets/wmslogo.png';

const wms_user_name = "test";
const wms_user_email = "test@test.com";
const user_profile = "https://cdn.startupful.io/img/main_page/profile1.png";

// 메뉴 리스트
const wms_menu = [
    { title: '출고', link: '/wms' },
];

const WmsHeader = () => {
    return (
        <>
            <div className="fixed top-16 left-0 right-0 z-10 flex bg-gray-50 dark:bg-[#252731] items-center h-16">
                <div className="flex w-1/5 justify-start ml-16">
                    <a href="/wms">
                        <img src={logo} alt="logo" className="w-16 h-16"/>
                    </a>
                </div>
                {/* 메뉴 리스트 */}
                <div className="flex w-4/5 mx-auto">
                    <nav className="flex">
                        <ul className="flex">
                            {wms_menu.map((menu, index) => (
                                <li key={index} className="px-4 dark:text-white hover:text-gray-500 dark:hover:text-gray-400">
                                    <a href={menu.link}>{menu.title}</a>
                                </li>
                            ))}
                        </ul>
                    </nav>
                    {/* 유저 정보 */}
                    <div className="absolute bottom-0 right-0 p-4 border-t border-gray-200 dark:border-gray-700">
                      <div className="flex items-center space-x-3">
                        <img src={user_profile} alt="Profile" className="w-10 h-10 rounded-full"/>
                        <div>
                            <div className="text-sm font-medium text-black dark:text-white">{wms_user_name}</div>
                            <div className="text-xs text-gray-500 dark:text-gray-400">{wms_user_email}</div>
                        </div>
                        <a href="/project/wms" className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-500 dark:hover:text-gray-400">로그아웃</a>
                    </div>
                </div>
                
                
                </div>
            </div>


        </>
    );
};

export default WmsHeader;