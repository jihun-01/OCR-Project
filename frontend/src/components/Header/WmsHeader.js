import React, { useState } from 'react';
import logo from '../../assets/wmslogo.png';
import useDarkMode from '../../hooks/useDarkMode';

const wms_user_name = "test";
const wms_user_email = "test@test.com";
const user_profile = "https://cdn.startupful.io/img/main_page/profile1.png";

// 메뉴 리스트
const wms_menu = [
    { title: '출고 검증', link: '/' },
    { title: 'WMS', link: '/wms' },
];

const WmsHeader = () => {
    const { isDarkMode, toggleDarkMode } = useDarkMode();
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const toggleMenu = () => {
        setIsMenuOpen(!isMenuOpen);
    };

    return (
        
        <>
            <div className="fixed top-0 left-0 right-0 z-10 flex bg-gray-50 dark:bg-[#252731] items-center h-16">
                <div className="flex w-16 justify-start ml-16 h-16">
                    <a href="/wms">
                        <img src={logo} alt="logo" className="w-16 h-16"/>
                    </a>
                </div>
                {/* 메뉴 리스트 */}
                <div className="hidden md:flex w-4/5 mx-auto">
                    <nav className="flex">
                        <ul className="flex">
                            {wms_menu.map((menu, index) => (
                                <li key={index} className="px-4 dark:text-white hover:text-gray-500 dark:hover:text-gray-400">
                                    <a href={menu.link}>{menu.title}</a>
                                </li>
                            ))}
                        </ul>
                    </nav>
                    {/* 다크모드 토글 */}
                <div className="hidden md:block absolute right-64">
                    <label className="flex items-center cursor-pointer">
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">다크모드</span>
                        <div className="relative ml-3">
                            <input 
                                type="checkbox" 
                                className="sr-only peer" 
                                id="toggleBasic"
                                checked={isDarkMode}
                                onChange={toggleDarkMode}
                            />
                            <div className="dark:bg-gray-700 w-11 h-6 bg-gray-200 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-500"></div>
                        </div>
                    </label>
                </div>
                    {/* 유저 정보 */}
                    <div className="hidden md:block absolute bottom-0 right-0 p-4 border-t border-gray-200 dark:border-gray-700">
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
            {/* 모바일 메뉴 버튼 */}
            <div className="md:hidden sm:block absolute left-4">
                        <button onClick={toggleMenu} className="focus:outline-none">
                            {/* 메뉴 버튼 */}
                            <svg className="w-6 h-6 text-gray-700 dark:text-white" fill="none" stroke="currentColor" strokeWidth="2"
                                viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                        </button>
                    </div>
                                
                    {isMenuOpen && (
                        <div className="absolute top-full left-16 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 z-50 dark:bg-gray-800">
                            <div className="py-2">
                                {wms_menu.map((menu, index) => (
                                    <a
                                        key={index}
                                        href={menu.link}
                                        className="dark:text-white w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors block"
                                    >
                                        {menu.title}
                                    </a>
                                ))}
                            </div>
                            {/* 다크모드 토글 */}
                            <div className="py-2 border-t border-gray-200 dark:border-gray-700">
                                <label className="flex items-center cursor-pointer px-4">
                                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">다크모드</span>
                                    <div className="relative ml-3">
                                        <input 
                                            type="checkbox" 
                                            className="sr-only peer" 
                                            id="toggleBasicMobile"
                                            checked={isDarkMode}
                                            onChange={toggleDarkMode}
                                        />
                                        <div className="dark:bg-gray-700 w-11 h-6 bg-gray-200 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-500"></div>
                                    </div>
                                </label>
                            </div>
                            {/* 유저 정보 */}
                            <div className="py-2 border-t border-gray-200 dark:border-gray-700 px-4">
                                <div className="flex items-center space-x-3">
                                    <img src={user_profile} alt="Profile" className="w-10 h-10 rounded-full"/>
                                    <div>
                                        <div className="text-sm font-medium text-black dark:text-white">{wms_user_name}</div>
                                        <div className="text-xs text-gray-500 dark:text-gray-400">{wms_user_email}</div>
                                    </div>
                                </div>
                                <a href="/project/wms" className="block mt-2 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-500 dark:hover:text-gray-400">로그아웃</a>
                            </div>
                        </div>
                    )}
                </div>
        

        </>
    );
};

export default WmsHeader;