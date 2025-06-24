import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import useDarkMode from '../../hooks/useDarkMode';

const Header = () => {
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const navigate = useNavigate();
    const { isDarkMode, toggleDarkMode } = useDarkMode();

    const toggleMenu = () => {
        setIsMenuOpen(!isMenuOpen);
    };

    const handleMenuClick = (path) => {
        navigate(path);
        setIsMenuOpen(false);
    };

    return (
        <header className="dark:bg-gray-800 max-w-screen-sm mx-auto h-16 bg-purple-100 px-4 py-3  shadow relative grid grid-cols-3 items-center">
            <div className="relative flex items-center">
                <button onClick={toggleMenu} className="focus:outline-none">
                    {/* 메뉴 버튼 */}
                    <svg className="w-6 h-6 text-gray-700 dark:text-white" fill="none" stroke="currentColor" strokeWidth="2"
                        viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                </button>
                {/* 드롭다운 메뉴 */}
                {isMenuOpen && (
                    <div className="dark:bg-gray-800 absolute top-full left-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
                        <div className="py-2">
                            <button
                                onClick={() => handleMenuClick('/')}
                                className="dark:text-white w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                            >
                                출고 검증
                            </button>
                            <button
                                onClick={() => handleMenuClick('/wms')}
                                className="dark:text-white w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                            >
                                WMS
                            </button>
                        </div>
                    </div>
                )}
            </div>
            <h1 className="dark:text-white text-lg font-semibold text-gray-800 text-center">출고 검증</h1>
            {/* 다크모드 토글 */}
            <div className="flex items-center justify-end">
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
        </header>
    );
};

export default Header;