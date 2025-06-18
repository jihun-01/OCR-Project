import React from 'react';

const Header = () => {
    return (
        <header className="max-w-screen-sm mx-auto bg-purple-100 flex items-center justify-between px-4 py-3 rounded-2xl shadow">
            <button>
                <svg className="w-6 h-6 text-gray-700" fill="none" stroke="currentColor" stroke-width="2"
                    viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 6h16M4 12h16M4 18h16" />
                </svg>
            </button>
            <h1 className="text-lg font-semibold text-gray-800">출고 검증</h1>
            <button>
                <svg className="w-6 h-6 text-gray-700" fill="none" stroke="currentColor" stroke-width="2"
                    viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4z" />
                <path d="M6 20v-2c0-2.21 3.58-4 6-4s6 1.79 6 4v2H6z" />
                </svg>
            </button>
        </header>
    );
};

export default Header;