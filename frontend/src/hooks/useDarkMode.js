import { useState, useEffect } from 'react';

const useDarkMode = () => {
    // localStorage에서 다크모드 설정을 가져오거나 기본값으로 시스템 설정 사용
    const [isDarkMode, setIsDarkMode] = useState(() => {
        try {
            const savedMode = localStorage.getItem('darkMode');
            if (savedMode !== null) {
                return JSON.parse(savedMode);
            }
            // 저장된 설정이 없으면 시스템 다크모드 설정 확인
            return window.matchMedia('(prefers-color-scheme: dark)').matches;
        } catch (error) {
            console.error('다크모드 설정 로드 중 오류:', error);
            return false;
        }
    });

    // 다크모드 상태 변경 시 DOM과 localStorage 업데이트
    useEffect(() => {
        try {
            if (isDarkMode) {
                document.documentElement.classList.add('dark');
            } else {
                document.documentElement.classList.remove('dark');
            }
            localStorage.setItem('darkMode', JSON.stringify(isDarkMode));
        } catch (error) {
            console.error('다크모드 설정 저장 중 오류:', error);
        }
    }, [isDarkMode]);

    // 다크모드 토글 함수
    const toggleDarkMode = () => {
        setIsDarkMode(prevMode => !prevMode);
    };

    return { isDarkMode, toggleDarkMode };
};

export default useDarkMode;