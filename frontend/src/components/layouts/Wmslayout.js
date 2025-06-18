import React from 'react';
import WmsHeader from '../Header/WmsHeader';
import { Outlet } from 'react-router-dom';

const WmsLayout = () => {
    return (
        <div>
            <WmsHeader />
            <Outlet />
        </div>
    );
};

export default WmsLayout;