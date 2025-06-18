import React from 'react';
import WmsOutbound from './components/page/WmsOutbound';
import WmsHeader from './components/Header/WmsHeader';
import { Routes, Route } from 'react-router-dom';



function WmsApp() {
    return (
        <div>
            
            <Routes>
                <Route path="/wms" element={<WmsOutbound />} />
            </Routes>

        </div>
    );
}

export default WmsApp;




