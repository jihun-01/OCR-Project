import './App.css';
import Header from './components/Header/Header';
import OutboundQR from './components/page/OutboundQR';
import OutboundOCR from './components/page/OutboundOCR';
import OutboundResult from './components/page/OutboundResult';
import {Routes, Route, BrowserRouter as Router } from 'react-router-dom';
import WmsHeader from './components/Header/WmsHeader';
import WmsOutbound from './components/page/WmsOutbound';
import WmsApp from './WmsApp';

function App() {
  return (
    <>
      <Router>
        <Header />
        <Routes>
          <Route path="/" element={<OutboundQR />} />
          <Route path="/ocr" element={<OutboundOCR />} />
          <Route path="/result" element={<OutboundResult />} />
        </Routes>
      </Router>


      <Router>
        <WmsHeader />
        <Routes>
          <Route path="/wms" element={<WmsOutbound />} />
        </Routes>
      </Router>
    </>
  );
}

export default App;
