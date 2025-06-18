import OutboundQR from './components/pages/OutboundQR';
import OutboundOCR from './components/pages/OutboundOCR';
import OutboundResult from './components/pages/OutboundResult'; 
import { useRoutes } from 'react-router-dom';
import NotFound from './components/pages/NotFound';
import WmsOutbound from './components/pages/WmsOutbound';
import MobileLayout from './components/layouts/Mobilelayout';
import WmsLayout from './components/layouts/Wmslayout';
import WmsOutboundDetail from './components/pages/WmsOutboundDetail';




function App() {
  const routes = [
    {
      element: <MobileLayout />,
      children: [
        { path: '/', element: <OutboundQR /> },
        { path: '/ocr', element: <OutboundOCR /> },
        { path: '/result', element: <OutboundResult /> },
      ],
    },
    {
      element: <WmsLayout />,
      children: [
        { path: '/wms', element: <WmsOutbound /> },
        { path: '/wms/search/:id', element: <WmsOutboundDetail /> },
      ],
    },
    {
      path: '*',
      element: <NotFound />,
    },
  ];

  return useRoutes(routes);
}

export default App;
