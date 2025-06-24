import { createContext, useContext, useState } from 'react';

const OrderContext = createContext();

export const useOrder = () => useContext(OrderContext);

export const OrderProvider = ({ children }) => {
  const [orderId, setOrderId] = useState('');
  const [extractedText, setExtractedText] = useState('');

  return (
    <OrderContext.Provider
      value={{
        orderId,
        setOrderId,
        extractedText,
        setExtractedText,
      }}
    >
      {children}
    </OrderContext.Provider>
  );
};

export default OrderContext;