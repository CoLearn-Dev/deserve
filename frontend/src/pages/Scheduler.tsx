import { Table, Tag, Card } from 'antd-v5';
import ChatBox from '@/components/Chatbox';
import { CheckCircleOutlined } from '@ant-design/icons';

import React from 'react';
import { useEffect, useState } from 'react';

const Network: React.FC = () => {

  return (
    <div>
      <Card>
        <h1>
          <b style={{ color: '#FDB515' }}>  Scheduler </b> 
        </h1>
        <p>
          Display scheduled circuit.
        </p>
      </Card>
      <br />
    </div>
  );
};

export default Network;
