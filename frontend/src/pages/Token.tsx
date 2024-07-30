import { Table, Tag, Card } from 'antd-v5';
import ChatBox from '@/components/Chatbox';
import { CheckCircleOutlined } from '@ant-design/icons';

import React from 'react';
import { useEffect, useState } from 'react';
import { useParams } from 'umi';
import { history, useModel } from 'umi';
import { currentUser as queryCurrentUser } from '@/services/ant-design-pro/api';

const Token: React.FC = () => {
  const queryParameters = new URLSearchParams(window.location.search)
  const token = queryParameters.get('token');
  localStorage.setItem("token", token);
  useEffect(() => {
    history.push('/home');
    window.location.reload();
  }, []);
  return (
    <div>
      <Card>
        <h1>
          <b style={{ color: '#FDB515' }}> Token Loading </b> 
          <p> {JSON.stringify(token)} </p>
        </h1>
      </Card>
      <br />
    </div>
  );
};

export default Token;
