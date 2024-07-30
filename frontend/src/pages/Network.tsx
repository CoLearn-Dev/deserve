import { Table, Tag, Card } from 'antd-v5';
import ChatBox from '@/components/Chatbox';
import { CheckCircleOutlined } from '@ant-design/icons';
import { history, useModel } from 'umi';

import React from 'react';
import { useEffect, useState } from 'react';

const Network: React.FC = () => {
  const { initialState, setInitialState } = useModel('@@initialState');
  const { currentUser } = initialState;

  const [dataSource, setDataSource] = useState<
    Array<{ w_id: string; worker_url: string; last_seen: string; created_at: string }>
  >([]);
  const [worker_err, setWorkerErr] = useState<string>('');
  useEffect(() => {
    let interval = setInterval(() => {
      const HTTP_TIMEOUT = 1000;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), HTTP_TIMEOUT);
      const worker_list_api_url = 'https://serving-api.colearn.cloud:8443/get_worker_list';
      fetch(worker_list_api_url, {
        signal: controller.signal,
        headers: { 'api-token': currentUser.api_token },
      })
        .then((res) => res.json())
        .then((data) => {
          // align the format
          data = data.workers.map((item: any) => {
            return {
              w_id: `${item.nickname ? item.nickname : 'No name'} (${item.worker_id}) owned by ${
                item.owner_email
              }`,
              worker_url: item.url,
              last_seen: item.last_seen,
              created_at: item.created_at,
            };
          });
          setDataSource(data);
          setWorkerErr('[connected]');
        })
        .catch((e) => {
          console.log(e);
          setWorkerErr('[disconnected]');
        })
        .finally(() => {
          clearTimeout(timeoutId);
        });
    }, 1000);
    return () => {
      clearInterval(interval);
    };
  }, []);
  return (
    <div>
      <Card>
        <h1>
          <b style={{ color: '#FDB515' }}> Golden Fleece </b> Network
        </h1>
        <p>
          The Golden Fleece Network is a decentralized network of workers that process requests from
          the Golden Fleece API. The network is currently in alpha.
        </p>
      </Card>
      <br />

      <Card
        title={
          <div>
            Worker monitoring{' '}
            <b style={worker_err.includes('dis') ? { color: 'red' } : { color: 'green' }}>
              {worker_err}
            </b>
          </div>
        }
      >
        <Table
          dataSource={dataSource}
          columns={[
            {
              title: 'Worker ID',
              key: 'w_id',
              dataIndex: 'w_id',
            },
            {
              title: 'Worker URL',
              key: 'worker_url',
              dataIndex: 'worker_url',
              render: (v) => {
                return (
                  <Tag icon={<CheckCircleOutlined />} color="default">
                    {v}
                  </Tag>
                );
              },
            },
            {
              title: 'Last Seen',
              key: 'last_seen',
              dataIndex: 'last_seen',
              render: (v) => {
                // convert timestamp to date
                const date = new Date(v * 1000);
                return date.toLocaleString();
              },
            },
            {
              title: 'Created At',
              key: 'created_at',
              dataIndex: 'created_at',
              render: (v) => {
                // convert timestamp to date
                const date = new Date(v * 1000);
                return date.toLocaleString();
              },
              sorter: (a, b) => a.created_at - b.created_at,
            },
          ]}
        />
      </Card>
    </div>
  );
};

export default Network;
