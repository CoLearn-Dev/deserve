import { Table, Tag, Card, Divider, Progress, Row, Col, Typography, Statistic } from 'antd-v5';
import {
  AlertFilled,
  DesktopOutlined,
  FrownFilled,
  GiftFilled,
  HeartFilled,
  MinusCircleFilled,
  PlayCircleFilled,
  PlusCircleFilled,
  SplitCellsOutlined,
  StarFilled,
  UnorderedListOutlined,
  UsergroupAddOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

import CreditProgress from '@/components/CreditProgress';

import React from 'react';
import { useEffect, useState } from 'react';
import { history, useModel } from 'umi';

const Credit: React.FC = () => {
  const [credit, setCredit] = useState<number>(0);
  const [tx, setTx] = useState<any>([]);

  const { initialState, setInitialState } = useModel('@@initialState');
  const { currentUser } = initialState;

  useEffect(() => {
    const credit_api_url = 'https://serving-api.colearn.cloud:8443/get_remaining_credit';
    if (currentUser.api_token) {
      fetch(credit_api_url, { headers: { 'api-token': currentUser.api_token } })
        .then((response) => response.json())
        .then((data) => {
          setCredit(data.remaining_credit);
        });
    }

    console.log('123');

    const tx_api_url = 'https://serving-api.colearn.cloud:8443/get_credit_transactions';
    if (currentUser.api_token) {
      fetch(tx_api_url, { headers: { 'api-token': currentUser.api_token } })
        .then((response) => response.json())
        .then((data) => {
          let t = data.credit_transactions;
          t.forEach((e: any) => {
            e.event = {
              etype: e.event_type,
              detail: e.event_detail,
            };
          });
          setTx(t);
        });
    }
  }, currentUser);

  return (
    <div>
      <Card>
        <h1>
          My <b style={{ color: '#FDB515' }}> Credit </b>
        </h1>
        <Divider />
        <Row>
          <Col span={1} />
          <Col span={4} justify={'center'}>
            <CreditProgress credit={credit} size={200} type={'circle'} width={'100%'} />
          </Col>
          <Col span={1} />
          <Col span={18}>
            <Row gutter={32}>
              <Col span={8}>
                <Statistic
                  title="Remaining credit"
                  value={credit}
                  valueStyle={{ color: '#FDB515' }}
                  prefix={<StarFilled />}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Total earned"
                  value={tx
                    .filter((e: any) => e.amount > 0)
                    .reduce((a: any, b: any) => a + b.amount, 0)}
                  valueStyle={{ color: 'green' }}
                  prefix={<PlusCircleFilled />}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Total spent"
                  value={
                    -tx.filter((e: any) => e.amount < 0).reduce((a: any, b: any) => a + b.amount, 0)
                  }
                  valueStyle={{ color: 'red' }}
                  prefix={<MinusCircleFilled />}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Number of transactions"
                  value={tx.length}
                  valueStyle={{ color: 'black' }}
                  prefix={<UnorderedListOutlined />}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="# Chat completions"
                  value={tx.filter((e: any) => e.event_type === 'chat_completion').length}
                  valueStyle={{ color: '#389e0d' }}
                  prefix={<PlayCircleFilled />}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="# Request served"
                  value={tx.filter((e: any) => e.event_type === 'serve').length}
                  valueStyle={{ color: '#531dab' }}
                  prefix={<SplitCellsOutlined />}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Account type"
                  value={currentUser.email.split('@')[1]}
                  valueStyle={{ color: 'black' }}
                  prefix={<UsergroupAddOutlined />}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="# Runtime connected"
                  value={tx.filter((e: any) => e.event_type === 'heartbeat').length}
                  // TODO: only keep one heartbeat for the same machines
                  valueStyle={{ color: '#531dab' }}
                  prefix={<DesktopOutlined />}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="# Heartbeat reported"
                  value={tx.filter((e: any) => e.event_type === 'heartbeat').length}
                  valueStyle={{ color: '#c41d7f' }}
                  prefix={<AlertFilled />}
                />
              </Col>
            </Row>
          </Col>
        </Row>

        <Divider />
        <Table
          size={'small'}
          dataSource={tx}
          columns={[
            {
              title: 'Event type',
              dataIndex: 'event_type',
              render: function (text: string) {
                switch (text) {
                  case 'bonus':
                    return (
                      <Tag color="gold" icon={<GiftFilled />}>
                        Bonus
                      </Tag>
                    );
                  case 'chat_completion':
                    return (
                      <Tag color="green" icon={<PlayCircleFilled />}>
                        Chat completion
                      </Tag>
                    );
                  case 'heartbeat':
                    return (
                      <Tag color="magenta" icon={<AlertFilled />}>
                        Heartbeat
                      </Tag>
                    );
                  case 'serve':
                    return (
                      <Tag color="purple" icon={<SplitCellsOutlined />}>
                        Serve
                      </Tag>
                    );
                  case 'penalty':
                    return (
                      <Tag color="cyan" icon={<FrownFilled />}>
                        Penalty
                      </Tag>
                    );
                  default:
                    return <Tag color="blue">{text}</Tag>;
                }
              },
            },
            {
              title: 'Detail',
              dataIndex: 'event',
              render: function (e) {
                switch (e.etype) {
                  case 'bonus':
                    return e.detail;
                  case 'chat_completion': {
                    // chat_id, model, input_token, output_token
                    let detail = JSON.parse(e.detail);
                    return (
                      <div>
                        <Tag color="grey">
                          <b>Chat ID</b>: {detail.chat_id}
                        </Tag>
                        <Tag color="#FDB515">
                          <b>Model</b>: {detail.model}
                        </Tag>
                        <Tag color="blue">
                          <b>#Input</b>: {detail.input_token}
                        </Tag>
                        <Tag color="blue">
                          <b>#Output</b>: {detail.output_token}
                        </Tag>
                      </div>
                    );
                  }
                  case 'heartbeat':
                    return <Tag color="red" icon={<HeartFilled />}></Tag>;
                  case 'serve': {
                    // chat_id, layers, input_token, output_token
                    let detail = JSON.parse(e.detail);
                    return (
                      <div>
                        <Tag color="grey">
                          <b>Chat ID</b>: {detail.chat_id}
                        </Tag>
                        <Tag color="#FDB515">
                          <b>#Layers</b>:
                          <Progress
                            steps={35}
                            percent={(100 * detail.layers.length) / 35}
                            size={'small'}
                            showInfo={false}
                            strokeColor={'white'}
                            style={{ paddingLeft: '2px', paddingRight: '2px' }}
                          />
                          {detail.layers.length} / 35
                        </Tag>
                        <Tag color="blue">
                          <b>#Input</b>: {detail.input_token}
                        </Tag>
                        <Tag color="blue">
                          <b>#Output</b>: {detail.output_token}
                        </Tag>
                      </div>
                    );
                  }
                  case 'penalty':
                    return e.detail;
                  default:
                    return JSON.stringify(e.detail);
                }
              },
            },
            {
              title: 'Amount',
              dataIndex: 'amount',
              align: 'right',
              render: (text: string) => (
                <b style={{ color: parseInt(text) > 0 ? 'red' : 'green' }}>
                  {`${parseInt(text) > 0 ? '+' : ''}${parseInt(text).toLocaleString()}`}
                </b>
              ),
            },
            {
              title: 'Timestamp',
              dataIndex: 'timestamp',
              align: 'right',
              render: (text: string) => {
                const date = new Date(text * 1000);
                return date.toLocaleString();
              },
            },
          ]}
        />
      </Card>
      <br />
    </div>
  );
};

export default Credit;
