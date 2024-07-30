import { Alert, Row, Col, Button, Card, Divider, Table, Tag, Space, Typography } from 'antd-v5';
import ChatBox from '@/components/Chatbox';
import React from 'react';
import { useEffect, useState } from 'react';
import Cover from './fleece.png';

const { Title, Paragraph, Text, Link } = Typography;

const Home: React.FC = () => {
  return (
    <div>
      <Space direction="vertical" size="large" align="center" style={{ display: 'flex' }}>
        <Row>
          <Col span={3}></Col>
          <Col span={7}>
            <Card
              align="center"
              cover={
                <img
                  src={Cover}
                  alt="Fleece"
                  style={{ height: 300, objectFit: 'cover', objectPosition: '0 30%' }}
                />
              }
            >
              <Title style={{ marginBottom: '0' }}>
                {' '}
                <b style={{ color: '#FDB515' }}>Golden</b> Fleece{' '}
              </Title>
              <b style={{ color: '#FDB515' }}>Decentralized</b> LLM Serving Platform
            </Card>
          </Col>
          <Col span={1}></Col>
          <Col span={10} align="center">
            <div style={{ marginTop: '4em' }}></div>
            <Alert
              message="Notice"
              description="New `.edu` members can claim 1 billion credit to get started!"
              type="info"
              showIcon
              style={{ textAlign: 'left' }}
            />
            <br></br>
            <Paragraph align="justify">
              <ul>
                <li>
                  {' '}
                  Alice has two RTX 3060 Ti, each with <b style={{ color: '#FDB515' }}>8GB</b> GPU
                  memory.{' '}
                </li>
                <li>
                  {' '}
                  Bob has a Telsa V100 with <b style={{ color: '#FDB515' }}>16GB</b> GPU memory.{' '}
                </li>
                <li>
                  {' '}
                  None of them is able to efficiently run LLaMA-2-13b, which takes at least{' '}
                  <b style={{ color: '#FDB515' }}>30GB</b>.{' '}
                </li>
              </ul>
              Golden Fleece enables seamless collaboration between entities like Alice and Bob
              through
              <b style={{ color: '#FDB515' }}> model pipelining </b> and{' '}
              <b style={{ color: '#FDB515' }}> task scheduling </b>. Our platform is designed to
              facilitate the sharing of resources for the decentralized serving of Large Language
              Models (LLMs), leveraging decentralized computing resources for the collective
              benefit.
            </Paragraph>
            <Space size="middle">
              <a href="/playground">
                <Button type="primary">Try it now</Button>
              </a>
              <Button href="/account/credit">Earn credits</Button>
              <Button>See docs</Button>
              <Button href="https://join.slack.com/t/colearn-dev/shared_invite/zt-1gyr9fekz-sPgb_v0PA~XdROupQqy8bQ">
                Join Slack
              </Button>
            </Space>
          </Col>
          <Col span={3}></Col>
        </Row>
        <Row>
          <Col span={3}></Col>
          <Col span={5}>
            <Card title="Simple API access" bordered={false} align="justify">
              Our serving system support OpenAI-compatible endpoint with streaming feature enabled.
            </Card>
          </Col>
          <Col span={1}></Col>
          <Col span={6}>
            <Card title="Decentralized resource" bordered={false} align="justify">
              Decentralized computation resource allows high-throughput LLM inference for faster
              prototyping and experiments.
            </Card>
          </Col>
          <Col span={1}></Col>
          <Col span={5}>
            <Card title="Mine to earn credits" bordered={false} align="justify">
              Trade your spare GPU hours to earn API request credits, accumulate credits to use
              larger models.
            </Card>
          </Col>
          <Col span={3}></Col>
        </Row>
      </Space>
      <br></br>
    </div>
  );
};

export default Home;
