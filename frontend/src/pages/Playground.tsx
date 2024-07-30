import { Badge, Card, Col, Descriptions, Divider, Progress, Row, Select } from 'antd-v5';
import React from 'react';
import ChatBox from '@/components/Chatbox';
import Circuit from '@/components/Circuit';
import CreditProgress from '@/components/CreditProgress';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { a11yLight, a11yDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import { history, useModel } from 'umi';
import { useEffect, useState } from 'react';

const Playground: React.FC = () => {
  const [credit, setCredit] = useState<number>(0);
  const [chatId, setChatId] = useState<string>('');
  const [modelOptions, setModelOptions] = useState<any[]>([
    {
      label: <span> LLaMA </span>,
      title: 'llama',
      options: [
        { label: <span> llama-2-7b-chat </span>, value: 'llama-2-7b-chat' },
        { label: <span> llama-2-13b-chat </span>, value: 'llama-2-13b-chat' },
        { label: <span> llama-2-70b-chat </span>, value: 'llama-2-70b-chat' },
      ],
    },
    {
      label: <span> Others </span>,
      title: 'others',
      options: [{ label: <span> dummy </span>, value: 'dummy' }],
    },
  ]);
  const [currentModel, setCurrentModel] = useState<string>('llama-2-7b-chat');

  const { initialState, setInitialState } = useModel('@@initialState');
  const { currentUser } = initialState;

  const api_key = currentUser.api_token
    ? currentUser.api_token
    : '<Please login first to get API token>';
  const credit_api_url = 'https://serving-api.colearn.cloud:8443/get_remaining_credit';
  if (currentUser.api_token) {
    fetch(credit_api_url, { headers: { 'api-token': currentUser.api_token } })
      .then((response) => response.json())
      .then((data) => {
        setCredit(data.remaining_credit);
      });
  }

  const api_endpoint = 'https://serving-api.colearn.cloud:9000/v1';

  var api_items = [
    {
      key: '1',
      label: 'Your API key',
      children: api_key,
    },
    {
      key: '2',
      label: 'Your remaining credit',
      children: <CreditProgress credit={credit} width={200} size="small" type="line" />,
    },
    {
      key: '3',
      label: 'Service Status',
      span: 3,
      children: <Badge status="processing" text="Running" />,
    },
  ];
  var request_items = [
    {
      key: '1',
      label: 'Chat ID',
      children: chatId ? chatId : '...',
    },
  ];
  useEffect(() => {
    const chat_info_api_url = 'https://serving-api.colearn.cloud:8443/get_chat_info';
    fetch(chat_info_api_url, {
      headers: { 'api-token': currentUser.api_token, 'Content-Type': 'application/json' },
      method: 'POST',
      body: JSON.stringify({ chat_id: chatId }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
      });
  }, [chatId]);

  return (
    <div>
      <Card>
        <h1>Try it out</h1>
        <p>
          We provide both
          <a style={{ color: '#FDB515' }}> API access </a>
          and
          <a style={{ color: '#FDB515' }}> Chatbox demo </a>
          for trying out our serving sytem.
        </p>
        <p>
          <b>Model to try out</b>: &nbsp;
          <Select
            showSearch
            size="large"
            defaultValue={currentModel}
            value={currentModel}
            placeholder="Select a model"
            optionFilterProp="children"
            onInputKeyDown={(e) => {
              if (e.key === 'Enter') {
                let mo = modelOptions;
                let v = e.target.value;
                // check if the option exists
                let found = false;
                for (let i = 0; i < mo.length; i++) {
                  for (let j = 0; j < mo[i].options.length; j++) {
                    if (mo[i].options[j].value === v) {
                      found = true;
                      break;
                    }
                  }
                }
                if (!found) {
                  mo[mo.length - 1].options.push({ label: <span> {v} </span>, value: v });
                  setModelOptions(mo);
                  console.log('mo', mo);
                }
                console.log('v', v);
                setCurrentModel(v);
              }
            }}
            style={{ width: '100%' }}
            notFoundContent={<span> Custom model name? Press enter to confirm. </span>}
            onChange={(value) => {
              setCurrentModel(value);
            }}
            // filterOption={filterOption}
            options={modelOptions}
          />
        </p>
      </Card>
      <br />
      <Card>
        <h1>API access</h1>

        <Descriptions layout="vertical" bordered={true} items={api_items} />
        <Divider />
        <h3> Try it in your terminal: </h3>
        <SyntaxHighlighter language="bash" style={a11yDark}>
          {`curl ` +
            api_endpoint +
            `/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ` +
            api_key +
            `" \\
  -d '{
    "model": "` +
            currentModel +
            `",
    "messages": [{"role": "user", "content": "Hello world!"}]
  }'\
`}
        </SyntaxHighlighter>
        <Divider />
        <h3>
          {' '}
          Try it with <a href="https://pypi.org/project/openai/">Python OpenAI package</a>:{' '}
        </h3>
        <SyntaxHighlighter language="python" style={a11yDark}>
          {`\
from openai import OpenAI

client = OpenAI(
    api_key="` +
            api_key +
            `",
    base_url="` +
            api_endpoint +
            `",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Hello world!",
        }
    ],
    model="` +
            currentModel +
            `",
)

print(chat_completion)\
  `}
        </SyntaxHighlighter>
      </Card>

      <br />

      <Card>
        <h1> Chatbot demo </h1>
        <Row>
          <Col span={14}>
            <ChatBox setChatId={setChatId} currentModel={currentModel} />
          </Col>
          <Col span={1}></Col>
          <Col span={9}>
            <Card title="Serving details" bordered>
              <Descriptions layout="vertical" bordered={true} items={request_items} />
              <Divider />
              <Circuit chatId={chatId} />
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default Playground;
