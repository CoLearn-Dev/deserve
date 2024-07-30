import { Avatar, Timeline, Tag, List } from 'antd-v5';
import { Space, Input, Button, Card } from 'antd';
import { CheckCircleOutlined, MessageOutlined } from '@ant-design/icons';
import OpenAI from 'openai';
import { useState } from 'react';
import llamaTokenizer from 'llama-tokenizer-js';
import { currentUser } from '@/services/ant-design-pro/api';
import { history, useModel } from 'umi';

const ChatBox: React.FC = ({ setChatId, currentModel }) => {
  const { initialState, setInitialState } = useModel('@@initialState');
  const { currentUser } = initialState;

  const [prompt, setPrompt] = useState<string>('');
  const [items, setItems] = useState<Array<{ label: string; children: string }>>([]);
  const [isNewChat, setIsNewChat] = useState<boolean>(true);
  const [pending, setPending] = useState<string>('Waiting for the first message...');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [tagText, setTagText] = useState<string>('ready');

  const sendMsg = async () => {
    setIsRunning(true);
    setChatId('');
    // current time
    const date = new Date();
    // map items to messages, change field: label->user, children->content
    const messages = items.map((item) => ({ role: item.label, content: item.children }));
    // add new message to messages
    messages.push({ role: 'user', content: prompt });
    // update items
    setItems([...items, { label: 'user', children: prompt }]);
    // update pending
    if (isNewChat) {
      setIsNewChat(false);
      setPending('');
    }
    const openai = new OpenAI({
      apiKey: currentUser.api_token,
      baseURL: 'https://serving-api.colearn.cloud:9000/v1',
      dangerouslyAllowBrowser: true,
    });
    const stream = await openai.chat.completions.create({
      model: currentModel,
      messages: messages,
      stream: true,
    });
    setPrompt('');
    var reply = [];
    let chatId = null;
    for await (const chunk of stream) {
      chatId = chunk.id;
      const content = chunk.choices[0]?.delta?.content;
      // content is undefined, or is in the format of "[id]token", extract id here
      if (content) {
        const token_id = content?.split(']')[0].slice(1);
        reply.push(token_id);
      }
      setPending(llamaTokenizer.decode(reply, false, false));
      // calculate the speed
      const speed = (reply.length / (new Date() - date)) * 1000;
      setTagText(`${speed.toFixed(2)} tokens/s`);
    }
    // update items
    setPending('');
    setItems([
      ...items,
      { label: 'user', children: prompt },
      { label: 'assistant', children: llamaTokenizer.decode(reply, false, false) },
    ]);
    setChatId(chatId);
    setIsRunning(false);
  };
  return (
    <Card
      title={
        <Space>
          <MessageOutlined />
          Chat with Fleece backend
          <Tag color="blue">{currentModel}</Tag>
        </Space>
      }
      bordered={true}
    >
      <List
        itemLayout="horizontal"
        dataSource={items.concat(pending ? [{ label: 'assistant', children: pending }] : [])}
        renderItem={(item, index) => (
          <List.Item>
            <List.Item.Meta
              avatar={
                <Avatar
                  src={item.label == 'user' ? currentUser.avatar : '/static/fleece.e90710a5.png'}
                />
              }
              title={item.label}
              description={item.children}
            />
          </List.Item>
        )}
      />

      <Tag
        icon={<CheckCircleOutlined />}
        color={isRunning ? 'processing' : 'success'}
        style={{ float: 'right', marginRight: 0 }}
      >
        {tagText}
      </Tag>
      <Space.Compact style={{ width: '100%' }}>
        <Input
          placeholder="Enter your prompt here..."
          onChange={(e) => setPrompt(e.target.value)}
          value={prompt}
          onPressEnter={(e) => sendMsg()}
          disabled={isRunning}
        />
        <Button type="primary" onClick={(e) => sendMsg()} disabled={isRunning}>
          Send
        </Button>
      </Space.Compact>
    </Card>
  );
};

export default ChatBox;
