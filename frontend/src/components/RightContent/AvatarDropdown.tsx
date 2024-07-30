import { outLogin } from '@/services/ant-design-pro/api';
import {
  LoginOutlined,
  LogoutOutlined,
  SettingOutlined,
  UserOutlined,
  WalletOutlined,
} from '@ant-design/icons';
import { Avatar, Menu, Spin } from 'antd';
import type { ItemType } from 'antd/lib/menu/hooks/useItems';
import { stringify } from 'querystring';
import type { MenuInfo } from 'rc-menu/lib/interface';
import React, { useCallback } from 'react';
import { history, useModel } from 'umi';
import HeaderDropdown from '../HeaderDropdown';
import styles from './index.less';

export type GlobalHeaderRightProps = {
  menu?: boolean;
};

const AvatarDropdown: React.FC<GlobalHeaderRightProps> = ({ menu }) => {
  const { initialState, setInitialState } = useModel('@@initialState');

  const onMenuClick = useCallback(
    (event: MenuInfo) => {
      const { key } = event;
      if (key === 'login') {
        const url = 'https://serving-api.colearn.cloud:8443/oauth?code=test@colearn.dev';
        window.location.href = url;
        return;
      }
      if (key === 'logout') {
        localStorage.removeItem('token');
        setInitialState((s) => ({
          ...s,
          currentUser: {
            email: 'Guest',
            avatar: '/icons8-who-100.png',
          },
        }));
        window.location.reload();
        return;
      }
      history.push(`/account/${key}`);
    },
    [setInitialState],
  );

  const loading = (
    <span className={`${styles.action} ${styles.account}`}>
      <Spin
        size="small"
        style={{
          marginLeft: 8,
          marginRight: 8,
        }}
      />
    </span>
  );

  if (!initialState) {
    return loading;
  }

  const { currentUser } = initialState;

  if (!currentUser || !currentUser.email) {
    console.log(currentUser);
    return loading;
  }

  const menuItems: ItemType[] = [
    ...(menu
      ? [
          {
            key: 'center',
            icon: <UserOutlined />,
            label: '个人中心',
          },
          {
            key: 'settings',
            icon: <SettingOutlined />,
            label: '个人设置',
          },
          {
            type: 'divider' as const,
          },
        ]
      : []),
    ...(currentUser.email == 'Guest'
      ? [
          {
            key: 'login',
            icon: <LoginOutlined />,
            label: 'Login',
          },
        ]
      : []),
    {
      key: 'credit',
      icon: <WalletOutlined />,
      label: 'My Credit',
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: 'Logout',
    },
  ];

  const menuHeaderDropdown = (
    <Menu className={styles.menu} selectedKeys={[]} onClick={onMenuClick} items={menuItems} />
  );

  if (currentUser.email == 'Guest') {
    return (
      <a
        href="https://colearn-dev.slack.com/openid/connect/authorize?scope=openid%20email&amp;response_type=code&amp;redirect_uri=https%3A%2F%2Fserving-api.colearn.cloud%3A8443%2Foauth&amp;client_id=3293416877427.6805031237120"
        style={{
          alignItems: 'center',
          color: '#000',
          backgroundColor: '#fff',
          border: '1px solid #ddd',
          borderRadius: '4px',
          display: 'inline-flex',
          fontFamily: 'Lato, sans-serif',
          fontSize: '14px',
          fontWeight: 600,
          height: '44px',
          justifyContent: 'center',
          width: '224px',
          marginTop: '4px',
        }}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          style={{
            height: '16px',
            width: '16px',
            marginRight: '12px',
          }}
          viewBox="0 0 122.8 122.8"
        >
          <path
            d="M25.8 77.6c0 7.1-5.8 12.9-12.9 12.9S0 84.7 0 77.6s5.8-12.9 12.9-12.9h12.9v12.9zm6.5 0c0-7.1 5.8-12.9 12.9-12.9s12.9 5.8 12.9 12.9v32.3c0 7.1-5.8 12.9-12.9 12.9s-12.9-5.8-12.9-12.9V77.6z"
            fill="#e01e5a"
          ></path>
          <path
            d="M45.2 25.8c-7.1 0-12.9-5.8-12.9-12.9S38.1 0 45.2 0s12.9 5.8 12.9 12.9v12.9H45.2zm0 6.5c7.1 0 12.9 5.8 12.9 12.9s-5.8 12.9-12.9 12.9H12.9C5.8 58.1 0 52.3 0 45.2s5.8-12.9 12.9-12.9h32.3z"
            fill="#36c5f0"
          ></path>
          <path
            d="M97 45.2c0-7.1 5.8-12.9 12.9-12.9s12.9 5.8 12.9 12.9-5.8 12.9-12.9 12.9H97V45.2zm-6.5 0c0 7.1-5.8 12.9-12.9 12.9s-12.9-5.8-12.9-12.9V12.9C64.7 5.8 70.5 0 77.6 0s12.9 5.8 12.9 12.9v32.3z"
            fill="#2eb67d"
          ></path>
          <path
            d="M77.6 97c7.1 0 12.9 5.8 12.9 12.9s-5.8 12.9-12.9 12.9-12.9-5.8-12.9-12.9V97h12.9zm0-6.5c-7.1 0-12.9-5.8-12.9-12.9s5.8-12.9 12.9-12.9h32.3c7.1 0 12.9 5.8 12.9 12.9s-5.8 12.9-12.9 12.9H77.6z"
            fill="#ecb22e"
          ></path>
        </svg>
        Sign in with Slack
      </a>
    );
  }
  return (
    <HeaderDropdown overlay={menuHeaderDropdown}>
      <span className={`${styles.action} ${styles.account}`}>
        <Avatar size="small" className={styles.avatar} src={currentUser.avatar} alt="avatar" />
        <span className={`${styles.name} anticon`}>
          {currentUser.email ? currentUser.email : currentUser.name}
        </span>
      </span>
    </HeaderDropdown>
  );
};

export default AvatarDropdown;
