// @ts-ignore
/* eslint-disable */
import { request } from 'umi';
import { sha256 } from 'js-sha256';

/** 获取当前的用户 GET /api/currentUser */
export async function currentUser(options?: { [key: string]: any }) {
  // first, try to make request to url using the existing cookie
  try {
    // throw new Error('Network response was not ok.');  // TODO: temporary for testing without backend
    const user_url = 'https://serving-api.colearn.cloud:8443/get_user';

    const response = await fetch(user_url, {
      headers: { 'api-token': localStorage.getItem('token') },
    });
    console.log(response);
    if (response.ok) {
      const data = await response.json();
      const email = data.email;
      const address = String(email).trim().toLowerCase();
      const hash = sha256(address);
      data.avatar = `https://www.gravatar.com/avatar/${hash}`;
      console.log(data);
      return data;
    } else {
      throw new Error('Network response was not ok.');
    }
  } catch (e) {
    return {
      email: 'Guest',
      avatar: '/icons8-who-100.png',
    };
    // return request<{
    //   data: API.CurrentUser;
    // }>('/api/currentUser', {
    //   method: 'GET',
    //   ...(options || {}),
    // });
  }
}

/** 退出登录接口 POST /api/login/outLogin */
export async function outLogin(options?: { [key: string]: any }) {
  return request<Record<string, any>>('/api/login/outLogin', {
    method: 'POST',
    ...(options || {}),
  });
}

/** 登录接口 POST /api/login/account */
export async function login(body: API.LoginParams, options?: { [key: string]: any }) {
  return request<API.LoginResult>('/api/login/account', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  });
}

/** 此处后端没有提供注释 GET /api/notices */
export async function getNotices(options?: { [key: string]: any }) {
  return request<API.NoticeIconList>('/api/notices', {
    method: 'GET',
    ...(options || {}),
  });
}

/** 获取规则列表 GET /api/rule */
export async function rule(
  params: {
    // query
    /** 当前的页码 */
    current?: number;
    /** 页面的容量 */
    pageSize?: number;
  },
  options?: { [key: string]: any },
) {
  return request<API.RuleList>('/api/rule', {
    method: 'GET',
    params: {
      ...params,
    },
    ...(options || {}),
  });
}

/** 新建规则 PUT /api/rule */
export async function updateRule(options?: { [key: string]: any }) {
  return request<API.RuleListItem>('/api/rule', {
    method: 'PUT',
    ...(options || {}),
  });
}

/** 新建规则 POST /api/rule */
export async function addRule(options?: { [key: string]: any }) {
  return request<API.RuleListItem>('/api/rule', {
    method: 'POST',
    ...(options || {}),
  });
}

/** 删除规则 DELETE /api/rule */
export async function removeRule(options?: { [key: string]: any }) {
  return request<Record<string, any>>('/api/rule', {
    method: 'DELETE',
    ...(options || {}),
  });
}
