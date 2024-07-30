export default [
  {
    path: '/user',
    layout: false,
    routes: [
      { name: 'Login', path: '/user/login', component: './user/Login' },
      { component: './404' },
    ],
  },
  { path: '/account/credit', hideInMenu: true, name: 'Credit', component: './Credit' },

  { path: '/', redirect: '/home' },
  { path: '/login', hideInMenu: true, name: 'Login', component: './Token' },
  { path: '/home', name: 'Home', icon: 'crown', component: './Home' },
  { path: '/network', name: 'Network', icon: 'desktop', component: './Network' },
  { path: '/playground', name: 'Playground', icon: 'playSquare', component: './Playground' },
  { path: '/scheduler', name: 'Scheduler', icon: 'cluster', component: './Scheduler' },
  {
    path: '/docs',
    name: 'Docs',
    icon: 'read',
    children: [
      {
        path: '/docs/getting_started',
        name: 'Get Started',
        // icon: "star"
      },
      {
        path: '/docs/private_ent',
        name: 'Self-host Private Network',
        // icon: "cluster"
      },
    ],
  },
  // { path: '/controller', name: 'Controller', icon: 'cluster', component: './Controller' },
  // { path: '/worker', name: 'Worker', icon: 'desktop' },
  // { path: '/simulator', name: 'Simulator', icon: 'experiment' },
  {
    path: 'https://llm-speedtest.colearn.cloud/',
    name: 'SpeedTest',
    icon: 'dashboard',
    target: '_blank',
  },
  { component: './404' },
];
