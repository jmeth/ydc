/**
 * Vue Router configuration.
 *
 * Defines routes for the four main application modes.
 * Views are lazy-loaded to reduce initial bundle size.
 */

import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      redirect: '/scan',
    },
    {
      path: '/scan',
      name: 'scan',
      component: () => import('@/views/ScanView.vue'),
    },
    {
      path: '/dataset',
      name: 'dataset',
      component: () => import('@/views/DatasetView.vue'),
    },
    {
      path: '/train',
      name: 'train',
      component: () => import('@/views/TrainView.vue'),
    },
    {
      path: '/model',
      name: 'model',
      component: () => import('@/views/ModelView.vue'),
    },
  ],
})

export default router
