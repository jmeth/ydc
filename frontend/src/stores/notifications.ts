/**
 * Notification state store.
 *
 * Manages toast messages (auto-dismiss) and banners (persistent)
 * displayed to the user.
 */

import { defineStore } from 'pinia'

export interface Toast {
  id: string
  level: 'info' | 'success' | 'warning' | 'error'
  title: string
  message: string
}

export interface Banner {
  id: string
  level: 'info' | 'success' | 'warning' | 'error'
  title: string
  message: string
}

interface NotificationState {
  toasts: Toast[]
  banners: Banner[]
}

let nextId = 0

export const useNotificationStore = defineStore('notifications', {
  state: (): NotificationState => ({
    toasts: [],
    banners: [],
  }),

  actions: {
    /**
     * Show a toast notification that auto-dismisses after 5 seconds.
     *
     * @param toast - Toast content (id is assigned automatically)
     */
    showToast(toast: Omit<Toast, 'id'>) {
      const id = `toast-${++nextId}`
      this.toasts.push({ ...toast, id })
      setTimeout(() => this.dismiss(id), 5000)
    },

    /** Show a persistent banner until manually dismissed. */
    showBanner(banner: Omit<Banner, 'id'>) {
      const id = `banner-${++nextId}`
      this.banners.push({ ...banner, id })
    },

    /** Dismiss a toast or banner by id. */
    dismiss(id: string) {
      this.toasts = this.toasts.filter((t) => t.id !== id)
      this.banners = this.banners.filter((b) => b.id !== id)
    },
  },
})
