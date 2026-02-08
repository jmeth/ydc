/**
 * Notification state store.
 *
 * Manages toast messages (auto-dismiss), banners (persistent),
 * and a history of all server-pushed notifications. The
 * handleServerNotification() action routes incoming WebSocket
 * notification messages to the appropriate UI display method.
 */

import { defineStore } from 'pinia'
import type { WsNotificationMessage } from '@/types/websocket'

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

export interface ServerNotification {
  id: string
  type: string
  level: 'info' | 'success' | 'warning' | 'error'
  category: string
  title: string
  message: string
  timestamp: string
  read: boolean
  data: Record<string, unknown> | null
}

interface NotificationState {
  toasts: Toast[]
  banners: Banner[]
  notifications: ServerNotification[]
}

let nextId = 0

export const useNotificationStore = defineStore('notifications', {
  state: (): NotificationState => ({
    toasts: [],
    banners: [],
    notifications: [],
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

    /**
     * Handle a server-pushed notification from the WebSocket.
     *
     * Routes the notification to the appropriate UI display based
     * on its type: toast → showToast, banner/alert → showBanner,
     * status → silent (history only).
     *
     * @param msg - The parsed WsNotificationMessage from the server.
     */
    handleServerNotification(msg: WsNotificationMessage) {
      // Add to history (newest first, capped at 100)
      this.notifications.unshift({
        id: msg.id,
        type: msg.notification_type,
        level: msg.level,
        category: msg.category,
        title: msg.title,
        message: msg.message,
        timestamp: msg.timestamp,
        read: false,
        data: msg.data,
      })
      if (this.notifications.length > 100) {
        this.notifications = this.notifications.slice(0, 100)
      }

      // Route to UI based on notification type
      const content = { level: msg.level, title: msg.title, message: msg.message }

      switch (msg.notification_type) {
        case 'toast':
          this.showToast(content)
          break
        case 'banner':
        case 'alert':
          this.showBanner(content)
          break
        case 'status':
          // Silent — stored in history only
          break
      }
    },

    /** Mark a notification as read in local history. */
    markRead(id: string) {
      const n = this.notifications.find((n) => n.id === id)
      if (n) n.read = true
    },

    /** Clear all notification history. */
    clearHistory() {
      this.notifications = []
    },
  },
})
