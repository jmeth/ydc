/**
 * Composable for WebSocket connection lifecycle.
 *
 * Manages connection, auto-reconnect on close, and provides a
 * reactive `isConnected` ref and typed `send` method. Automatically
 * connects on component mount and disconnects on unmount.
 *
 * @param url - WebSocket URL path (e.g. "/ws/events")
 */

import { ref, onMounted, onUnmounted } from 'vue'

export function useWebSocket(url: string) {
  const socket = ref<WebSocket | null>(null)
  const isConnected = ref(false)
  const lastMessage = ref<unknown>(null)

  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let intentionalClose = false

  /** Open a WebSocket connection and wire up event handlers. */
  function connect() {
    // Build absolute WS URL from relative path
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}${url}`

    socket.value = new WebSocket(wsUrl)

    socket.value.onopen = () => {
      isConnected.value = true
    }

    socket.value.onmessage = (event: MessageEvent) => {
      try {
        lastMessage.value = JSON.parse(event.data)
      } catch {
        lastMessage.value = event.data
      }
    }

    socket.value.onclose = () => {
      isConnected.value = false
      // Auto-reconnect after 2 seconds unless intentionally closed
      if (!intentionalClose) {
        reconnectTimer = setTimeout(connect, 2000)
      }
    }

    socket.value.onerror = () => {
      // onclose will fire after onerror, triggering reconnect
    }
  }

  /**
   * Send JSON data through the WebSocket.
   * @param data - Object to JSON-serialize and send
   */
  function send(data: unknown) {
    if (socket.value?.readyState === WebSocket.OPEN) {
      socket.value.send(JSON.stringify(data))
    }
  }

  onMounted(connect)

  onUnmounted(() => {
    intentionalClose = true
    if (reconnectTimer) clearTimeout(reconnectTimer)
    socket.value?.close()
  })

  return { isConnected, lastMessage, send }
}
