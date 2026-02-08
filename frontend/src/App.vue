<script setup lang="ts">
/**
 * Root application component.
 *
 * Renders the app shell (header, nav, content, status bar) and
 * connects to the /ws/events WebSocket to route server-pushed
 * notifications into the notification store.
 */
import { watch } from 'vue'
import AppHeader from '@/components/common/AppHeader.vue'
import AppNav from '@/components/common/AppNav.vue'
import StatusBar from '@/components/common/StatusBar.vue'
import Toast from '@/components/common/Toast.vue'
import Banner from '@/components/common/Banner.vue'
import { useWebSocket } from '@/composables/useWebSocket'
import { useNotificationStore } from '@/stores/notifications'
import type { WsNotificationMessage } from '@/types/websocket'

const { lastMessage } = useWebSocket('/ws/events')
const notificationStore = useNotificationStore()

// Route incoming WebSocket messages to the notification store
watch(lastMessage, (msg) => {
  if (msg && typeof msg === 'object' && 'type' in msg) {
    const wsMsg = msg as { type: string }
    if (wsMsg.type === 'notification') {
      notificationStore.handleServerNotification(msg as WsNotificationMessage)
    }
  }
})
</script>

<template>
  <div class="app">
    <Banner />
    <AppHeader />
    <AppNav />
    <main class="app-content">
      <RouterView />
    </main>
    <StatusBar />
    <Toast />
  </div>
</template>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: var(--color-bg);
  color: var(--color-text);
}

.app-content {
  flex: 1;
  padding: 1rem;
}
</style>
