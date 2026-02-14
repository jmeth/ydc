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
import { useTrainingStore } from '@/stores/training'
import { useCaptureStore } from '@/stores/capture'
import type { WsNotificationMessage, WsTrainingProgressMessage, WsCaptureEventMessage } from '@/types/websocket'

const { lastMessage } = useWebSocket('/ws/events')
const notificationStore = useNotificationStore()
const trainingStore = useTrainingStore()
const captureStore = useCaptureStore()

// Route incoming WebSocket messages to the appropriate stores
watch(lastMessage, (msg) => {
  if (msg && typeof msg === 'object' && 'type' in msg) {
    const wsMsg = msg as { type: string }

    if (wsMsg.type === 'notification') {
      notificationStore.handleServerNotification(msg as WsNotificationMessage)
    } else if (wsMsg.type === 'training.progress') {
      const tp = msg as WsTrainingProgressMessage
      trainingStore.status = 'training'
      trainingStore.updateProgress({
        epoch: tp.epoch,
        totalEpochs: tp.total_epochs,
        loss: tp.loss,
        eta: tp.eta,
      })
    } else if (wsMsg.type === 'capture.event') {
      const ce = msg as WsCaptureEventMessage
      if (ce.capture_type === 'positive') {
        captureStore.updateStats({
          totalCaptures: captureStore.stats.totalCaptures + 1,
          positiveCaptures: captureStore.stats.positiveCaptures + 1,
        })
      } else {
        captureStore.updateStats({
          totalCaptures: captureStore.stats.totalCaptures + 1,
          negativeCaptures: captureStore.stats.negativeCaptures + 1,
        })
      }
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
