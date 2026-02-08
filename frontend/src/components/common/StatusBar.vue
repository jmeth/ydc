<script setup lang="ts">
/**
 * Bottom status bar.
 *
 * Displays connection status, capture stats, and training
 * progress at a glance.
 */
import { useAppStore } from '@/stores/app'
import { useCaptureStore } from '@/stores/capture'
import { useTrainingStore } from '@/stores/training'

const appStore = useAppStore()
const captureStore = useCaptureStore()
const trainingStore = useTrainingStore()
</script>

<template>
  <footer class="status-bar">
    <span class="status-item">
      <span
        class="status-dot"
        :class="appStore.isConnected ? 'connected' : 'disconnected'"
      ></span>
      {{ appStore.isConnected ? 'Connected' : 'Disconnected' }}
    </span>

    <span v-if="captureStore.status === 'running'" class="status-item">
      Captures: {{ captureStore.stats.totalCaptures }}
    </span>

    <span v-if="trainingStore.status === 'training'" class="status-item">
      Training: {{ trainingStore.progress.epoch }}/{{ trainingStore.progress.totalEpochs }}
    </span>
  </footer>
</template>

<style scoped>
.status-bar {
  display: flex;
  gap: 1.5rem;
  padding: 0.375rem 1rem;
  background: var(--color-bg-header);
  border-top: 1px solid var(--color-border);
  font-size: 0.75rem;
  color: var(--color-text-muted);
}

.status-item {
  display: flex;
  align-items: center;
  gap: 0.375rem;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-dot.connected {
  background: var(--color-success);
}

.status-dot.disconnected {
  background: var(--color-error);
}
</style>
