<script setup lang="ts">
/**
 * Training progress display.
 *
 * Shows progress bar (epoch/total), loss, metrics, elapsed time,
 * and a stop button. Displayed when training status is 'training'.
 *
 * Props:
 *   (reads from training store directly)
 */
import { computed } from 'vue'
import { useTrainingStore } from '@/stores/training'
import { useNotificationStore } from '@/stores/notifications'

const trainingStore = useTrainingStore()
const notificationStore = useNotificationStore()

const progress = computed(() => trainingStore.progress)
const status = computed(() => trainingStore.status)

/** Progress bar percentage. */
const pct = computed(() => {
  if (progress.value.totalEpochs === 0) return 0
  return Math.min(100, (progress.value.epoch / progress.value.totalEpochs) * 100)
})

/** Stop the current training job. */
async function stopTraining() {
  try {
    await trainingStore.stopTraining()
    notificationStore.showToast('Training stopped', 'info')
  } catch {
    notificationStore.showToast('Failed to stop training', 'error')
  }
}
</script>

<template>
  <div v-if="status === 'training'" class="card progress-card">
    <div class="card-header">Training Progress</div>

    <!-- Progress bar -->
    <div class="progress-bar-container">
      <div class="progress-bar" :style="{ width: pct + '%' }"></div>
    </div>

    <!-- Stats row -->
    <div class="stat-row">
      <div class="stat-item">
        <span class="stat-label">Epoch</span>
        <span class="stat-value">{{ progress.epoch }} / {{ progress.totalEpochs }}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Loss</span>
        <span class="stat-value">{{ progress.loss.toFixed(4) }}</span>
      </div>
      <div v-if="progress.metrics.mAP50" class="stat-item">
        <span class="stat-label">mAP@50</span>
        <span class="stat-value">{{ (progress.metrics.mAP50 * 100).toFixed(1) }}%</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Progress</span>
        <span class="stat-value">{{ progress.progressPct.toFixed(1) }}%</span>
      </div>
    </div>

    <button class="btn btn-danger stop-btn" @click="stopTraining">
      Stop Training
    </button>
  </div>

  <!-- Completion / error messages -->
  <div v-else-if="status === 'completed'" class="card">
    <div class="card-header">Training Complete</div>
    <p class="completion-msg">Training completed successfully.</p>
    <div v-if="progress.metrics.mAP50" class="stat-row">
      <div class="stat-item">
        <span class="stat-label">Best mAP@50</span>
        <span class="stat-value">{{ (progress.metrics.mAP50 * 100).toFixed(1) }}%</span>
      </div>
    </div>
  </div>

  <div v-else-if="status === 'error'" class="card">
    <div class="card-header" style="color: var(--color-error)">Training Error</div>
    <p class="error-msg">{{ trainingStore.error || 'An unknown error occurred.' }}</p>
  </div>
</template>

<style scoped>
.progress-card {
  border-color: var(--color-accent);
}

.progress-bar-container {
  height: 8px;
  background: var(--color-bg);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 0.75rem;
}

.progress-bar {
  height: 100%;
  background: var(--color-accent);
  transition: width 0.3s ease;
  border-radius: 4px;
}

.stop-btn {
  margin-top: 0.75rem;
}

.completion-msg {
  color: var(--color-success);
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
}

.error-msg {
  color: var(--color-error);
  font-size: 0.875rem;
}
</style>
