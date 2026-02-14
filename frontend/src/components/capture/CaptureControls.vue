<script setup lang="ts">
/**
 * Controls for managing frame capture sessions.
 *
 * Provides start/stop capture, manual trigger, dataset/split selectors,
 * and configuration inputs (interval, negative ratio, confidence threshold).
 * Supports both raw capture (from a source feed) and inference capture
 * (from an inference output feed).
 *
 * Props:
 *   inferenceRunning - Whether an inference session is active
 *   outputFeedId - Feed ID of the inference output (used for inference capture)
 *   sourceFeedId - Feed ID of the raw source feed (used for raw capture)
 */
import { ref, computed } from 'vue'
import { useAppStore } from '@/stores/app'
import { useCaptureStore } from '@/stores/capture'
import { useNotificationStore } from '@/stores/notifications'

const props = defineProps<{
  inferenceRunning: boolean
  outputFeedId: string | null
  sourceFeedId: string | null
}>()

const appStore = useAppStore()
const captureStore = useCaptureStore()
const notificationStore = useNotificationStore()

/** Selected dataset for capture target. */
const selectedDataset = ref('')
/** Selected split for capture target. */
const selectedSplit = ref('train')
/** Local config inputs. */
const interval = ref(captureStore.config.captureInterval)
const negativeRatio = ref(captureStore.config.negativeRatio)
const confidenceThreshold = ref(captureStore.config.confidenceThreshold)

const isRunning = computed(() => captureStore.status === 'running')

/** The feed ID to capture from: inference output if running, else raw source. */
const captureFeedId = computed(() =>
  props.inferenceRunning && props.outputFeedId
    ? props.outputFeedId
    : props.sourceFeedId
)

const canStart = computed(() =>
  captureFeedId.value &&
  selectedDataset.value &&
  !isRunning.value
)

/** Start a capture session with the current configuration. */
async function startCapture() {
  if (!captureFeedId.value || !selectedDataset.value) return
  try {
    captureStore.config.captureInterval = interval.value
    captureStore.config.negativeRatio = negativeRatio.value
    captureStore.config.confidenceThreshold = confidenceThreshold.value
    await captureStore.startCapture(captureFeedId.value, selectedDataset.value, selectedSplit.value)
  } catch (err) {
    notificationStore.showToast('Failed to start capture', 'error')
  }
}

/** Stop the active capture session. */
async function stopCapture() {
  try {
    await captureStore.stopCapture()
  } catch (err) {
    notificationStore.showToast('Failed to stop capture', 'error')
  }
}

/** Manually capture a single frame. */
async function triggerCapture() {
  try {
    const res = await captureStore.triggerManualCapture()
    notificationStore.showToast(`Captured ${res.filename}`, 'success')
  } catch (err) {
    notificationStore.showToast('Failed to capture frame', 'error')
  }
}
</script>

<template>
  <div class="card">
    <div class="card-header">Capture</div>

    <div v-if="!captureFeedId" class="empty-state">
      <div class="empty-state-text">Select a feed to enable capture</div>
    </div>

    <template v-else>
      <!-- Dataset and split selection -->
      <div class="capture-row">
        <div class="form-group" style="flex: 1">
          <label class="form-label">Dataset</label>
          <select v-model="selectedDataset" class="form-select" :disabled="isRunning">
            <option value="">Select dataset...</option>
            <option
              v-for="ds in appStore.datasets"
              :key="ds.name"
              :value="ds.name"
            >
              {{ ds.name }}
            </option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">Split</label>
          <select v-model="selectedSplit" class="form-select" :disabled="isRunning">
            <option value="train">Train</option>
            <option value="val">Val</option>
            <option value="test">Test</option>
          </select>
        </div>
      </div>

      <!-- Configuration -->
      <div class="capture-row">
        <div class="form-group">
          <label class="form-label">Interval (s)</label>
          <input
            v-model.number="interval"
            type="number"
            class="form-input"
            min="0.1"
            step="0.5"
            :disabled="isRunning"
          />
        </div>
        <div class="form-group">
          <label class="form-label">Neg. Ratio</label>
          <input
            v-model.number="negativeRatio"
            type="number"
            class="form-input"
            min="0"
            max="1"
            step="0.05"
            :disabled="isRunning"
          />
        </div>
        <div class="form-group">
          <label class="form-label">Conf. Thresh</label>
          <input
            v-model.number="confidenceThreshold"
            type="number"
            class="form-input"
            min="0"
            max="1"
            step="0.05"
            :disabled="isRunning"
          />
        </div>
      </div>

      <!-- Action buttons -->
      <div class="capture-actions">
        <button
          v-if="!isRunning"
          class="btn btn-primary"
          :disabled="!canStart"
          @click="startCapture"
        >
          Start Capture
        </button>
        <button
          v-else
          class="btn btn-danger"
          @click="stopCapture"
        >
          Stop Capture
        </button>
        <button
          class="btn"
          :disabled="!isRunning"
          @click="triggerCapture"
        >
          Capture Now
        </button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.capture-row {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.capture-actions {
  display: flex;
  gap: 0.5rem;
}
</style>
