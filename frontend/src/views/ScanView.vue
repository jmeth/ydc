<script setup lang="ts">
/**
 * Scan mode view — inference + capture workflow.
 *
 * Two-column layout: left side shows feed selector, video player
 * with detection overlay, and inference stats; right sidebar shows
 * inference controls (start/stop, model selector, prompts),
 * capture controls, and capture stats.
 *
 * On mount: fetches feeds, datasets, and current inference/capture status.
 */
import { ref, computed, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import { useInferenceStore } from '@/stores/inference'
import { useCaptureStore } from '@/stores/capture'
import { useTrainingStore } from '@/stores/training'
import { useNotificationStore } from '@/stores/notifications'
import VideoPlayer from '@/components/inference/VideoPlayer.vue'
import InferenceStats from '@/components/inference/InferenceStats.vue'
import FeedSelector from '@/components/inference/FeedSelector.vue'
import ModelSelector from '@/components/inference/ModelSelector.vue'
import PromptEditor from '@/components/inference/PromptEditor.vue'
import CaptureControls from '@/components/capture/CaptureControls.vue'
import CaptureStats from '@/components/capture/CaptureStats.vue'

const appStore = useAppStore()
const inferenceStore = useInferenceStore()
const captureStore = useCaptureStore()
const trainingStore = useTrainingStore()
const notificationStore = useNotificationStore()

/** Selected source feed for inference. */
const selectedFeedId = ref<string | null>(null)
/** Selected model for inference. */
const selectedModelPath = ref<string | null>(null)

const isInferenceRunning = computed(() => inferenceStore.status === 'running')

/**
 * The feed ID to display in the video player.
 * When inference is running, show the output (derived) feed;
 * otherwise show the selected raw feed.
 */
const displayFeedId = computed(() => {
  if (isInferenceRunning.value && inferenceStore.outputFeedId) {
    return inferenceStore.outputFeedId
  }
  return selectedFeedId.value
})

/** Start an inference session on the selected feed. */
async function startInference() {
  if (!selectedFeedId.value) return
  try {
    await inferenceStore.startInference(
      selectedFeedId.value,
      selectedModelPath.value || undefined,
      Object.keys(inferenceStore.prompts).length > 0 ? inferenceStore.prompts : undefined,
    )
    // Refresh feeds to see the new derived feed
    await appStore.fetchFeeds()
  } catch (err) {
    notificationStore.showToast('Failed to start inference', 'error')
  }
}

/** Stop the current inference session. */
async function stopInference() {
  try {
    // Stop capture first if running
    if (captureStore.status === 'running') {
      await captureStore.stopCapture()
    }
    await inferenceStore.stopInference()
    await appStore.fetchFeeds()
  } catch (err) {
    notificationStore.showToast('Failed to stop inference', 'error')
  }
}

/** Save updated prompts to the active session. */
async function savePrompts(prompts: Record<number, string[]>) {
  try {
    await inferenceStore.updatePrompts(prompts)
    notificationStore.showToast('Prompts updated', 'success')
  } catch (err) {
    notificationStore.showToast('Failed to update prompts', 'error')
  }
}

onMounted(async () => {
  await Promise.all([
    appStore.fetchFeeds(),
    appStore.fetchDatasets(),
    inferenceStore.fetchStatus(),
    captureStore.fetchStatus(),
    trainingStore.fetchModels(),
  ])

  // Restore selected feed if inference is already running
  if (inferenceStore.sourceFeedId) {
    selectedFeedId.value = inferenceStore.sourceFeedId
  }
  if (inferenceStore.modelId) {
    selectedModelPath.value = inferenceStore.modelId
  }
})
</script>

<template>
  <div class="scan-layout">
    <!-- Left: Video feed area -->
    <div class="scan-main">
      <FeedSelector
        v-model="selectedFeedId"
      />

      <VideoPlayer :feed-id="displayFeedId" />

      <InferenceStats
        v-if="isInferenceRunning"
        :fps="inferenceStore.stats.fps"
        :detection-count="inferenceStore.stats.detections"
        :inference-time-ms="inferenceStore.stats.inferenceTimeMs"
      />

      <CaptureStats />
    </div>

    <!-- Right: Controls sidebar -->
    <div class="scan-sidebar">
      <!-- Inference controls -->
      <div class="card">
        <div class="card-header">Inference</div>

        <ModelSelector v-model="selectedModelPath" />

        <div class="sidebar-actions">
          <button
            v-if="!isInferenceRunning"
            class="btn btn-primary"
            :disabled="!selectedFeedId"
            @click="startInference"
          >
            Start Inference
          </button>
          <button
            v-else
            class="btn btn-danger"
            @click="stopInference"
          >
            Stop Inference
          </button>
        </div>
      </div>

      <!-- Prompt editor -->
      <PromptEditor
        :prompts="inferenceStore.prompts"
        :disabled="!isInferenceRunning"
        @save="savePrompts"
      />

      <!-- Capture controls -->
      <CaptureControls
        :inference-running="isInferenceRunning"
        :output-feed-id="inferenceStore.outputFeedId"
      />
    </div>
  </div>
</template>

<style scoped>
.scan-layout {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 1rem;
  height: 100%;
}

.scan-main {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.scan-sidebar {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.sidebar-actions {
  margin-top: 0.75rem;
}

@media (max-width: 900px) {
  .scan-layout {
    grid-template-columns: 1fr;
  }
}
</style>
