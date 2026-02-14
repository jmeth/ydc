<script setup lang="ts">
/**
 * Model mode view — model management and inference testing.
 *
 * Three-panel layout: ModelList (left), FeedSelector + VideoPlayer
 * (center), DetectionStats + InferenceStats (right). Select a model,
 * select a feed, start inference, and see live results.
 *
 * On mount: fetches models, feeds, and current inference status.
 */
import { ref, computed, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import { useInferenceStore } from '@/stores/inference'
import { useTrainingStore } from '@/stores/training'
import { useNotificationStore } from '@/stores/notifications'
import DownloadModelForm from '@/components/training/DownloadModelForm.vue'
import ModelList from '@/components/training/ModelList.vue'
import FeedSelector from '@/components/inference/FeedSelector.vue'
import VideoPlayer from '@/components/inference/VideoPlayer.vue'
import InferenceStats from '@/components/inference/InferenceStats.vue'
import DetectionStats from '@/components/inference/DetectionStats.vue'
import type { Detection } from '@/types/models'

const appStore = useAppStore()
const inferenceStore = useInferenceStore()
const trainingStore = useTrainingStore()
const notificationStore = useNotificationStore()

/** Selected feed and model for testing. */
const selectedFeedId = ref<string | null>(null)
const selectedModelName = ref<string | null>(null)

/** Live detections from the video player. */
const videoPlayerRef = ref<InstanceType<typeof VideoPlayer> | null>(null)

const isInferenceRunning = computed(() => inferenceStore.status === 'running')

/** Active detections from the video stream. */
const currentDetections = computed<Detection[]>(() => {
  return videoPlayerRef.value?.detections ?? []
})

/** Feed to display: output feed when inference is running, else raw feed. */
const displayFeedId = computed(() => {
  if (isInferenceRunning.value && inferenceStore.outputFeedId) {
    return inferenceStore.outputFeedId
  }
  return selectedFeedId.value
})

/**
 * Handle model selection from the model list.
 * Finds the model path for inference.
 *
 * @param name - Model name
 */
function onSelectModel(name: string) {
  selectedModelName.value = name
}

/** Start inference with the selected model and feed. */
async function startInference() {
  if (!selectedFeedId.value || !selectedModelName.value) return
  try {
    await inferenceStore.startInference(selectedFeedId.value, selectedModelName.value)
    await appStore.fetchFeeds()
  } catch {
    notificationStore.showToast('Failed to start inference', 'error')
  }
}

/** Stop the current inference session. */
async function stopInference() {
  try {
    await inferenceStore.stopInference()
    await appStore.fetchFeeds()
  } catch {
    notificationStore.showToast('Failed to stop inference', 'error')
  }
}

onMounted(async () => {
  await Promise.all([
    appStore.fetchFeeds(),
    trainingStore.fetchModels(),
    inferenceStore.fetchStatus(),
  ])

  // Restore state if inference is already running
  if (inferenceStore.sourceFeedId) {
    selectedFeedId.value = inferenceStore.sourceFeedId
  }
})
</script>

<template>
  <div class="model-layout">
    <!-- Left: Download form + Model list -->
    <div class="model-sidebar">
      <DownloadModelForm />
      <ModelList @select="onSelectModel" />
    </div>

    <!-- Center: Video + controls -->
    <div class="model-main">
      <div class="model-controls">
        <FeedSelector v-model="selectedFeedId" />

        <div v-if="selectedModelName" class="selected-model">
          <span class="form-label">Model:</span>
          <span class="model-name-display">{{ selectedModelName }}</span>
        </div>

        <div class="model-actions">
          <button
            v-if="!isInferenceRunning"
            class="btn btn-primary"
            :disabled="!selectedFeedId || !selectedModelName"
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

      <VideoPlayer ref="videoPlayerRef" :feed-id="displayFeedId" />

      <InferenceStats
        v-if="isInferenceRunning"
        :fps="inferenceStore.stats.fps"
        :detection-count="inferenceStore.stats.detections"
        :inference-time-ms="inferenceStore.stats.inferenceTimeMs"
      />
    </div>

    <!-- Right: Detection stats -->
    <div class="model-stats">
      <DetectionStats :detections="currentDetections" />
    </div>
  </div>
</template>

<style scoped>
.model-layout {
  display: grid;
  grid-template-columns: 280px 1fr 240px;
  gap: 1rem;
  height: 100%;
}

.model-sidebar {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  overflow-y: auto;
}

.model-main {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.model-controls {
  display: flex;
  align-items: flex-end;
  gap: 0.75rem;
}

.model-controls > :first-child {
  flex: 1;
}

.selected-model {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  padding-bottom: 0.5rem;
}

.model-name-display {
  font-weight: 600;
  font-size: 0.875rem;
}

.model-actions {
  padding-bottom: 0.25rem;
}

.model-stats {
  overflow-y: auto;
}

@media (max-width: 1100px) {
  .model-layout {
    grid-template-columns: 1fr;
  }
}
</style>
