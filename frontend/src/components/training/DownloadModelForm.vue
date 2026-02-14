<script setup lang="ts">
/**
 * Form for downloading pretrained YOLO models.
 *
 * Provides a dropdown of common pretrained models (YOLO11, YOLOv8),
 * an option for custom model identifiers, an optional display name,
 * and a download button with loading/error/success feedback.
 *
 * Emits:
 *   downloaded - When a model is successfully downloaded
 */
import { ref, computed } from 'vue'
import { useTrainingStore } from '@/stores/training'
import { useNotificationStore } from '@/stores/notifications'

const emit = defineEmits<{
  downloaded: []
}>()

const trainingStore = useTrainingStore()
const notificationStore = useNotificationStore()

/** Well-known pretrained models available for download. */
const PRETRAINED_MODELS = [
  { value: 'yolo11n.pt', label: 'YOLO11 Nano (yolo11n.pt)' },
  { value: 'yolo11s.pt', label: 'YOLO11 Small (yolo11s.pt)' },
  { value: 'yolo11m.pt', label: 'YOLO11 Medium (yolo11m.pt)' },
  { value: 'yolo11l.pt', label: 'YOLO11 Large (yolo11l.pt)' },
  { value: 'yolo11x.pt', label: 'YOLO11 XLarge (yolo11x.pt)' },
  { value: 'yolov8n.pt', label: 'YOLOv8 Nano (yolov8n.pt)' },
  { value: 'yolov8s.pt', label: 'YOLOv8 Small (yolov8s.pt)' },
  { value: 'yolov8m.pt', label: 'YOLOv8 Medium (yolov8m.pt)' },
]

const selectedModel = ref('')
const customModelId = ref('')
const customName = ref('')
const isDownloading = ref(false)

/** Whether the user has selected the "custom" option. */
const isCustom = computed(() => selectedModel.value === '__custom__')

/** The effective model ID to download. */
const effectiveModelId = computed(() =>
  isCustom.value ? customModelId.value.trim() : selectedModel.value
)

/** Whether the form is valid for submission. */
const canSubmit = computed(() =>
  !isDownloading.value && effectiveModelId.value.length > 0
)

/** Download the selected pretrained model. */
async function handleDownload() {
  if (!canSubmit.value) return

  isDownloading.value = true
  try {
    const name = customName.value.trim() || undefined
    await trainingStore.downloadPretrained(effectiveModelId.value, name)
    notificationStore.showToast(
      `Downloaded model "${effectiveModelId.value}"`,
      'success',
    )
    // Reset form
    selectedModel.value = ''
    customModelId.value = ''
    customName.value = ''
    emit('downloaded')
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Download failed'
    notificationStore.showToast(message, 'error')
  } finally {
    isDownloading.value = false
  }
}
</script>

<template>
  <div class="download-form card">
    <div class="card-header">Download Pretrained Model</div>

    <div class="form-group">
      <label class="form-label" for="model-select">Model</label>
      <select
        id="model-select"
        v-model="selectedModel"
        class="form-select"
        :disabled="isDownloading"
      >
        <option value="" disabled>Select a model...</option>
        <option
          v-for="m in PRETRAINED_MODELS"
          :key="m.value"
          :value="m.value"
        >
          {{ m.label }}
        </option>
        <option value="__custom__">Custom model ID...</option>
      </select>
    </div>

    <div v-if="isCustom" class="form-group">
      <label class="form-label" for="custom-model-id">Model ID</label>
      <input
        id="custom-model-id"
        v-model="customModelId"
        class="form-input"
        type="text"
        placeholder="e.g. yolov8x.pt"
        :disabled="isDownloading"
      />
    </div>

    <div class="form-group">
      <label class="form-label" for="model-name">Name (optional)</label>
      <input
        id="model-name"
        v-model="customName"
        class="form-input"
        type="text"
        placeholder="Auto-generated from model ID"
        :disabled="isDownloading"
      />
    </div>

    <button
      class="btn btn-primary download-btn"
      :disabled="!canSubmit"
      @click="handleDownload"
    >
      <span v-if="isDownloading" class="loading-spinner loading-spinner-sm" />
      {{ isDownloading ? 'Downloading...' : 'Download' }}
    </button>
  </div>
</template>

<style scoped>
.download-form {
  display: flex;
  flex-direction: column;
  gap: 0.625rem;
}

.download-btn {
  align-self: stretch;
  justify-content: center;
  margin-top: 0.25rem;
}
</style>
