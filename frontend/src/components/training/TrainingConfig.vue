<script setup lang="ts">
/**
 * Training configuration form.
 *
 * Allows configuring dataset, base model, epochs, batch size, image size,
 * and advanced options (patience, freeze layers, learning rates).
 * Start Training button triggers training via the store.
 *
 * Props:
 *   disabled - Whether the form is disabled (e.g. training active)
 *
 * Emits:
 *   start - When start training is clicked
 */
import { ref, computed } from 'vue'
import { useAppStore } from '@/stores/app'
import { useTrainingStore } from '@/stores/training'
import { useNotificationStore } from '@/stores/notifications'

const props = defineProps<{
  disabled?: boolean
}>()

const appStore = useAppStore()
const trainingStore = useTrainingStore()
const notificationStore = useNotificationStore()

/** Form state. */
const datasetName = ref('')
const baseModel = ref('yolo11n.pt')
const epochs = ref(100)
const batchSize = ref(16)
const imageSize = ref(640)
const modelName = ref('')
const showAdvanced = ref(false)
const patience = ref(50)
const freezeLayers = ref(0)
const lr0 = ref(0.01)
const lrf = ref(0.01)

/** Base model presets. */
const modelPresets = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']

const canStart = computed(() =>
  datasetName.value && !props.disabled
)

/** Start a training job with the current configuration. */
async function startTraining() {
  if (!canStart.value) return
  try {
    await trainingStore.startTraining({
      datasetName: datasetName.value,
      baseModel: baseModel.value,
      epochs: epochs.value,
      batchSize: batchSize.value,
      imageSize: imageSize.value,
      patience: patience.value,
      freezeLayers: freezeLayers.value,
      lr0: lr0.value,
      lrf: lrf.value,
      modelName: modelName.value || undefined,
    })
    notificationStore.showToast('Training started', 'success')
  } catch (err) {
    notificationStore.showToast('Failed to start training', 'error')
  }
}
</script>

<template>
  <div class="card">
    <div class="card-header">Training Configuration</div>

    <!-- Dataset and model selection -->
    <div class="config-row">
      <div class="form-group" style="flex: 1">
        <label class="form-label">Dataset</label>
        <select v-model="datasetName" class="form-select" :disabled="props.disabled">
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
      <div class="form-group" style="flex: 1">
        <label class="form-label">Base Model</label>
        <select v-model="baseModel" class="form-select" :disabled="props.disabled">
          <option v-for="m in modelPresets" :key="m" :value="m">{{ m }}</option>
        </select>
      </div>
    </div>

    <!-- Basic parameters -->
    <div class="config-row">
      <div class="form-group">
        <label class="form-label">Epochs</label>
        <input v-model.number="epochs" type="number" class="form-input" min="1" :disabled="props.disabled" />
      </div>
      <div class="form-group">
        <label class="form-label">Batch Size</label>
        <input v-model.number="batchSize" type="number" class="form-input" min="1" :disabled="props.disabled" />
      </div>
      <div class="form-group">
        <label class="form-label">Image Size</label>
        <input v-model.number="imageSize" type="number" class="form-input" min="32" step="32" :disabled="props.disabled" />
      </div>
    </div>

    <!-- Model name -->
    <div class="form-group">
      <label class="form-label">Model Name (optional)</label>
      <input
        v-model="modelName"
        class="form-input"
        placeholder="Auto-generated if empty"
        :disabled="props.disabled"
      />
    </div>

    <!-- Advanced section -->
    <button class="btn btn-sm advanced-toggle" @click="showAdvanced = !showAdvanced">
      {{ showAdvanced ? 'Hide' : 'Show' }} Advanced
    </button>

    <div v-if="showAdvanced" class="config-row">
      <div class="form-group">
        <label class="form-label">Patience</label>
        <input v-model.number="patience" type="number" class="form-input" min="0" :disabled="props.disabled" />
      </div>
      <div class="form-group">
        <label class="form-label">Freeze Layers</label>
        <input v-model.number="freezeLayers" type="number" class="form-input" min="0" :disabled="props.disabled" />
      </div>
      <div class="form-group">
        <label class="form-label">LR0</label>
        <input v-model.number="lr0" type="number" class="form-input" min="0" step="0.001" :disabled="props.disabled" />
      </div>
      <div class="form-group">
        <label class="form-label">LRF</label>
        <input v-model.number="lrf" type="number" class="form-input" min="0" step="0.001" :disabled="props.disabled" />
      </div>
    </div>

    <button
      class="btn btn-primary start-btn"
      :disabled="!canStart"
      @click="startTraining"
    >
      Start Training
    </button>
  </div>
</template>

<style scoped>
.config-row {
  display: flex;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}

.config-row .form-group {
  flex: 1;
}

.advanced-toggle {
  margin-bottom: 0.75rem;
}

.start-btn {
  margin-top: 0.5rem;
  width: 100%;
}
</style>
