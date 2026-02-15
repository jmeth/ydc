<script setup lang="ts">
/**
 * Training configuration form.
 *
 * Allows configuring dataset, base model, epochs, batch size, image size,
 * and advanced options (patience, freeze layers, learning rates, data
 * augmentation). Start Training button triggers training via the store.
 *
 * Props:
 *   disabled - Whether the form is disabled (e.g. training active)
 *
 * Emits:
 *   start - When start training is clicked
 */
import { ref, reactive, computed } from 'vue'
import { useAppStore } from '@/stores/app'
import { useTrainingStore } from '@/stores/training'
import { useNotificationStore } from '@/stores/notifications'
import type { AugmentationConfig } from '@/types/api'

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
const showAugmentation = ref(false)
const patience = ref(50)
const freezeLayers = ref(0)
const lr0 = ref(0.01)
const lrf = ref(0.01)

/**
 * Augmentation state — undefined means "use ultralytics default".
 * When the user enables a field, its value is set to the ultralytics
 * default so they can adjust from there.
 */
const augEnabled = reactive<Record<string, boolean>>({})
const augValues = reactive<Record<string, number | string | null>>({})

/** Augmentation parameter definitions grouped by category. */
const augGroups = [
  {
    label: 'Color Space',
    params: [
      { key: 'hsv_h', label: 'HSV Hue', default: 0.015, min: 0, max: 1, step: 0.005, desc: 'Hue shift fraction' },
      { key: 'hsv_s', label: 'HSV Saturation', default: 0.7, min: 0, max: 1, step: 0.05, desc: 'Saturation shift fraction' },
      { key: 'hsv_v', label: 'HSV Value', default: 0.4, min: 0, max: 1, step: 0.05, desc: 'Brightness shift fraction' },
    ],
  },
  {
    label: 'Geometric',
    params: [
      { key: 'degrees', label: 'Rotation', default: 0, min: 0, max: 180, step: 5, desc: 'Max rotation degrees' },
      { key: 'translate', label: 'Translate', default: 0.1, min: 0, max: 1, step: 0.05, desc: 'Max shift fraction' },
      { key: 'scale', label: 'Scale', default: 0.5, min: 0, max: 1, step: 0.05, desc: 'Scale variation factor' },
      { key: 'shear', label: 'Shear', default: 0, min: -180, max: 180, step: 5, desc: 'Shear angle degrees' },
      { key: 'perspective', label: 'Perspective', default: 0, min: 0, max: 0.001, step: 0.0001, desc: 'Perspective warp' },
      { key: 'flipud', label: 'Flip Vertical', default: 0, min: 0, max: 1, step: 0.1, desc: 'Vertical flip probability' },
      { key: 'fliplr', label: 'Flip Horizontal', default: 0.5, min: 0, max: 1, step: 0.1, desc: 'Horizontal flip probability' },
      { key: 'bgr', label: 'BGR Swap', default: 0, min: 0, max: 1, step: 0.1, desc: 'Channel swap probability' },
    ],
  },
  {
    label: 'Advanced',
    params: [
      { key: 'mosaic', label: 'Mosaic', default: 1.0, min: 0, max: 1, step: 0.1, desc: 'Mosaic probability (4-image)' },
      { key: 'mixup', label: 'MixUp', default: 0, min: 0, max: 1, step: 0.1, desc: 'MixUp blend probability' },
      { key: 'copy_paste', label: 'Copy-Paste', default: 0, min: 0, max: 1, step: 0.1, desc: 'Copy-paste probability' },
      { key: 'erasing', label: 'Erasing', default: 0.4, min: 0, max: 0.9, step: 0.05, desc: 'Random erasing probability' },
    ],
  },
]

/** Toggle an augmentation param on/off, initializing to ultralytics default. */
function toggleAug(key: string, defaultVal: number) {
  augEnabled[key] = !augEnabled[key]
  if (augEnabled[key] && augValues[key] === undefined) {
    augValues[key] = defaultVal
  }
}

/** Base model presets. */
const modelPresets = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']

const canStart = computed(() =>
  datasetName.value && !props.disabled
)

/** Build the augmentation config from enabled fields. */
function buildAugmentation(): AugmentationConfig | undefined {
  const cfg: Record<string, unknown> = {}
  for (const group of augGroups) {
    for (const p of group.params) {
      if (augEnabled[p.key] && augValues[p.key] !== undefined) {
        cfg[p.key] = Number(augValues[p.key])
      }
    }
  }
  return Object.keys(cfg).length > 0 ? cfg as AugmentationConfig : undefined
}

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
      augmentation: buildAugmentation(),
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

    <div v-if="showAdvanced">
      <div class="config-row">
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

      <!-- Data Augmentation sub-section -->
      <button class="btn btn-sm aug-toggle" @click="showAugmentation = !showAugmentation">
        {{ showAugmentation ? 'Hide' : 'Show' }} Data Augmentation
      </button>

      <div v-if="showAugmentation" class="aug-section">
        <p class="aug-hint">Enable individual augmentation overrides. Disabled params use ultralytics defaults.</p>

        <div v-for="group in augGroups" :key="group.label" class="aug-group">
          <div class="aug-group-label">{{ group.label }}</div>
          <div class="aug-grid">
            <div v-for="p in group.params" :key="p.key" class="aug-param">
              <label class="aug-param-header">
                <input
                  type="checkbox"
                  :checked="augEnabled[p.key]"
                  :disabled="props.disabled"
                  @change="toggleAug(p.key, p.default)"
                />
                <span class="aug-param-label">{{ p.label }}</span>
                <span class="aug-param-default">({{ p.default }})</span>
              </label>
              <input
                v-if="augEnabled[p.key]"
                v-model.number="augValues[p.key]"
                type="number"
                class="form-input aug-input"
                :min="p.min"
                :max="p.max"
                :step="p.step"
                :disabled="props.disabled"
                :title="p.desc"
              />
              <span v-else class="aug-disabled-hint">default</span>
            </div>
          </div>
        </div>
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

.aug-toggle {
  margin-bottom: 0.5rem;
}

.aug-section {
  margin-bottom: 0.75rem;
  padding: 0.75rem;
  border: 1px solid var(--color-border, #333);
  border-radius: 6px;
  background: var(--color-bg-soft, #1a1a2e);
}

.aug-hint {
  font-size: 0.8rem;
  color: var(--color-text-muted, #888);
  margin: 0 0 0.75rem 0;
}

.aug-group {
  margin-bottom: 0.75rem;
}

.aug-group:last-child {
  margin-bottom: 0;
}

.aug-group-label {
  font-weight: 600;
  font-size: 0.85rem;
  margin-bottom: 0.4rem;
  color: var(--color-text-secondary, #aaa);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.aug-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 0.5rem;
}

.aug-param {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.aug-param-header {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.82rem;
  cursor: pointer;
}

.aug-param-label {
  font-weight: 500;
}

.aug-param-default {
  font-size: 0.75rem;
  color: var(--color-text-muted, #888);
}

.aug-input {
  padding: 0.25rem 0.4rem;
  font-size: 0.82rem;
}

.aug-disabled-hint {
  font-size: 0.75rem;
  color: var(--color-text-muted, #666);
  font-style: italic;
}

.start-btn {
  margin-top: 0.5rem;
  width: 100%;
}
</style>
