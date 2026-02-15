<script setup lang="ts">
/**
 * Table/card list of trained models.
 *
 * Displays model name, base model, dataset, date, epochs, mAP@50,
 * active badge. Provides activate/delete/export actions per model
 * and an import button in the header.
 *
 * Emits:
 *   select - When a model row is clicked (model name)
 */
import { ref } from 'vue'
import { useTrainingStore } from '@/stores/training'
import { useNotificationStore } from '@/stores/notifications'

const props = defineProps<{
  selectedModel?: string | null
}>()

const emit = defineEmits<{
  select: [modelName: string]
}>()

const trainingStore = useTrainingStore()
const notificationStore = useNotificationStore()

/** Hidden file input ref for import. */
const importInput = ref<HTMLInputElement | null>(null)

/** Format a unix timestamp. */
function formatDate(ts: number): string {
  if (!ts) return '-'
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/**
 * Activate a model for inference and select it.
 *
 * @param name - Model name to activate
 */
async function activateModel(name: string) {
  try {
    await trainingStore.activateModel(name)
    emit('select', name)
    notificationStore.showToast(`Activated model "${name}"`, 'success')
  } catch {
    notificationStore.showToast('Failed to activate model', 'error')
  }
}

/**
 * Delete a trained model.
 *
 * @param name - Model name to delete
 */
async function deleteModel(name: string) {
  if (!confirm(`Delete model "${name}"?`)) return
  try {
    await trainingStore.deleteModel(name)
    notificationStore.showToast(`Deleted model "${name}"`, 'success')
  } catch {
    notificationStore.showToast('Failed to delete model', 'error')
  }
}

/**
 * Export a model as a zip download.
 *
 * @param name - Model name to export
 */
function exportModel(name: string) {
  trainingStore.exportModel(name)
}

/**
 * Handle file input change for model import.
 * Reads the selected zip file and uploads it.
 */
async function handleImportFile(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  try {
    await trainingStore.importModel(file)
    notificationStore.showToast('Model imported successfully', 'success')
  } catch {
    notificationStore.showToast('Failed to import model', 'error')
  }
  // Reset input so the same file can be re-selected
  input.value = ''
}
</script>

<template>
  <div class="model-list card">
    <div class="card-header">
      <span>Trained Models</span>
      <button class="btn btn-sm btn-secondary" @click="importInput?.click()">
        Import
      </button>
      <input
        ref="importInput"
        type="file"
        accept=".zip"
        style="display: none"
        @change="handleImportFile"
      />
    </div>

    <div v-if="trainingStore.models.length === 0" class="empty-state">
      <div class="empty-state-text">No trained models yet</div>
    </div>

    <div
      v-for="model in trainingStore.models"
      :key="model.name"
      class="model-row"
      :class="{ 'model-row-selected': props.selectedModel === model.name }"
      @click="emit('select', model.name)"
    >
      <div class="model-info">
        <div class="model-name">
          {{ model.name }}
          <span v-if="model.isActive" class="badge badge-success">active</span>
        </div>
        <div class="model-meta">
          {{ model.baseModel }} &middot; {{ model.datasetName }} &middot;
          {{ model.epochsCompleted }} epochs &middot;
          <template v-if="model.bestMap50 !== null">
            mAP: {{ (model.bestMap50 * 100).toFixed(1) }}%
          </template>
          <template v-else>-</template>
        </div>
        <div class="model-date">{{ formatDate(model.createdAt) }}</div>
      </div>

      <div class="model-actions" @click.stop>
        <button
          v-if="!model.isActive"
          class="btn btn-sm btn-primary"
          @click="activateModel(model.name)"
        >
          Activate
        </button>
        <button
          class="btn btn-sm btn-secondary"
          @click="exportModel(model.name)"
        >
          Export
        </button>
        <button
          class="btn btn-sm btn-danger"
          @click="deleteModel(model.name)"
        >
          Delete
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.model-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  padding: 0.625rem 0;
  border-bottom: 1px solid var(--color-border);
  cursor: pointer;
  gap: 0.375rem;
}

.model-row:last-child {
  border-bottom: none;
}

.model-row:hover {
  background: var(--color-bg);
  margin: 0 -1rem;
  padding-left: 1rem;
  padding-right: 1rem;
}

.model-row-selected {
  background: var(--color-bg);
  margin: 0 -1rem;
  padding-left: 1rem;
  padding-right: 1rem;
  border-left: 3px solid var(--color-primary, #4fc3f7);
}

.model-name {
  font-weight: 600;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.375rem;
}

.model-meta {
  font-size: 0.75rem;
  color: var(--color-text-muted);
  margin-top: 0.125rem;
}

.model-date {
  font-size: 0.6875rem;
  color: var(--color-text-muted);
}

.model-info {
  min-width: 0;
  flex: 1 1 100%;
}

.model-actions {
  display: flex;
  gap: 0.375rem;
  flex-shrink: 0;
  margin-left: auto;
}
</style>
