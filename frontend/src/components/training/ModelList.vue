<script setup lang="ts">
/**
 * Table/card list of trained models.
 *
 * Displays model name, base model, dataset, date, epochs, mAP@50,
 * active badge. Provides activate/delete actions per model.
 *
 * Emits:
 *   select - When a model row is clicked (model name)
 */
import { useTrainingStore } from '@/stores/training'
import { useNotificationStore } from '@/stores/notifications'

const emit = defineEmits<{
  select: [modelName: string]
}>()

const trainingStore = useTrainingStore()
const notificationStore = useNotificationStore()

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
 * Activate a model for inference.
 *
 * @param name - Model name to activate
 */
async function activateModel(name: string) {
  try {
    await trainingStore.activateModel(name)
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
</script>

<template>
  <div class="model-list card">
    <div class="card-header">Trained Models</div>

    <div v-if="trainingStore.models.length === 0" class="empty-state">
      <div class="empty-state-text">No trained models yet</div>
    </div>

    <div
      v-for="model in trainingStore.models"
      :key="model.name"
      class="model-row"
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
.model-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.625rem 0;
  border-bottom: 1px solid var(--color-border);
  cursor: pointer;
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

.model-actions {
  display: flex;
  gap: 0.375rem;
  flex-shrink: 0;
}
</style>
