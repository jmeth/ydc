<script setup lang="ts">
/**
 * Manage class labels for the current dataset.
 *
 * Displays each class with its index and a delete button.
 * Provides an input to add new classes. Changes are saved
 * immediately via PUT /api/datasets/{name}.
 *
 * Props:
 *   classes - Current ordered list of class names
 */
import { ref } from 'vue'
import { useDatasetStore } from '@/stores/dataset'
import { useNotificationStore } from '@/stores/notifications'

const props = defineProps<{
  classes: string[]
}>()

const datasetStore = useDatasetStore()
const notificationStore = useNotificationStore()

/** Input value for new class name. */
const newClassName = ref('')
/** Loading state during save. */
const saving = ref(false)

/**
 * Add a new class label to the dataset.
 * Validates that the name is non-empty and not a duplicate.
 */
async function addClass() {
  const name = newClassName.value.trim()
  if (!name) return
  if (props.classes.includes(name)) {
    notificationStore.showToast(`Class "${name}" already exists`, 'error')
    return
  }

  saving.value = true
  try {
    await datasetStore.updateClasses([...props.classes, name])
    newClassName.value = ''
    notificationStore.showToast(`Added class "${name}"`, 'success')
  } catch {
    notificationStore.showToast('Failed to add class', 'error')
  } finally {
    saving.value = false
  }
}

/**
 * Remove a class label from the dataset.
 * Warns that existing annotations referencing this class may become invalid.
 *
 * @param index - Index of the class to remove
 */
async function removeClass(index: number) {
  const name = props.classes[index]
  if (!confirm(`Remove class "${name}"? Annotations using this class will be deleted and remaining class IDs will be remapped.`)) {
    return
  }

  saving.value = true
  try {
    const updated = props.classes.filter((_, i) => i !== index)
    await datasetStore.updateClasses(updated)
    notificationStore.showToast(`Removed class "${name}"`, 'success')
  } catch {
    notificationStore.showToast('Failed to remove class', 'error')
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="class-manager card">
    <div class="card-header">Class Labels</div>

    <div v-if="props.classes.length === 0" class="empty-state">
      <div class="empty-state-text">No classes defined</div>
    </div>

    <div
      v-for="(cls, idx) in props.classes"
      :key="idx"
      class="class-row"
    >
      <span class="class-index">{{ idx }}</span>
      <span class="class-name">{{ cls }}</span>
      <button
        class="btn btn-sm btn-danger class-delete"
        :disabled="saving"
        @click="removeClass(idx)"
        :title="`Remove class '${cls}'`"
      >
        &times;
      </button>
    </div>

    <div class="add-class-row">
      <input
        v-model="newClassName"
        class="form-input add-class-input"
        type="text"
        placeholder="New class name..."
        :disabled="saving"
        @keyup.enter="addClass"
      />
      <button
        class="btn btn-sm btn-primary"
        :disabled="saving || !newClassName.trim()"
        @click="addClass"
      >
        Add
      </button>
    </div>
  </div>
</template>

<style scoped>
.class-manager {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.class-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.375rem 0;
  border-bottom: 1px solid var(--color-border);
  font-size: 0.8125rem;
}

.class-row:last-of-type {
  border-bottom: none;
}

.class-index {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--color-text-muted);
  min-width: 1.5rem;
  text-align: right;
}

.class-name {
  flex: 1;
  font-weight: 500;
}

.class-delete {
  padding: 0.125rem 0.375rem;
  font-size: 0.75rem;
  line-height: 1;
}

.add-class-row {
  display: flex;
  gap: 0.375rem;
  margin-top: 0.5rem;
}

.add-class-input {
  flex: 1;
  min-width: 0;
  font-size: 0.8125rem;
  padding: 0.25rem 0.5rem;
}
</style>
