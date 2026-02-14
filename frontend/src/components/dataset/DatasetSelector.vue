<script setup lang="ts">
/**
 * Dataset selector with create, delete, import, and export actions.
 *
 * Dropdown of datasets with a create modal (name + classes form),
 * delete confirmation, and import/export buttons.
 *
 * Emits:
 *   select - When a dataset is chosen
 */
import { ref, computed } from 'vue'
import { useDatasetStore } from '@/stores/dataset'
import { useNotificationStore } from '@/stores/notifications'
import Modal from '@/components/common/Modal.vue'

const emit = defineEmits<{
  select: [name: string]
}>()

const datasetStore = useDatasetStore()
const notificationStore = useNotificationStore()

/** Create modal state. */
const showCreateModal = ref(false)
const newName = ref('')
const newClasses = ref('')

/** Delete confirmation state. */
const showDeleteConfirm = ref(false)
const deleteTarget = ref('')

const selectedDataset = computed(() => datasetStore.currentDataset)

/**
 * Handle dataset selection from the dropdown.
 *
 * @param event - Native change event
 */
function onSelect(event: Event) {
  const value = (event.target as HTMLSelectElement).value
  if (value) {
    emit('select', value)
  }
}

/** Open create dataset modal. */
function openCreate() {
  newName.value = ''
  newClasses.value = ''
  showCreateModal.value = true
}

/** Submit create dataset form. */
async function createDataset() {
  const classes = newClasses.value.split(',').map((c) => c.trim()).filter(Boolean)
  if (!newName.value || classes.length === 0) return
  try {
    await datasetStore.createDataset(newName.value, classes)
    showCreateModal.value = false
    emit('select', newName.value)
    notificationStore.showToast(`Created dataset "${newName.value}"`, 'success')
  } catch (err) {
    notificationStore.showToast('Failed to create dataset', 'error')
  }
}

/**
 * Prompt for dataset deletion.
 *
 * @param name - Dataset name to delete
 */
function confirmDelete(name: string) {
  deleteTarget.value = name
  showDeleteConfirm.value = true
}

/** Execute dataset deletion. */
async function deleteDataset() {
  try {
    await datasetStore.deleteDataset(deleteTarget.value)
    showDeleteConfirm.value = false
    notificationStore.showToast(`Deleted dataset "${deleteTarget.value}"`, 'success')
  } catch (err) {
    notificationStore.showToast('Failed to delete dataset', 'error')
  }
}

/** Trigger dataset export (zip download). */
function exportDataset() {
  datasetStore.exportDataset()
}

/** Handle zip file import. */
async function importDataset(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  try {
    const { upload } = await import('@/composables/useApi').then((m) => m.useApi())
    await upload('/datasets/import', file)
    await datasetStore.fetchDatasets()
    notificationStore.showToast('Dataset imported', 'success')
  } catch (err) {
    notificationStore.showToast('Failed to import dataset', 'error')
  }
  input.value = ''
}
</script>

<template>
  <div class="dataset-selector">
    <div class="selector-row">
      <div class="form-group" style="flex: 1">
        <label class="form-label">Dataset</label>
        <select
          class="form-select"
          :value="selectedDataset || ''"
          @change="onSelect"
        >
          <option value="">Select a dataset...</option>
          <option
            v-for="ds in datasetStore.datasets"
            :key="ds.name"
            :value="ds.name"
          >
            {{ ds.name }} ({{ Object.values(ds.num_images).reduce((a, b) => a + b, 0) }} images)
          </option>
        </select>
      </div>

      <div class="selector-actions">
        <button class="btn btn-sm btn-primary" @click="openCreate">New</button>
        <button
          class="btn btn-sm btn-danger"
          :disabled="!selectedDataset"
          @click="confirmDelete(selectedDataset!)"
        >
          Delete
        </button>
        <button
          class="btn btn-sm"
          :disabled="!selectedDataset"
          @click="exportDataset"
        >
          Export
        </button>
        <label class="btn btn-sm">
          Import
          <input type="file" accept=".zip" hidden @change="importDataset" />
        </label>
      </div>
    </div>
  </div>

  <!-- Create dataset modal -->
  <Modal v-if="showCreateModal" title="Create Dataset" @close="showCreateModal = false">
    <div class="modal-form">
      <div class="form-group">
        <label class="form-label">Name</label>
        <input
          v-model="newName"
          class="form-input"
          placeholder="my-dataset"
          @keyup.enter="createDataset"
        />
      </div>
      <div class="form-group">
        <label class="form-label">Classes (comma-separated)</label>
        <input
          v-model="newClasses"
          class="form-input"
          placeholder="cat, dog, bird"
          @keyup.enter="createDataset"
        />
      </div>
      <div class="modal-actions">
        <button class="btn" @click="showCreateModal = false">Cancel</button>
        <button class="btn btn-primary" @click="createDataset">Create</button>
      </div>
    </div>
  </Modal>

  <!-- Delete confirmation modal -->
  <Modal v-if="showDeleteConfirm" title="Delete Dataset" @close="showDeleteConfirm = false">
    <p>Are you sure you want to delete <strong>{{ deleteTarget }}</strong>? This cannot be undone.</p>
    <div class="modal-actions">
      <button class="btn" @click="showDeleteConfirm = false">Cancel</button>
      <button class="btn btn-danger" @click="deleteDataset">Delete</button>
    </div>
  </Modal>
</template>

<style scoped>
.selector-row {
  display: flex;
  align-items: flex-end;
  gap: 0.75rem;
}

.selector-actions {
  display: flex;
  gap: 0.375rem;
}

.modal-form {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
  margin-top: 1rem;
}
</style>
