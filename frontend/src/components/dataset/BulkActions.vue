<script setup lang="ts">
/**
 * Toolbar for bulk operations on selected images.
 *
 * Shows when images are selected. Provides change split dropdown,
 * delete with confirmation, and upload button.
 *
 * Props:
 *   selectedImages - Array of selected ImageInfo objects
 *
 * Emits:
 *   clear - When selection should be cleared
 */
import { ref } from 'vue'
import { useDatasetStore } from '@/stores/dataset'
import { useNotificationStore } from '@/stores/notifications'
import type { ImageInfo } from '@/types/models'

const props = defineProps<{
  selectedImages: ImageInfo[]
}>()

const emit = defineEmits<{
  clear: []
}>()

const datasetStore = useDatasetStore()
const notificationStore = useNotificationStore()

const showDeleteConfirm = ref(false)
const targetSplit = ref('train')

/**
 * Move all selected images to a different split.
 */
async function changeSplit() {
  try {
    for (const img of props.selectedImages) {
      if (img.split !== targetSplit.value) {
        await datasetStore.changeSplit(img.filename, img.split, targetSplit.value)
      }
    }
    notificationStore.showToast(`Moved ${props.selectedImages.length} images to ${targetSplit.value}`, 'success')
    emit('clear')
  } catch (err) {
    notificationStore.showToast('Failed to move images', 'error')
  }
}

/** Delete all selected images after confirmation. */
async function deleteSelected() {
  try {
    for (const img of props.selectedImages) {
      await datasetStore.deleteImage(img.split, img.filename)
    }
    notificationStore.showToast(`Deleted ${props.selectedImages.length} images`, 'success')
    showDeleteConfirm.value = false
    emit('clear')
  } catch (err) {
    notificationStore.showToast('Failed to delete images', 'error')
  }
}

/** Handle file upload from the file input. */
async function handleUpload(event: Event) {
  const input = event.target as HTMLInputElement
  const files = input.files
  if (!files) return
  try {
    for (const file of files) {
      await datasetStore.uploadImage(file, targetSplit.value)
    }
    notificationStore.showToast(`Uploaded ${files.length} images`, 'success')
  } catch (err) {
    notificationStore.showToast('Failed to upload images', 'error')
  }
  input.value = ''
}
</script>

<template>
  <div class="bulk-actions">
    <div class="bulk-left">
      <span v-if="props.selectedImages.length > 0" class="bulk-count">
        {{ props.selectedImages.length }} selected
      </span>

      <!-- Move to split -->
      <template v-if="props.selectedImages.length > 0">
        <select v-model="targetSplit" class="form-select btn-sm">
          <option value="train">Train</option>
          <option value="val">Val</option>
          <option value="test">Test</option>
        </select>
        <button class="btn btn-sm" @click="changeSplit">Move</button>
        <button class="btn btn-sm btn-danger" @click="showDeleteConfirm = true">Delete</button>
        <button class="btn btn-sm" @click="emit('clear')">Clear</button>
      </template>
    </div>

    <div class="bulk-right">
      <label class="btn btn-sm btn-primary">
        Upload Images
        <input type="file" accept="image/*" multiple hidden @change="handleUpload" />
      </label>
    </div>

    <!-- Delete confirmation -->
    <div v-if="showDeleteConfirm" class="delete-confirm">
      <span>Delete {{ props.selectedImages.length }} images? This cannot be undone.</span>
      <button class="btn btn-sm" @click="showDeleteConfirm = false">Cancel</button>
      <button class="btn btn-sm btn-danger" @click="deleteSelected">Confirm</button>
    </div>
  </div>
</template>

<style scoped>
.bulk-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0;
  min-height: 2.5rem;
  position: relative;
}

.bulk-left {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.bulk-right {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.bulk-count {
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--color-accent);
}

.delete-confirm {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem;
  background: var(--color-bg-surface);
  border: 1px solid var(--color-error);
  border-radius: var(--radius-md);
  font-size: 0.8125rem;
  z-index: 10;
}
</style>
