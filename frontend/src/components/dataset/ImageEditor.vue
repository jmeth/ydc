<script setup lang="ts">
/**
 * Modal image editor with annotation canvas and annotation list.
 *
 * Contains AnnotationCanvas + AnnotationList + toolbar (mode toggle,
 * class selector for new annotations, save, prev/next navigation).
 * Loads labels on open, saves via PUT on save button.
 *
 * Props:
 *   image - ImageInfo of the image to edit
 *   images - Full list of images for prev/next navigation
 *   classes - Available class names (index = class_id)
 *
 * Emits:
 *   close - When editor should be closed
 *   save - After annotations are saved
 */
import { ref, computed, watch } from 'vue'
import { useDatasetStore } from '@/stores/dataset'
import { useNotificationStore } from '@/stores/notifications'
import type { ImageInfo, Annotation } from '@/types/models'
import AnnotationCanvas from './AnnotationCanvas.vue'
import AnnotationList from './AnnotationList.vue'

const props = defineProps<{
  image: ImageInfo
  images: ImageInfo[]
  classes: string[]
}>()

const emit = defineEmits<{
  close: []
  save: []
}>()

const datasetStore = useDatasetStore()
const notificationStore = useNotificationStore()

/** Current image being edited. */
const currentImage = ref<ImageInfo>(props.image)
/** Annotations for the current image. */
const annotations = ref<Annotation[]>([])
/** Editor interaction mode. */
const mode = ref<'view' | 'draw' | 'edit'>('edit')
/** Class ID for new annotations. */
const activeClassId = ref(0)
/** Currently selected annotation index. */
const selectedIndex = ref(-1)
/** Whether we have unsaved changes. */
const dirty = ref(false)
/** Loading state. */
const loading = ref(false)

/** Build the image data URL. */
const imageSrc = computed(() =>
  `/api/datasets/${datasetStore.currentDataset}/images/${currentImage.value.split}/${currentImage.value.filename}/data`
)

/** Current index in the images array. */
const currentIdx = computed(() =>
  props.images.findIndex(
    (i) => i.filename === currentImage.value.filename && i.split === currentImage.value.split,
  )
)

const hasPrev = computed(() => currentIdx.value > 0)
const hasNext = computed(() => currentIdx.value < props.images.length - 1)

/** Load annotations for the current image. */
async function loadAnnotations() {
  loading.value = true
  try {
    const anns = await datasetStore.loadLabels(currentImage.value.split, currentImage.value.filename)
    annotations.value = [...anns]
    dirty.value = false
    selectedIndex.value = -1
  } catch {
    annotations.value = []
  } finally {
    loading.value = false
  }
}

/** Save current annotations to the backend. */
async function saveAnnotations() {
  try {
    await datasetStore.saveLabels(currentImage.value.split, currentImage.value.filename, annotations.value)
    dirty.value = false
    notificationStore.showToast('Annotations saved', 'success')
    emit('save')
  } catch {
    notificationStore.showToast('Failed to save annotations', 'error')
  }
}

/** Navigate to the previous image. */
async function prev() {
  if (!hasPrev.value) return
  if (dirty.value) await saveAnnotations()
  const img = props.images[currentIdx.value - 1]
  if (img) currentImage.value = img
}

/** Navigate to the next image. */
async function next() {
  if (!hasNext.value) return
  if (dirty.value) await saveAnnotations()
  const img = props.images[currentIdx.value + 1]
  if (img) currentImage.value = img
}

/** Handle annotation updates from the canvas. */
function onUpdateAnnotations(updated: Annotation[]) {
  annotations.value = updated
  dirty.value = true
}

/** Handle annotation delete from the list. */
function onDeleteAnnotation(index: number) {
  annotations.value = annotations.value.filter((_, i) => i !== index)
  selectedIndex.value = -1
  dirty.value = true
}

/** Handle keyboard shortcuts. */
function onKeyDown(event: KeyboardEvent) {
  if (event.key === 'ArrowLeft') prev()
  else if (event.key === 'ArrowRight') next()
  else if (event.key === 's' && (event.ctrlKey || event.metaKey)) {
    event.preventDefault()
    saveAnnotations()
  }
}

// Load annotations when image changes
watch(currentImage, () => {
  loadAnnotations()
}, { immediate: true })
</script>

<template>
  <Teleport to="body">
    <div class="editor-overlay" @keydown="onKeyDown" tabindex="0">
      <div class="editor-modal">
        <!-- Toolbar -->
        <div class="editor-toolbar">
          <div class="toolbar-left">
            <span class="editor-filename">{{ currentImage.filename }}</span>
            <span class="badge badge-info">{{ currentImage.split }}</span>
            <span v-if="dirty" class="badge badge-warning">unsaved</span>
          </div>

          <div class="toolbar-center">
            <button
              class="btn btn-sm"
              :class="{ 'btn-primary': mode === 'edit' }"
              @click="mode = 'edit'"
            >
              Edit
            </button>
            <button
              class="btn btn-sm"
              :class="{ 'btn-primary': mode === 'draw' }"
              @click="mode = 'draw'"
            >
              Draw
            </button>
            <button
              class="btn btn-sm"
              :class="{ 'btn-primary': mode === 'view' }"
              @click="mode = 'view'"
            >
              View
            </button>

            <select v-model.number="activeClassId" class="form-select btn-sm class-select">
              <option
                v-for="(cls, idx) in props.classes"
                :key="idx"
                :value="idx"
              >
                {{ idx }}: {{ cls }}
              </option>
            </select>
          </div>

          <div class="toolbar-right">
            <button class="btn btn-sm" :disabled="!hasPrev" @click="prev">&larr; Prev</button>
            <button class="btn btn-sm" :disabled="!hasNext" @click="next">Next &rarr;</button>
            <button class="btn btn-sm btn-success" @click="saveAnnotations">Save</button>
            <button class="btn btn-sm" @click="emit('close')">Close</button>
          </div>
        </div>

        <!-- Main content -->
        <div class="editor-content">
          <div class="editor-canvas-area">
            <div v-if="loading" class="empty-state">
              <div class="loading-spinner"></div>
            </div>
            <AnnotationCanvas
              v-else
              :image-src="imageSrc"
              :annotations="annotations"
              :mode="mode"
              :active-class-id="activeClassId"
              :selected-index="selectedIndex"
              @update:annotations="onUpdateAnnotations"
              @select="selectedIndex = $event"
            />
          </div>

          <div class="editor-sidebar">
            <AnnotationList
              :annotations="annotations"
              :classes="props.classes"
              :selected-index="selectedIndex"
              @select="selectedIndex = $event"
              @update:annotations="onUpdateAnnotations"
              @delete="onDeleteAnnotation"
            />
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.editor-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.85);
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  outline: none;
}

.editor-modal {
  width: 95vw;
  height: 90vh;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.editor-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0.75rem;
  background: var(--color-bg-header);
  border-bottom: 1px solid var(--color-border);
  gap: 0.75rem;
}

.toolbar-left {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.toolbar-center {
  display: flex;
  align-items: center;
  gap: 0.375rem;
}

.toolbar-right {
  display: flex;
  align-items: center;
  gap: 0.375rem;
}

.editor-filename {
  font-size: 0.8125rem;
  font-weight: 600;
}

.class-select {
  width: 120px;
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
}

.editor-content {
  display: grid;
  grid-template-columns: 1fr 240px;
  flex: 1;
  overflow: hidden;
}

.editor-canvas-area {
  position: relative;
  overflow: hidden;
}

.editor-sidebar {
  border-left: 1px solid var(--color-border);
  padding: 0.5rem;
  overflow-y: auto;
}
</style>
