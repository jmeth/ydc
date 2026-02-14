<script setup lang="ts">
/**
 * Dataset mode view — image management and annotation.
 *
 * DatasetSelector at top for choosing/creating/deleting datasets.
 * ImageGrid as main content with filter bar and thumbnails.
 * BulkActions toolbar for multi-select operations.
 * Double-click an image to open the ImageEditor (added in Step 7).
 *
 * On mount: fetches all datasets.
 */
import { ref, onMounted, watch } from 'vue'
import { useDatasetStore } from '@/stores/dataset'
import type { ImageInfo } from '@/types/models'
import DatasetSelector from '@/components/dataset/DatasetSelector.vue'
import ImageGrid from '@/components/dataset/ImageGrid.vue'
import BulkActions from '@/components/dataset/BulkActions.vue'
import ImageEditor from '@/components/dataset/ImageEditor.vue'

const datasetStore = useDatasetStore()

/** Track selected images for bulk actions. */
const selectedImages = ref<ImageInfo[]>([])
/** Whether the image editor modal is open. */
const editorImage = ref<ImageInfo | null>(null)

/**
 * Handle dataset selection from the selector.
 *
 * @param name - Dataset name to load
 */
async function onSelectDataset(name: string) {
  selectedImages.value = []
  await datasetStore.loadDataset(name)
}

/**
 * Handle image click — toggle selection.
 *
 * @param image - Clicked image
 */
function onSelectImage(image: ImageInfo) {
  datasetStore.selectedImage = image
  const idx = selectedImages.value.findIndex(
    (i) => i.filename === image.filename && i.split === image.split,
  )
  if (idx >= 0) {
    selectedImages.value.splice(idx, 1)
  } else {
    selectedImages.value.push(image)
  }
}

/**
 * Handle image double-click — open editor.
 *
 * @param image - Image to edit
 */
function onOpenImage(image: ImageInfo) {
  editorImage.value = image
}

/** Clear selection after bulk action. */
function clearSelection() {
  selectedImages.value = []
}

/** Reload images when the filter split changes. */
watch(
  () => datasetStore.filter.split,
  async () => {
    if (datasetStore.currentDataset) {
      await datasetStore.loadImages()
    }
  },
)

onMounted(async () => {
  await datasetStore.fetchDatasets()
})
</script>

<template>
  <div class="dataset-view">
    <DatasetSelector @select="onSelectDataset" />

    <template v-if="datasetStore.currentDataset">
      <!-- Dataset info -->
      <div v-if="datasetStore.currentDatasetInfo" class="dataset-info">
        <span class="badge badge-info">{{ datasetStore.currentDatasetInfo.classes.length }} classes</span>
        <span
          v-for="(count, split) in datasetStore.currentDatasetInfo.num_images"
          :key="split"
          class="badge badge-success"
        >
          {{ split }}: {{ count }}
        </span>
      </div>

      <BulkActions
        :selected-images="selectedImages"
        @clear="clearSelection"
      />

      <div v-if="datasetStore.loading" class="empty-state">
        <div class="loading-spinner"></div>
        <div class="empty-state-text">Loading dataset...</div>
      </div>

      <ImageGrid
        v-else
        @select="onSelectImage"
        @open="onOpenImage"
      />
    </template>

    <div v-else class="empty-state">
      <div class="empty-state-title">No dataset selected</div>
      <div class="empty-state-text">Select or create a dataset to get started</div>
    </div>

    <!-- ImageEditor modal — opens on double-click from the grid -->
    <ImageEditor
      v-if="editorImage && datasetStore.currentDatasetInfo"
      :image="editorImage"
      :images="datasetStore.filteredImages"
      :classes="datasetStore.currentDatasetInfo.classes"
      @close="editorImage = null"
      @save="datasetStore.loadImages()"
    />
  </div>
</template>

<style scoped>
.dataset-view {
  max-width: 1400px;
  margin: 0 auto;
}

.dataset-info {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.75rem;
}
</style>
