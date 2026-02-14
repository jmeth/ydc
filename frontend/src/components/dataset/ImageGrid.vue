<script setup lang="ts">
/**
 * Grid display of dataset images with thumbnails.
 *
 * Shows images from the dataset store as a CSS grid of thumbnail
 * cards. Each card shows the image via the /data endpoint, a label
 * indicator badge, and split badge. Click selects, double-click
 * opens the editor. Includes filter bar for split and search.
 *
 * Emits:
 *   select - Single click on an image
 *   open - Double click to open annotation editor
 */
import { computed } from 'vue'
import { useDatasetStore } from '@/stores/dataset'
import type { ImageInfo } from '@/types/models'

const emit = defineEmits<{
  select: [image: ImageInfo]
  open: [image: ImageInfo]
}>()

const datasetStore = useDatasetStore()

/** Current images filtered by store state. */
const images = computed(() => datasetStore.filteredImages)

/**
 * Build the image data URL for a thumbnail.
 *
 * @param image - Image metadata
 * @returns URL to the image data endpoint
 */
function imageUrl(image: ImageInfo): string {
  return `/api/datasets/${datasetStore.currentDataset}/images/${image.split}/${image.filename}/data`
}

/**
 * Determine if an image is currently selected.
 *
 * @param image - Image to check
 * @returns True if the image is selected
 */
function isSelected(image: ImageInfo): boolean {
  return datasetStore.selectedImage?.filename === image.filename &&
    datasetStore.selectedImage?.split === image.split
}
</script>

<template>
  <div class="image-grid-container">
    <!-- Filter bar -->
    <div class="filter-bar">
      <div class="form-group">
        <label class="form-label">Split</label>
        <select v-model="datasetStore.filter.split" class="form-select filter-select">
          <option value="all">All</option>
          <option value="train">Train</option>
          <option value="val">Val</option>
          <option value="test">Test</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Labels</label>
        <select v-model="datasetStore.filter.annotated" class="form-select filter-select">
          <option value="all">All</option>
          <option value="yes">Labeled</option>
          <option value="no">Unlabeled</option>
        </select>
      </div>
      <div class="form-group" style="flex: 1">
        <label class="form-label">Search</label>
        <input
          v-model="datasetStore.filter.search"
          class="form-input"
          placeholder="Filter by filename..."
        />
      </div>
      <div class="filter-count">
        {{ images.length }} images
      </div>
    </div>

    <!-- Image grid -->
    <div v-if="images.length > 0" class="image-grid">
      <div
        v-for="image in images"
        :key="`${image.split}/${image.filename}`"
        class="image-card"
        :class="{ selected: isSelected(image) }"
        @click="emit('select', image)"
        @dblclick="emit('open', image)"
      >
        <div class="image-thumb">
          <img
            :src="imageUrl(image)"
            :alt="image.filename"
            loading="lazy"
          />
        </div>
        <div class="image-info">
          <span class="image-name" :title="image.filename">{{ image.filename }}</span>
          <div class="image-badges">
            <span class="badge badge-info">{{ image.split }}</span>
            <span v-if="image.hasLabels" class="badge badge-success">labeled</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Empty state -->
    <div v-else class="empty-state">
      <div class="empty-state-title">No images</div>
      <div class="empty-state-text">
        Upload images or start capturing from a feed
      </div>
    </div>
  </div>
</template>

<style scoped>
.image-grid-container {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.filter-bar {
  display: flex;
  align-items: flex-end;
  gap: 0.75rem;
}

.filter-select {
  width: 100px;
}

.filter-count {
  font-size: 0.8125rem;
  color: var(--color-text-muted);
  white-space: nowrap;
  padding-bottom: 0.5rem;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 0.75rem;
}

.image-card {
  background: var(--color-bg-surface);
  border: 2px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
  cursor: pointer;
  transition: border-color 0.15s;
}

.image-card:hover {
  border-color: var(--color-text-muted);
}

.image-card.selected {
  border-color: var(--color-accent);
}

.image-thumb {
  aspect-ratio: 1;
  background: #000;
  overflow: hidden;
}

.image-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.image-info {
  padding: 0.375rem 0.5rem;
}

.image-name {
  font-size: 0.6875rem;
  color: var(--color-text-muted);
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.image-badges {
  display: flex;
  gap: 0.25rem;
  margin-top: 0.25rem;
}
</style>
