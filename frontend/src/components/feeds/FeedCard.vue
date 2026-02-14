<script setup lang="ts">
/**
 * Card displaying a single feed's metadata and actions.
 *
 * Shows feed name, type badge, source, status indicator,
 * fps, resolution, and frame count. Provides pause/resume
 * and delete action buttons. Clicking the card toggles a
 * snapshot preview from the feed.
 *
 * Props:
 *   feed - FeedInfo object to display
 *
 * Emits:
 *   pause  - Feed should be paused
 *   resume - Feed should be resumed
 *   delete - Feed should be deleted (after confirmation)
 */
import { ref } from 'vue'
import type { FeedInfo } from '@/types/models'

const props = defineProps<{
  feed: FeedInfo
}>()

const emit = defineEmits<{
  pause: [feedId: string]
  resume: [feedId: string]
  delete: [feedId: string]
}>()

/** Whether the snapshot preview is visible. */
const showSnapshot = ref(false)
/** Whether a delete confirmation is pending. */
const confirmingDelete = ref(false)
/** Timestamp appended to snapshot URL to bust cache. */
const snapshotTs = ref(Date.now())

/**
 * Toggle the snapshot preview and refresh the image timestamp.
 */
function toggleSnapshot() {
  if (!showSnapshot.value) {
    snapshotTs.value = Date.now()
  }
  showSnapshot.value = !showSnapshot.value
}

/**
 * Begin delete confirmation flow.
 */
function requestDelete() {
  confirmingDelete.value = true
}

/**
 * Confirm and emit the delete event.
 */
function confirmDelete() {
  confirmingDelete.value = false
  emit('delete', props.feed.id)
}

/**
 * Cancel the delete confirmation.
 */
function cancelDelete() {
  confirmingDelete.value = false
}

/**
 * Format resolution as "WxH" or "N/A".
 *
 * @param res - [width, height] tuple or null
 * @returns Formatted string
 */
function formatResolution(res: [number, number] | null): string {
  if (!res) return 'N/A'
  return `${res[0]}x${res[1]}`
}
</script>

<template>
  <div class="card feed-card" @click="toggleSnapshot">
    <!-- Header row: name, type badge, status dot -->
    <div class="feed-card-header">
      <span class="feed-name">{{ feed.name }}</span>
      <span
        class="badge"
        :class="{
          'badge-info': feed.feedType === 'camera',
          'badge-success': feed.feedType === 'rtsp',
          'badge-warning': feed.feedType === 'file',
          'badge-error': feed.feedType === 'inference',
        }"
      >
        {{ feed.feedType }}
      </span>
      <span
        class="status-dot"
        :class="{
          'status-active': feed.status === 'active',
          'status-paused': feed.status === 'paused',
          'status-error': feed.status === 'error' || feed.status === 'stopped',
        }"
        :title="feed.status"
      />
    </div>

    <!-- Feed details -->
    <div class="feed-card-details">
      <div class="detail-row">
        <span class="detail-label">Source</span>
        <span class="detail-value">{{ feed.source }}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Status</span>
        <span class="detail-value">{{ feed.status }}</span>
      </div>
      <div class="stat-row">
        <span class="stat-item">
          <span class="stat-label">FPS</span>
          <span class="stat-value">{{ feed.fps.toFixed(1) }}</span>
        </span>
        <span class="stat-item">
          <span class="stat-label">Res</span>
          <span class="stat-value">{{ formatResolution(feed.resolution) }}</span>
        </span>
        <span class="stat-item">
          <span class="stat-label">Frames</span>
          <span class="stat-value">{{ feed.frameCount }}</span>
        </span>
      </div>
    </div>

    <!-- Snapshot preview (toggles on card click) -->
    <div v-if="showSnapshot" class="feed-snapshot" @click.stop>
      <img
        :src="`/api/feeds/${feed.id}/snapshot?t=${snapshotTs}`"
        :alt="`Snapshot from ${feed.name}`"
        class="snapshot-img"
        @error="showSnapshot = false"
      />
    </div>

    <!-- Action buttons -->
    <div class="feed-card-actions" @click.stop>
      <button
        v-if="feed.status === 'active'"
        class="btn btn-sm"
        title="Pause feed"
        @click="emit('pause', feed.id)"
      >
        Pause
      </button>
      <button
        v-else-if="feed.status === 'paused'"
        class="btn btn-sm btn-success"
        title="Resume feed"
        @click="emit('resume', feed.id)"
      >
        Resume
      </button>

      <!-- Delete with confirmation -->
      <template v-if="!confirmingDelete">
        <button
          class="btn btn-sm btn-danger"
          title="Delete feed"
          @click="requestDelete"
        >
          Delete
        </button>
      </template>
      <template v-else>
        <span class="confirm-text">Delete?</span>
        <button class="btn btn-sm btn-danger" @click="confirmDelete">Yes</button>
        <button class="btn btn-sm" @click="cancelDelete">No</button>
      </template>
    </div>
  </div>
</template>

<style scoped>
.feed-card {
  cursor: pointer;
  transition: border-color 0.15s;
}

.feed-card:hover {
  border-color: var(--color-accent);
}

.feed-card-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.feed-name {
  font-weight: 600;
  font-size: 0.9375rem;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  background: var(--color-text-muted);
}

.status-active {
  background: var(--color-success);
}

.status-paused {
  background: var(--color-warning);
}

.status-error {
  background: var(--color-error);
}

.feed-card-details {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
  margin-bottom: 0.75rem;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  font-size: 0.8125rem;
}

.detail-label {
  color: var(--color-text-muted);
}

.detail-value {
  font-family: var(--font-mono);
  font-size: 0.8125rem;
  max-width: 60%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: right;
}

.feed-snapshot {
  margin-bottom: 0.75rem;
  border-radius: var(--radius-md);
  overflow: hidden;
  background: var(--color-bg);
}

.snapshot-img {
  width: 100%;
  display: block;
}

.feed-card-actions {
  display: flex;
  align-items: center;
  gap: 0.375rem;
}

.confirm-text {
  font-size: 0.75rem;
  color: var(--color-error);
  font-weight: 500;
}
</style>
