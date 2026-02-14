<script setup lang="ts">
/**
 * Feeds management view.
 *
 * Displays the AddFeedForm at the top and a grid of FeedCard
 * components for each registered feed. Handles add, pause, resume,
 * and delete actions through the app store.
 *
 * On mount: fetches the current feed list from the backend.
 */
import { ref, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import { useNotificationStore } from '@/stores/notifications'
import AddFeedForm from '@/components/feeds/AddFeedForm.vue'
import FeedCard from '@/components/feeds/FeedCard.vue'

const appStore = useAppStore()
const notificationStore = useNotificationStore()

/** Ref to the AddFeedForm for calling reset/setError. */
const addFormRef = ref<InstanceType<typeof AddFeedForm> | null>(null)

/**
 * Handle new feed submission from AddFeedForm.
 *
 * @param data - Feed creation params from the form
 */
async function onAddFeed(data: { feed_type: string; source: string; name: string; buffer_size: number }) {
  try {
    await appStore.addFeed(data)
    addFormRef.value?.reset()
    notificationStore.showToast('Feed added', 'success')
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Failed to add feed'
    addFormRef.value?.setError(msg)
  }
}

/**
 * Pause a feed by ID.
 *
 * @param feedId - The feed to pause
 */
async function onPause(feedId: string) {
  try {
    await appStore.pauseFeed(feedId)
    notificationStore.showToast('Feed paused', 'info')
  } catch {
    notificationStore.showToast('Failed to pause feed', 'error')
  }
}

/**
 * Resume a paused feed by ID.
 *
 * @param feedId - The feed to resume
 */
async function onResume(feedId: string) {
  try {
    await appStore.resumeFeed(feedId)
    notificationStore.showToast('Feed resumed', 'success')
  } catch {
    notificationStore.showToast('Failed to resume feed', 'error')
  }
}

/**
 * Delete a feed by ID.
 *
 * @param feedId - The feed to delete
 */
async function onDelete(feedId: string) {
  try {
    await appStore.removeFeed(feedId)
    notificationStore.showToast('Feed deleted', 'info')
  } catch {
    notificationStore.showToast('Failed to delete feed', 'error')
  }
}

onMounted(() => {
  appStore.fetchFeeds()
})
</script>

<template>
  <div class="feeds-view">
    <AddFeedForm ref="addFormRef" @submit="onAddFeed" />

    <div v-if="appStore.feeds.length === 0" class="empty-state">
      <div class="empty-state-title">No feeds</div>
      <div class="empty-state-text">Add a camera, RTSP stream, or video file above to get started.</div>
    </div>

    <div v-else class="feeds-grid">
      <FeedCard
        v-for="feed in appStore.feeds"
        :key="feed.id"
        :feed="feed"
        @pause="onPause"
        @resume="onResume"
        @delete="onDelete"
      />
    </div>
  </div>
</template>

<style scoped>
.feeds-view {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.feeds-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1rem;
}
</style>
