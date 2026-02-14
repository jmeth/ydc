<script setup lang="ts">
/**
 * Dropdown selector for available video feeds.
 *
 * Displays feeds from the app store grouped by type (raw/inference)
 * with a status indicator dot. Emits the selected feed ID via v-model.
 *
 * Props:
 *   modelValue - Currently selected feed ID (v-model)
 */
import { computed } from 'vue'
import { useAppStore } from '@/stores/app'

const props = defineProps<{
  modelValue: string | null
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string | null]
}>()

const appStore = useAppStore()

/** Raw (source) feeds. */
const rawFeeds = computed(() =>
  appStore.feeds.filter((f) => f.type === 'raw')
)

/** Inference (derived) feeds. */
const inferenceFeeds = computed(() =>
  appStore.feeds.filter((f) => f.type === 'inference')
)

/**
 * Handle selection change from the <select> element.
 *
 * @param event - Native change event
 */
function onSelect(event: Event) {
  const value = (event.target as HTMLSelectElement).value
  emit('update:modelValue', value || null)
}
</script>

<template>
  <div class="form-group">
    <label class="form-label">Video Feed</label>
    <select
      class="form-select"
      :value="props.modelValue || ''"
      @change="onSelect"
    >
      <option value="">Select a feed...</option>

      <optgroup v-if="rawFeeds.length" label="Camera Feeds">
        <option
          v-for="feed in rawFeeds"
          :key="feed.id"
          :value="feed.id"
        >
          {{ feed.name }} ({{ feed.status }})
        </option>
      </optgroup>

      <optgroup v-if="inferenceFeeds.length" label="Inference Feeds">
        <option
          v-for="feed in inferenceFeeds"
          :key="feed.id"
          :value="feed.id"
        >
          {{ feed.name }} ({{ feed.status }})
        </option>
      </optgroup>
    </select>
  </div>
</template>
