<script setup lang="ts">
/**
 * Form for adding a new video feed.
 *
 * Provides inputs for feed type (camera/rtsp/file), source identifier,
 * optional name, and buffer size. Placeholder text on the source input
 * changes based on selected feed type.
 *
 * Emits:
 *   submit - Form data for creating a new feed
 */
import { ref, computed } from 'vue'

const emit = defineEmits<{
  submit: [data: { feed_type: string; source: string; name: string; buffer_size: number }]
}>()

/** Selected feed type. */
const feedType = ref('camera')
/** Source identifier (camera index, RTSP URL, or file path). */
const source = ref('')
/** Optional human-readable name. */
const name = ref('')
/** Ring buffer size. */
const bufferSize = ref(30)
/** Whether a submission is in progress. */
const submitting = ref(false)
/** Error message from a failed submission. */
const error = ref<string | null>(null)

/**
 * Placeholder text for the source input, varies by feed type.
 */
const sourcePlaceholder = computed(() => {
  switch (feedType.value) {
    case 'camera':
      return 'Camera index (e.g. 0)'
    case 'rtsp':
      return 'RTSP URL (e.g. rtsp://192.168.1.10:554/stream)'
    case 'file':
      return 'Video file path (e.g. /path/to/video.mp4)'
    default:
      return 'Source identifier'
  }
})

/** Whether the form is valid for submission. */
const isValid = computed(() => source.value.trim().length > 0)

/**
 * Handle form submission. Emits the submit event and resets on success.
 * The parent is responsible for calling the API; this component
 * exposes setError() and setSubmitting() for the parent to control state.
 */
function onSubmit() {
  if (!isValid.value || submitting.value) return
  error.value = null
  submitting.value = true

  emit('submit', {
    feed_type: feedType.value,
    source: source.value.trim(),
    name: name.value.trim(),
    buffer_size: bufferSize.value,
  })
}

/**
 * Reset the form to its initial state after a successful add.
 */
function reset() {
  source.value = ''
  name.value = ''
  bufferSize.value = 30
  error.value = null
  submitting.value = false
}

/**
 * Set an error message (called by parent on API failure).
 *
 * @param msg - Error message to display
 */
function setError(msg: string) {
  error.value = msg
  submitting.value = false
}

/**
 * Set the submitting state (called by parent when done).
 *
 * @param val - Whether still submitting
 */
function setSubmitting(val: boolean) {
  submitting.value = val
}

defineExpose({ reset, setError, setSubmitting })
</script>

<template>
  <div class="card add-feed-form">
    <div class="card-header">Add Feed</div>

    <form @submit.prevent="onSubmit">
      <div class="form-row">
        <div class="form-group">
          <label class="form-label" for="feed-type">Type</label>
          <select
            id="feed-type"
            v-model="feedType"
            class="form-select"
          >
            <option value="camera">Camera</option>
            <option value="rtsp">RTSP</option>
            <option value="file">File</option>
          </select>
        </div>

        <div class="form-group form-group-grow">
          <label class="form-label" for="feed-source">Source</label>
          <input
            id="feed-source"
            v-model="source"
            type="text"
            class="form-input"
            :placeholder="sourcePlaceholder"
          />
        </div>
      </div>

      <div class="form-row">
        <div class="form-group form-group-grow">
          <label class="form-label" for="feed-name">Name (optional)</label>
          <input
            id="feed-name"
            v-model="name"
            type="text"
            class="form-input"
            placeholder="Human-readable name"
          />
        </div>

        <div class="form-group">
          <label class="form-label" for="feed-buffer">Buffer Size</label>
          <input
            id="feed-buffer"
            v-model.number="bufferSize"
            type="number"
            class="form-input"
            min="1"
            max="300"
          />
        </div>
      </div>

      <div v-if="error" class="form-error">{{ error }}</div>

      <button
        type="submit"
        class="btn btn-primary"
        :disabled="!isValid || submitting"
      >
        <span v-if="submitting" class="loading-spinner loading-spinner-sm" />
        {{ submitting ? 'Adding...' : 'Add Feed' }}
      </button>
    </form>
  </div>
</template>

<style scoped>
.add-feed-form form {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.form-row {
  display: flex;
  gap: 0.75rem;
}

.form-group-grow {
  flex: 1;
}

.form-error {
  font-size: 0.8125rem;
  color: var(--color-error);
  padding: 0.375rem 0.625rem;
  background: rgba(239, 83, 80, 0.1);
  border-radius: var(--radius-md);
}

@media (max-width: 600px) {
  .form-row {
    flex-direction: column;
  }
}
</style>
