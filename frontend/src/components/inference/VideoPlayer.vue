<script setup lang="ts">
/**
 * Video player component that displays a live feed via WebSocket.
 *
 * Subscribes to a video feed using the useVideoStream composable,
 * renders base64-encoded JPEG frames in an <img> element with a
 * 16:9 aspect ratio container, and overlays DetectionOverlay for
 * bounding boxes when detections are present.
 *
 * Props:
 *   feedId - Feed ID to subscribe to (null = no feed)
 */
import { ref, computed, toRef, onMounted, onUnmounted } from 'vue'
import { useVideoStream } from '@/composables/useVideoStream'
import DetectionOverlay from './DetectionOverlay.vue'

const props = defineProps<{
  feedId: string | null
}>()

const feedIdRef = toRef(props, 'feedId')
const { frame, fps, detections, isConnected } = useVideoStream(feedIdRef)

/** Container element ref for measuring dimensions. */
const containerRef = ref<HTMLElement | null>(null)
const containerWidth = ref(0)
const containerHeight = ref(0)

/** Image natural dimensions for detection overlay viewBox. */
const imageRef = ref<HTMLImageElement | null>(null)
const imageWidth = ref(640)
const imageHeight = ref(480)

/** Frame source as data URL for the <img> element. */
const frameSrc = computed(() => {
  if (!frame.value) return null
  return `data:image/jpeg;base64,${frame.value}`
})

/**
 * Update natural image dimensions when a new frame loads.
 * Used as the viewBox for the detection overlay SVG so
 * bounding box pixel coords map correctly.
 */
function onImageLoad() {
  if (imageRef.value) {
    imageWidth.value = imageRef.value.naturalWidth
    imageHeight.value = imageRef.value.naturalHeight
  }
}

/** Update container dimensions on resize. */
function updateContainerSize() {
  if (containerRef.value) {
    containerWidth.value = containerRef.value.clientWidth
    containerHeight.value = containerRef.value.clientHeight
  }
}

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  updateContainerSize()
  if (containerRef.value) {
    resizeObserver = new ResizeObserver(updateContainerSize)
    resizeObserver.observe(containerRef.value)
  }
})

onUnmounted(() => {
  resizeObserver?.disconnect()
})

defineExpose({ fps, detections, isConnected })
</script>

<template>
  <div ref="containerRef" class="video-player">
    <!-- Active feed with frame data -->
    <div v-if="frameSrc" class="video-frame-container">
      <img
        ref="imageRef"
        :src="frameSrc"
        class="video-frame"
        alt="Video feed"
        @load="onImageLoad"
      />
      <DetectionOverlay
        v-if="detections.length > 0"
        :detections="detections"
        :width="imageWidth"
        :height="imageHeight"
      />
    </div>

    <!-- No feed selected -->
    <div v-else-if="!props.feedId" class="empty-state">
      <div class="empty-state-title">No feed selected</div>
      <div class="empty-state-text">Select a video feed to begin</div>
    </div>

    <!-- Feed selected but no frames yet -->
    <div v-else class="empty-state">
      <div class="loading-spinner"></div>
      <div class="empty-state-text">Connecting to feed...</div>
    </div>
  </div>
</template>

<style scoped>
.video-player {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #000;
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.video-frame-container {
  position: relative;
  width: 100%;
  height: 100%;
}

.video-frame {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
</style>
