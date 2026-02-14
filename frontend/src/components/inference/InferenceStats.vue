<script setup lang="ts">
/**
 * Compact stats row showing inference performance metrics.
 *
 * Displays FPS, detection count, and inference time in a
 * horizontal row. Used below the video player in ScanView
 * and ModelView.
 *
 * Props:
 *   fps - Current frames per second
 *   detectionCount - Number of detections in current frame
 *   inferenceTimeMs - Inference time in milliseconds
 */
defineProps<{
  fps: number
  detectionCount: number
  inferenceTimeMs: number
}>()
</script>

<template>
  <div class="stat-row">
    <div class="stat-item">
      <span class="stat-label">FPS</span>
      <span class="stat-value">{{ fps.toFixed(1) }}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Detections</span>
      <span class="stat-value">{{ detectionCount }}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Inference</span>
      <span class="stat-value">{{ inferenceTimeMs.toFixed(0) }}ms</span>
    </div>
  </div>
</template>
