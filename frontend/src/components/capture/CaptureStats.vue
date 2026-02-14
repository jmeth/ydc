<script setup lang="ts">
/**
 * Displays capture session statistics.
 *
 * Shows total, positive, and negative capture counts
 * from the capture store in a compact stat row.
 */
import { useCaptureStore } from '@/stores/capture'

const captureStore = useCaptureStore()
</script>

<template>
  <div v-if="captureStore.status === 'running'" class="stat-row">
    <div class="stat-item">
      <span class="stat-label">Total</span>
      <span class="stat-value">{{ captureStore.stats.totalCaptures }}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Positive</span>
      <span class="stat-value">{{ captureStore.stats.positiveCaptures }}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Negative</span>
      <span class="stat-value">{{ captureStore.stats.negativeCaptures }}</span>
    </div>
  </div>
</template>
