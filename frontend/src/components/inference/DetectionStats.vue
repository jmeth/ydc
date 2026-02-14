<script setup lang="ts">
/**
 * Real-time detection statistics by class.
 *
 * Displays detection count per class and average confidence,
 * updating from the current frame detections.
 *
 * Props:
 *   detections - Array of Detection objects from current frame
 */
import { computed } from 'vue'
import type { Detection } from '@/types/models'

const COLORS = [
  '#4fc3f7', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc',
  '#26c6da', '#ffca28', '#ec407a', '#8d6e63', '#78909c',
]

const props = defineProps<{
  detections: Detection[]
}>()

/** Group detections by class and compute counts + avg confidence. */
const classStats = computed(() => {
  const map = new Map<string, { classId: number; count: number; totalConf: number }>()
  for (const d of props.detections) {
    const key = d.className
    const existing = map.get(key)
    if (existing) {
      existing.count++
      existing.totalConf += d.confidence
    } else {
      map.set(key, { classId: d.classId, count: 1, totalConf: d.confidence })
    }
  }
  return Array.from(map.entries()).map(([name, stats]) => ({
    name,
    classId: stats.classId,
    count: stats.count,
    avgConfidence: stats.totalConf / stats.count,
  }))
})
</script>

<template>
  <div class="detection-stats card">
    <div class="card-header">Detections</div>

    <div v-if="classStats.length === 0" class="empty-state">
      <div class="empty-state-text">No detections</div>
    </div>

    <div
      v-for="stat in classStats"
      :key="stat.name"
      class="det-row"
    >
      <span
        class="det-color"
        :style="{ background: COLORS[stat.classId % COLORS.length] }"
      ></span>
      <span class="det-name">{{ stat.name }}</span>
      <span class="det-count">{{ stat.count }}</span>
      <span class="det-conf">{{ (stat.avgConfidence * 100).toFixed(0) }}%</span>
    </div>

    <div v-if="props.detections.length > 0" class="det-total">
      Total: {{ props.detections.length }}
    </div>
  </div>
</template>

<style scoped>
.det-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0;
  font-size: 0.8125rem;
}

.det-color {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
}

.det-name {
  flex: 1;
}

.det-count {
  font-weight: 600;
  font-family: var(--font-mono);
}

.det-conf {
  color: var(--color-text-muted);
  font-family: var(--font-mono);
  font-size: 0.75rem;
}

.det-total {
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--color-border);
  font-size: 0.8125rem;
  font-weight: 600;
}
</style>
