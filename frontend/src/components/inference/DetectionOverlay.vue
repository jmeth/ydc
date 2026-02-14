<script setup lang="ts">
/**
 * SVG overlay rendering bounding boxes for detected objects.
 *
 * Takes detection results (bbox in pixel coords [x1,y1,x2,y2]) and
 * the container dimensions to render colored rectangles with class
 * name and confidence labels.
 *
 * Props:
 *   detections - Array of Detection objects with bbox pixel coords
 *   width - Container width in pixels (for SVG viewBox)
 *   height - Container height in pixels (for SVG viewBox)
 */
import type { Detection } from '@/types/models'

const props = defineProps<{
  detections: Detection[]
  width: number
  height: number
}>()

/** Predefined color palette for class IDs. */
const COLORS = [
  '#4fc3f7', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc',
  '#26c6da', '#ffca28', '#ec407a', '#8d6e63', '#78909c',
]

/**
 * Get a consistent color for a class ID.
 *
 * @param classId - Zero-based class index
 * @returns Hex color string
 */
function colorForClass(classId: number): string {
  return COLORS[classId % COLORS.length] ?? '#78909c'
}
</script>

<template>
  <svg
    v-if="props.width > 0 && props.height > 0"
    class="detection-overlay"
    :viewBox="`0 0 ${props.width} ${props.height}`"
    preserveAspectRatio="none"
  >
    <g v-for="(det, i) in props.detections" :key="i">
      <!-- Bounding box rectangle -->
      <rect
        :x="det.bbox[0]"
        :y="det.bbox[1]"
        :width="det.bbox[2] - det.bbox[0]"
        :height="det.bbox[3] - det.bbox[1]"
        :stroke="colorForClass(det.classId)"
        stroke-width="2"
        fill="none"
      />
      <!-- Label background -->
      <rect
        :x="det.bbox[0]"
        :y="Math.max(0, det.bbox[1] - 18)"
        :width="(det.className.length + 5) * 7"
        height="18"
        :fill="colorForClass(det.classId)"
        rx="2"
      />
      <!-- Label text -->
      <text
        :x="det.bbox[0] + 3"
        :y="Math.max(0, det.bbox[1] - 18) + 13"
        fill="#121212"
        font-size="12"
        font-weight="600"
        font-family="system-ui, sans-serif"
      >
        {{ det.className }} {{ (det.confidence * 100).toFixed(0) }}%
      </text>
    </g>
  </svg>
</template>

<style scoped>
.detection-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
</style>
