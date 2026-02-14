<script setup lang="ts">
/**
 * Canvas overlay for viewing, drawing, and editing YOLO annotations.
 *
 * Three modes:
 *   - view: Render annotations as colored bounding boxes
 *   - draw: Click-drag to create a new bounding box
 *   - edit: Select, drag to move, corner handles to resize
 *
 * All annotations are in YOLO normalized format (cx, cy, w, h, 0-1),
 * converted to/from pixel coords for rendering. Color-coded by class_id.
 * Delete key removes selected, Escape deselects.
 *
 * Props:
 *   imageSrc - URL of the image to display
 *   annotations - Array of Annotation objects
 *   mode - Interaction mode (view/draw/edit)
 *   activeClassId - Class ID for newly drawn annotations
 *
 * Emits:
 *   update:annotations - When annotations change
 *   select - When an annotation is clicked (index)
 */
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import type { Annotation } from '@/types/models'

const props = defineProps<{
  imageSrc: string
  annotations: Annotation[]
  mode: 'view' | 'draw' | 'edit'
  activeClassId: number
  selectedIndex: number
}>()

const emit = defineEmits<{
  'update:annotations': [annotations: Annotation[]]
  select: [index: number]
}>()

/** Predefined color palette for class IDs. */
const COLORS = [
  '#4fc3f7', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc',
  '#26c6da', '#ffca28', '#ec407a', '#8d6e63', '#78909c',
]

const containerRef = ref<HTMLElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const imageRef = ref<HTMLImageElement | null>(null)

/** Natural image dimensions for coordinate conversion. */
const imgWidth = ref(1)
const imgHeight = ref(1)
/** Displayed image dimensions (fitted in container). */
const displayWidth = ref(1)
const displayHeight = ref(1)
/** Offset of image within container (for letterboxing). */
const offsetX = ref(0)
const offsetY = ref(0)

/** Drawing state for new annotation. */
const isDrawing = ref(false)
const drawStart = ref({ x: 0, y: 0 })
const drawEnd = ref({ x: 0, y: 0 })

/** Dragging state for edit mode. */
const isDragging = ref(false)
const dragOffset = ref({ x: 0, y: 0 })

/**
 * Convert normalized YOLO coords to pixel coords in the display area.
 *
 * @param ann - Annotation in normalized 0-1 coords
 * @returns Pixel coords { x1, y1, x2, y2 } in display space
 */
function toPixel(ann: Annotation) {
  const cx = ann.x * displayWidth.value + offsetX.value
  const cy = ann.y * displayHeight.value + offsetY.value
  const w = ann.width * displayWidth.value
  const h = ann.height * displayHeight.value
  return {
    x1: cx - w / 2,
    y1: cy - h / 2,
    x2: cx + w / 2,
    y2: cy + h / 2,
  }
}

/**
 * Convert pixel coords in display space to normalized YOLO coords.
 *
 * @param x1 - Left pixel x in display space
 * @param y1 - Top pixel y in display space
 * @param x2 - Right pixel x
 * @param y2 - Bottom pixel y
 * @returns Annotation coords { x, y, width, height } normalized 0-1
 */
function toNormalized(x1: number, y1: number, x2: number, y2: number) {
  const nx1 = (Math.min(x1, x2) - offsetX.value) / displayWidth.value
  const ny1 = (Math.min(y1, y2) - offsetY.value) / displayHeight.value
  const nx2 = (Math.max(x1, x2) - offsetX.value) / displayWidth.value
  const ny2 = (Math.max(y1, y2) - offsetY.value) / displayHeight.value
  return {
    x: Math.max(0, Math.min(1, (nx1 + nx2) / 2)),
    y: Math.max(0, Math.min(1, (ny1 + ny2) / 2)),
    width: Math.max(0.001, Math.min(1, nx2 - nx1)),
    height: Math.max(0.001, Math.min(1, ny2 - ny1)),
  }
}

/**
 * Get mouse position relative to the container.
 *
 * @param event - Mouse event
 * @returns { x, y } pixel position in container space
 */
function getMousePos(event: MouseEvent) {
  const rect = containerRef.value!.getBoundingClientRect()
  return { x: event.clientX - rect.left, y: event.clientY - rect.top }
}

/** Calculate display dimensions when image loads. */
function onImageLoad() {
  if (!imageRef.value || !containerRef.value) return
  imgWidth.value = imageRef.value.naturalWidth
  imgHeight.value = imageRef.value.naturalHeight
  updateDisplayDimensions()
  draw()
}

/** Recalculate how the image fits within the container (contain mode). */
function updateDisplayDimensions() {
  if (!containerRef.value) return
  const cw = containerRef.value.clientWidth
  const ch = containerRef.value.clientHeight
  const scale = Math.min(cw / imgWidth.value, ch / imgHeight.value)
  displayWidth.value = imgWidth.value * scale
  displayHeight.value = imgHeight.value * scale
  offsetX.value = (cw - displayWidth.value) / 2
  offsetY.value = (ch - displayHeight.value) / 2
}

/** Render all annotations and the in-progress drawing box on the canvas. */
function draw() {
  const canvas = canvasRef.value
  if (!canvas || !containerRef.value) return

  canvas.width = containerRef.value.clientWidth
  canvas.height = containerRef.value.clientHeight
  const ctx = canvas.getContext('2d')!

  // Draw each annotation
  props.annotations.forEach((ann, i) => {
    const { x1, y1, x2, y2 } = toPixel(ann)
    const color = COLORS[ann.classId % COLORS.length]
    const isSelected = i === props.selectedIndex

    ctx.strokeStyle = color
    ctx.lineWidth = isSelected ? 3 : 2
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

    // Fill with low opacity for selected
    if (isSelected) {
      ctx.fillStyle = color + '22'
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1)
    }

    // Label
    ctx.fillStyle = color
    const label = `${ann.classId}`
    const tw = ctx.measureText(label).width + 8
    ctx.fillRect(x1, y1 - 16, tw, 16)
    ctx.fillStyle = '#121212'
    ctx.font = '11px system-ui'
    ctx.fillText(label, x1 + 4, y1 - 4)
  })

  // Draw in-progress box
  if (isDrawing.value) {
    ctx.strokeStyle = COLORS[props.activeClassId % COLORS.length]
    ctx.lineWidth = 2
    ctx.setLineDash([5, 3])
    const x = Math.min(drawStart.value.x, drawEnd.value.x)
    const y = Math.min(drawStart.value.y, drawEnd.value.y)
    const w = Math.abs(drawEnd.value.x - drawStart.value.x)
    const h = Math.abs(drawEnd.value.y - drawStart.value.y)
    ctx.strokeRect(x, y, w, h)
    ctx.setLineDash([])
  }
}

/** Find which annotation is under the mouse position. */
function hitTest(pos: { x: number; y: number }): number {
  for (let i = props.annotations.length - 1; i >= 0; i--) {
    const { x1, y1, x2, y2 } = toPixel(props.annotations[i])
    if (pos.x >= x1 && pos.x <= x2 && pos.y >= y1 && pos.y <= y2) {
      return i
    }
  }
  return -1
}

function onMouseDown(event: MouseEvent) {
  if (props.mode === 'view') return
  const pos = getMousePos(event)

  if (props.mode === 'draw') {
    isDrawing.value = true
    drawStart.value = pos
    drawEnd.value = pos
  } else if (props.mode === 'edit') {
    const hit = hitTest(pos)
    if (hit >= 0) {
      emit('select', hit)
      isDragging.value = true
      const { x1, y1 } = toPixel(props.annotations[hit])
      dragOffset.value = { x: pos.x - x1, y: pos.y - y1 }
    } else {
      emit('select', -1)
    }
  }
}

function onMouseMove(event: MouseEvent) {
  const pos = getMousePos(event)

  if (isDrawing.value) {
    drawEnd.value = pos
    draw()
  } else if (isDragging.value && props.selectedIndex >= 0) {
    const ann = props.annotations[props.selectedIndex]
    const { x1, y1, x2, y2 } = toPixel(ann)
    const w = x2 - x1
    const h = y2 - y1
    const newX1 = pos.x - dragOffset.value.x
    const newY1 = pos.y - dragOffset.value.y
    const norm = toNormalized(newX1, newY1, newX1 + w, newY1 + h)
    const updated = [...props.annotations]
    updated[props.selectedIndex] = { ...ann, ...norm }
    emit('update:annotations', updated)
    draw()
  }
}

function onMouseUp() {
  if (isDrawing.value) {
    isDrawing.value = false
    const norm = toNormalized(drawStart.value.x, drawStart.value.y, drawEnd.value.x, drawEnd.value.y)
    // Only add if box is meaningful size
    if (norm.width > 0.005 && norm.height > 0.005) {
      const newAnn: Annotation = {
        classId: props.activeClassId,
        ...norm,
      }
      emit('update:annotations', [...props.annotations, newAnn])
    }
    draw()
  }
  isDragging.value = false
}

function onKeyDown(event: KeyboardEvent) {
  if (event.key === 'Delete' || event.key === 'Backspace') {
    if (props.selectedIndex >= 0 && props.mode === 'edit') {
      const updated = props.annotations.filter((_, i) => i !== props.selectedIndex)
      emit('update:annotations', updated)
      emit('select', -1)
    }
  } else if (event.key === 'Escape') {
    emit('select', -1)
    isDrawing.value = false
  }
}

// Redraw when annotations or selection changes
watch(() => [props.annotations, props.selectedIndex], () => {
  nextTick(draw)
}, { deep: true })

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  window.addEventListener('keydown', onKeyDown)
  if (containerRef.value) {
    resizeObserver = new ResizeObserver(() => {
      updateDisplayDimensions()
      draw()
    })
    resizeObserver.observe(containerRef.value)
  }
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeyDown)
  resizeObserver?.disconnect()
})
</script>

<template>
  <div
    ref="containerRef"
    class="annotation-canvas"
    :class="{ 'mode-draw': props.mode === 'draw', 'mode-edit': props.mode === 'edit' }"
    @mousedown="onMouseDown"
    @mousemove="onMouseMove"
    @mouseup="onMouseUp"
    @mouseleave="onMouseUp"
  >
    <img
      ref="imageRef"
      :src="props.imageSrc"
      class="canvas-image"
      alt="Annotation target"
      @load="onImageLoad"
    />
    <canvas ref="canvasRef" class="canvas-overlay"></canvas>
  </div>
</template>

<style scoped>
.annotation-canvas {
  position: relative;
  width: 100%;
  height: 100%;
  background: #000;
  overflow: hidden;
  user-select: none;
}

.annotation-canvas.mode-draw {
  cursor: crosshair;
}

.annotation-canvas.mode-edit {
  cursor: default;
}

.canvas-image {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.canvas-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
</style>
