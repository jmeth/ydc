<script setup lang="ts">
/**
 * Canvas overlay for viewing, drawing, and editing YOLO annotations.
 *
 * Three modes:
 *   - view: Render annotations as colored bounding boxes
 *   - draw: Click-drag to create a new bounding box
 *   - edit: Select, drag to move, corner/edge handles to resize
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
 *   selectedIndex - Currently selected annotation index
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

/** Size of corner/edge resize handles in pixels. */
const HANDLE_SIZE = 8
const HANDLE_HALF = HANDLE_SIZE / 2

/** Handle positions: corners and edge midpoints. */
type HandleId = 'tl' | 'tr' | 'bl' | 'br' | 'tm' | 'bm' | 'ml' | 'mr'

/** Get a color for a class ID (wraps around palette). */
function colorFor(classId: number): string {
  return COLORS[classId % COLORS.length] ?? '#78909c'
}

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

/** Dragging state for edit mode (move). */
const isDragging = ref(false)
const dragOffset = ref({ x: 0, y: 0 })

/** Resizing state for edit mode (handle drag). */
const isResizing = ref(false)
const activeHandle = ref<HandleId | null>(null)
/** Anchor corner — the corner opposite to the handle being dragged. */
const resizeAnchor = ref({ x: 0, y: 0 })

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
 * Get the pixel positions of all 8 resize handles for a bounding box.
 *
 * @param x1 - Left pixel x
 * @param y1 - Top pixel y
 * @param x2 - Right pixel x
 * @param y2 - Bottom pixel y
 * @returns Record mapping HandleId to { x, y } center positions
 */
function getHandlePositions(x1: number, y1: number, x2: number, y2: number): Record<HandleId, { x: number; y: number }> {
  const mx = (x1 + x2) / 2
  const my = (y1 + y2) / 2
  return {
    tl: { x: x1, y: y1 },
    tr: { x: x2, y: y1 },
    bl: { x: x1, y: y2 },
    br: { x: x2, y: y2 },
    tm: { x: mx, y: y1 },
    bm: { x: mx, y: y2 },
    ml: { x: x1, y: my },
    mr: { x: x2, y: my },
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

/**
 * Render all annotations, handles, and the in-progress drawing box.
 */
function draw() {
  const canvas = canvasRef.value
  if (!canvas || !containerRef.value) return

  canvas.width = containerRef.value.clientWidth
  canvas.height = containerRef.value.clientHeight
  const ctx = canvas.getContext('2d')!

  // Draw each annotation
  props.annotations.forEach((ann, i) => {
    const { x1, y1, x2, y2 } = toPixel(ann)
    const color = colorFor(ann.classId)
    const isSelected = i === props.selectedIndex

    ctx.strokeStyle = color
    ctx.lineWidth = isSelected ? 3 : 2
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

    // Fill with low opacity for selected
    if (isSelected) {
      ctx.fillStyle = color + '22'
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1)

      // Draw resize handles
      const handles = getHandlePositions(x1, y1, x2, y2)
      for (const pos of Object.values(handles)) {
        ctx.fillStyle = '#ffffff'
        ctx.fillRect(pos.x - HANDLE_HALF, pos.y - HANDLE_HALF, HANDLE_SIZE, HANDLE_SIZE)
        ctx.strokeStyle = color
        ctx.lineWidth = 1.5
        ctx.strokeRect(pos.x - HANDLE_HALF, pos.y - HANDLE_HALF, HANDLE_SIZE, HANDLE_SIZE)
      }
    }

    // Label
    ctx.fillStyle = color
    const label = `${ann.classId}`
    ctx.font = '11px system-ui'
    const tw = ctx.measureText(label).width + 8
    ctx.fillRect(x1, y1 - 16, tw, 16)
    ctx.fillStyle = '#121212'
    ctx.fillText(label, x1 + 4, y1 - 4)
  })

  // Draw in-progress box
  if (isDrawing.value) {
    ctx.strokeStyle = colorFor(props.activeClassId)
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

/**
 * Test if a point is within a resize handle.
 * Returns the HandleId if hit, or null.
 *
 * @param pos - Mouse position in container coords
 * @param annIndex - Annotation index to test handles for
 */
function handleHitTest(pos: { x: number; y: number }, annIndex: number): HandleId | null {
  const ann = props.annotations[annIndex]
  if (!ann) return null

  const { x1, y1, x2, y2 } = toPixel(ann)
  const handles = getHandlePositions(x1, y1, x2, y2)
  const hitRadius = HANDLE_HALF + 3 // a bit generous for easier grabbing

  for (const [id, hpos] of Object.entries(handles)) {
    if (Math.abs(pos.x - hpos.x) <= hitRadius && Math.abs(pos.y - hpos.y) <= hitRadius) {
      return id as HandleId
    }
  }
  return null
}

/** Find which annotation is under the mouse position. */
function hitTest(pos: { x: number; y: number }): number {
  for (let i = props.annotations.length - 1; i >= 0; i--) {
    const ann = props.annotations[i]
    if (!ann) continue
    const { x1, y1, x2, y2 } = toPixel(ann)
    if (pos.x >= x1 && pos.x <= x2 && pos.y >= y1 && pos.y <= y2) {
      return i
    }
  }
  return -1
}

/**
 * Compute the anchor point (opposite corner/edge) for a resize handle.
 *
 * @param handle - Which handle is being dragged
 * @param x1 - Box left
 * @param y1 - Box top
 * @param x2 - Box right
 * @param y2 - Box bottom
 */
function getAnchorForHandle(handle: HandleId, x1: number, y1: number, x2: number, y2: number) {
  switch (handle) {
    case 'tl': return { x: x2, y: y2 }
    case 'tr': return { x: x1, y: y2 }
    case 'bl': return { x: x2, y: y1 }
    case 'br': return { x: x1, y: y1 }
    case 'tm': return { x: (x1 + x2) / 2, y: y2 }
    case 'bm': return { x: (x1 + x2) / 2, y: y1 }
    case 'ml': return { x: x2, y: (y1 + y2) / 2 }
    case 'mr': return { x: x1, y: (y1 + y2) / 2 }
  }
}

function onMouseDown(event: MouseEvent) {
  event.preventDefault() // prevent native image drag
  if (props.mode === 'view') return
  const pos = getMousePos(event)

  if (props.mode === 'draw') {
    isDrawing.value = true
    drawStart.value = pos
    drawEnd.value = pos
  } else if (props.mode === 'edit') {
    // First check if clicking a resize handle on the selected annotation
    if (props.selectedIndex >= 0) {
      const handle = handleHitTest(pos, props.selectedIndex)
      if (handle) {
        const ann = props.annotations[props.selectedIndex]!
        const { x1, y1, x2, y2 } = toPixel(ann)
        isResizing.value = true
        activeHandle.value = handle
        resizeAnchor.value = getAnchorForHandle(handle, x1, y1, x2, y2)
        return
      }
    }

    // Then check if clicking inside an annotation to move it
    const hit = hitTest(pos)
    const hitAnn = props.annotations[hit]
    if (hit >= 0 && hitAnn) {
      emit('select', hit)
      isDragging.value = true
      const { x1, y1 } = toPixel(hitAnn)
      dragOffset.value = { x: pos.x - x1, y: pos.y - y1 }
    } else {
      emit('select', -1)
    }
  }
}

function onMouseMove(event: MouseEvent) {
  event.preventDefault()
  const pos = getMousePos(event)

  if (isDrawing.value) {
    drawEnd.value = pos
    draw()
  } else if (isResizing.value && props.selectedIndex >= 0 && activeHandle.value) {
    // Resize: compute new box from anchor + current mouse position
    const ann = props.annotations[props.selectedIndex]
    if (!ann) return
    const { x1, y1, x2, y2 } = toPixel(ann)
    const handle = activeHandle.value
    const anchor = resizeAnchor.value

    // Determine new box edges based on which handle is dragged
    let newX1 = x1, newY1 = y1, newX2 = x2, newY2 = y2
    switch (handle) {
      case 'tl': newX1 = pos.x; newY1 = pos.y; break
      case 'tr': newX2 = pos.x; newY1 = pos.y; break
      case 'bl': newX1 = pos.x; newY2 = pos.y; break
      case 'br': newX2 = pos.x; newY2 = pos.y; break
      case 'tm': newY1 = pos.y; break
      case 'bm': newY2 = pos.y; break
      case 'ml': newX1 = pos.x; break
      case 'mr': newX2 = pos.x; break
    }

    // For corner handles, use anchor as the opposite corner
    if (['tl', 'tr', 'bl', 'br'].includes(handle)) {
      if (handle === 'tl') { newX2 = anchor.x; newY2 = anchor.y }
      else if (handle === 'tr') { newX1 = anchor.x; newY2 = anchor.y }
      else if (handle === 'bl') { newX2 = anchor.x; newY1 = anchor.y }
      else if (handle === 'br') { newX1 = anchor.x; newY1 = anchor.y }
    }

    const norm = toNormalized(newX1, newY1, newX2, newY2)
    const updated = [...props.annotations]
    updated[props.selectedIndex] = { classId: ann.classId, ...norm }
    emit('update:annotations', updated)
  } else if (isDragging.value && props.selectedIndex >= 0) {
    const ann = props.annotations[props.selectedIndex]
    if (!ann) return
    const { x1, y1, x2, y2 } = toPixel(ann)
    const w = x2 - x1
    const h = y2 - y1
    const newX1 = pos.x - dragOffset.value.x
    const newY1 = pos.y - dragOffset.value.y
    const norm = toNormalized(newX1, newY1, newX1 + w, newY1 + h)
    const updated = [...props.annotations]
    updated[props.selectedIndex] = { classId: ann.classId, ...norm }
    emit('update:annotations', updated)
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
  }
  isDragging.value = false
  isResizing.value = false
  activeHandle.value = null
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

/**
 * Get the CSS cursor for the current mouse position in edit mode.
 * Shows resize cursors over handles, move cursor over selected box.
 */
function getCursorForPos(pos: { x: number; y: number }): string {
  if (props.selectedIndex >= 0) {
    const handle = handleHitTest(pos, props.selectedIndex)
    if (handle) {
      const cursorMap: Record<HandleId, string> = {
        tl: 'nwse-resize', br: 'nwse-resize',
        tr: 'nesw-resize', bl: 'nesw-resize',
        tm: 'ns-resize', bm: 'ns-resize',
        ml: 'ew-resize', mr: 'ew-resize',
      }
      return cursorMap[handle]
    }
  }

  const hit = hitTest(pos)
  if (hit >= 0) return 'move'
  return 'default'
}

/** Update cursor based on mouse position during edit mode. */
function updateCursor(event: MouseEvent) {
  if (props.mode !== 'edit' || !containerRef.value) return
  if (isDragging.value || isResizing.value) return // keep current cursor during drag
  const pos = getMousePos(event)
  containerRef.value.style.cursor = getCursorForPos(pos)
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
    @mousemove="onMouseMove($event); updateCursor($event)"
    @mouseup="onMouseUp"
    @mouseleave="onMouseUp"
  >
    <img
      ref="imageRef"
      :src="props.imageSrc"
      class="canvas-image"
      alt="Annotation target"
      draggable="false"
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
  pointer-events: none;
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
