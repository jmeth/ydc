<script setup lang="ts">
/**
 * Sidebar list of annotations for the image editor.
 *
 * Displays each annotation's class, coordinates, and a delete button.
 * Click to select (syncs with canvas selection). Class selector
 * dropdown per annotation to change class ID.
 *
 * Props:
 *   annotations - Array of Annotation objects
 *   classes - Available class names (index = class_id)
 *   selectedIndex - Currently selected annotation index
 *
 * Emits:
 *   select - When an annotation row is clicked (index)
 *   update:annotations - When annotation class changes
 *   delete - When delete button is clicked (index)
 */
import type { Annotation } from '@/types/models'

const COLORS = [
  '#4fc3f7', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc',
  '#26c6da', '#ffca28', '#ec407a', '#8d6e63', '#78909c',
]

const props = defineProps<{
  annotations: Annotation[]
  classes: string[]
  selectedIndex: number
}>()

const emit = defineEmits<{
  select: [index: number]
  'update:annotations': [annotations: Annotation[]]
  delete: [index: number]
}>()

/**
 * Update the class ID of an annotation.
 *
 * @param index - Annotation index
 * @param classId - New class ID
 */
function changeClass(index: number, classId: number) {
  const updated = [...props.annotations]
  updated[index] = { ...updated[index], classId }
  emit('update:annotations', updated)
}
</script>

<template>
  <div class="annotation-list">
    <div class="list-header">
      Annotations ({{ props.annotations.length }})
    </div>

    <div v-if="props.annotations.length === 0" class="empty-state">
      <div class="empty-state-text">No annotations</div>
    </div>

    <div
      v-for="(ann, i) in props.annotations"
      :key="i"
      class="annotation-row"
      :class="{ selected: i === props.selectedIndex }"
      @click="emit('select', i)"
    >
      <span
        class="ann-color"
        :style="{ background: COLORS[ann.classId % COLORS.length] }"
      ></span>

      <select
        class="form-select ann-class-select"
        :value="ann.classId"
        @change="changeClass(i, Number(($event.target as HTMLSelectElement).value))"
        @click.stop
      >
        <option
          v-for="(cls, idx) in props.classes"
          :key="idx"
          :value="idx"
        >
          {{ idx }}: {{ cls }}
        </option>
      </select>

      <span class="ann-coords" :title="`cx=${ann.x.toFixed(3)} cy=${ann.y.toFixed(3)} w=${ann.width.toFixed(3)} h=${ann.height.toFixed(3)}`">
        {{ ann.x.toFixed(2) }}, {{ ann.y.toFixed(2) }}
      </span>

      <button
        class="btn btn-sm btn-danger ann-delete"
        @click.stop="emit('delete', i)"
      >
        &times;
      </button>
    </div>
  </div>
</template>

<style scoped>
.annotation-list {
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}

.list-header {
  font-size: 0.8125rem;
  font-weight: 600;
  padding: 0.5rem 0;
  color: var(--color-text-muted);
}

.annotation-row {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.375rem 0.25rem;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 0.75rem;
}

.annotation-row:hover {
  background: var(--color-bg-surface);
}

.annotation-row.selected {
  background: var(--color-bg-surface);
  outline: 1px solid var(--color-accent);
}

.ann-color {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
}

.ann-class-select {
  width: 90px;
  padding: 0.125rem 0.25rem;
  font-size: 0.6875rem;
}

.ann-coords {
  flex: 1;
  color: var(--color-text-muted);
  font-family: var(--font-mono);
  font-size: 0.6875rem;
}

.ann-delete {
  padding: 0.125rem 0.375rem;
  font-size: 0.75rem;
  line-height: 1;
}
</style>
