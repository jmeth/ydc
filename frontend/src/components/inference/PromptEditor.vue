<script setup lang="ts">
/**
 * Editor for YOLO-World class prompts.
 *
 * Allows editing a Record<number, string[]> mapping of class IDs
 * to prompt text arrays. Supports adding/removing classes and
 * editing prompt text per class. Emits save event with updated prompts.
 *
 * Props:
 *   prompts - Current prompt mapping
 *   disabled - Whether editing is disabled
 */
import { ref, watch } from 'vue'

const props = defineProps<{
  prompts: Record<number, string[]>
  disabled?: boolean
}>()

const emit = defineEmits<{
  save: [prompts: Record<number, string[]>]
}>()

/** Local mutable copy of prompts for editing. */
const localPrompts = ref<Record<number, string[]>>({})

/** Sync local state when props change externally. */
watch(
  () => props.prompts,
  (val) => {
    localPrompts.value = JSON.parse(JSON.stringify(val))
  },
  { immediate: true, deep: true },
)

/**
 * Update a prompt string for a specific class.
 *
 * @param classId - Class ID to update
 * @param text - Comma-separated prompt strings
 */
function updatePrompt(classId: number, text: string) {
  localPrompts.value[classId] = text.split(',').map((s) => s.trim()).filter(Boolean)
}

/** Add a new class entry with the next available ID. */
function addClass() {
  const ids = Object.keys(localPrompts.value).map(Number)
  const nextId = ids.length > 0 ? Math.max(...ids) + 1 : 0
  localPrompts.value[nextId] = ['']
}

/**
 * Remove a class entry.
 *
 * @param classId - Class ID to remove
 */
function removeClass(classId: number) {
  delete localPrompts.value[classId]
}

/** Emit save event with the current local prompts. */
function save() {
  emit('save', { ...localPrompts.value })
}
</script>

<template>
  <div class="prompt-editor card">
    <div class="card-header">
      Class Prompts
    </div>

    <div v-if="Object.keys(localPrompts).length === 0" class="empty-state">
      <div class="empty-state-text">No prompts configured</div>
    </div>

    <div
      v-for="(prompts, classId) in localPrompts"
      :key="classId"
      class="prompt-row"
    >
      <span class="prompt-class-id">{{ classId }}</span>
      <input
        type="text"
        class="form-input prompt-input"
        :value="prompts.join(', ')"
        :disabled="props.disabled"
        placeholder="e.g. a cat, feline"
        @input="updatePrompt(Number(classId), ($event.target as HTMLInputElement).value)"
      />
      <button
        class="btn btn-sm btn-danger"
        :disabled="props.disabled"
        @click="removeClass(Number(classId))"
      >
        &times;
      </button>
    </div>

    <div class="prompt-actions">
      <button
        class="btn btn-sm"
        :disabled="props.disabled"
        @click="addClass"
      >
        + Add Class
      </button>
      <button
        class="btn btn-sm btn-primary"
        :disabled="props.disabled"
        @click="save"
      >
        Save Prompts
      </button>
    </div>
  </div>
</template>

<style scoped>
.prompt-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.prompt-class-id {
  min-width: 1.5rem;
  text-align: center;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--color-text-muted);
  font-family: var(--font-mono);
}

.prompt-input {
  flex: 1;
}

.prompt-actions {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.5rem;
}
</style>
