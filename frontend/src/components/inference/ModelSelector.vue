<script setup lang="ts">
/**
 * Dropdown selector for trained YOLO models.
 *
 * Displays models from the training store with name, dataset,
 * and mAP@50 info. Emits the selected model path via v-model.
 *
 * Props:
 *   modelValue - Currently selected model path (v-model)
 */
import { useTrainingStore } from '@/stores/training'

const props = defineProps<{
  modelValue: string | null
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string | null]
}>()

const trainingStore = useTrainingStore()

/**
 * Handle selection change from the <select> element.
 *
 * @param event - Native change event
 */
function onSelect(event: Event) {
  const value = (event.target as HTMLSelectElement).value
  emit('update:modelValue', value || null)
}
</script>

<template>
  <div class="form-group">
    <label class="form-label">Model</label>
    <select
      class="form-select"
      :value="props.modelValue || ''"
      @change="onSelect"
    >
      <option value="">Select a model...</option>
      <option
        v-for="model in trainingStore.models"
        :key="model.name"
        :value="model.path"
      >
        {{ model.name }}
        <template v-if="model.datasetName"> ({{ model.datasetName }})</template>
        <template v-if="model.bestMap50 !== null"> - mAP: {{ (model.bestMap50 * 100).toFixed(1) }}%</template>
        <template v-if="model.isActive"> [active]</template>
      </option>
    </select>
  </div>
</template>
