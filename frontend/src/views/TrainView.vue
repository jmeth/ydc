<script setup lang="ts">
/**
 * Training mode view — configure and run YOLO model training.
 *
 * Stacked layout: TrainingConfig form at top, ProgressDisplay when
 * training is active, and TrainingHistory table showing past runs.
 *
 * On mount: fetches datasets, current training status, and history.
 */
import { computed, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import { useTrainingStore } from '@/stores/training'
import TrainingConfig from '@/components/training/TrainingConfig.vue'
import ProgressDisplay from '@/components/training/ProgressDisplay.vue'
import TrainingHistory from '@/components/training/TrainingHistory.vue'

const appStore = useAppStore()
const trainingStore = useTrainingStore()

const isTraining = computed(() => trainingStore.status === 'training')

onMounted(async () => {
  await Promise.all([
    appStore.fetchDatasets(),
    trainingStore.fetchStatus(),
    trainingStore.fetchHistory(),
  ])
})
</script>

<template>
  <div class="train-view">
    <TrainingConfig :disabled="isTraining" />
    <ProgressDisplay />
    <TrainingHistory />
  </div>
</template>

<style scoped>
.train-view {
  max-width: 800px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
</style>
