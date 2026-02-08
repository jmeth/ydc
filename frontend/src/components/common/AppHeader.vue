<script setup lang="ts">
/**
 * Application header bar.
 *
 * Shows the app title and dataset selector dropdown.
 */
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()
</script>

<template>
  <header class="app-header">
    <h1 class="app-title">YOLO Dataset Creator</h1>
    <div class="header-controls">
      <span v-if="appStore.currentDataset" class="current-dataset">
        Dataset: {{ appStore.currentDataset }}
      </span>
    </div>
  </header>
</template>

<style scoped>
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 1rem;
  background: var(--color-bg-header);
  border-bottom: 1px solid var(--color-border);
}

.app-title {
  font-size: 1.125rem;
  font-weight: 600;
  margin: 0;
}

.current-dataset {
  font-size: 0.875rem;
  color: var(--color-text-muted);
}
</style>
