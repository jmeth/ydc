<script setup lang="ts">
/**
 * Model mode view.
 *
 * Hosts inference player with trained models, model selector,
 * and detection stats. Placeholder until Phase 8.
 */
</script>

<template>
  <div class="model-view">
    <h1>Model Mode</h1>
    <p>Run inference using your trained YOLO models.</p>
    <div class="placeholder">
      <p>Inference player, model selector, and detection stats will be added in later phases.</p>
    </div>
  </div>
</template>

<style scoped>
.model-view {
  max-width: 1200px;
  margin: 0 auto;
}

.placeholder {
  border: 1px dashed var(--color-border);
  border-radius: 8px;
  padding: 3rem;
  text-align: center;
  color: var(--color-text-muted);
  margin-top: 1rem;
}
</style>
