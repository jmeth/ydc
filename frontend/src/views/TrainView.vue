<script setup lang="ts">
/**
 * Training mode view.
 *
 * Hosts training config form, progress display, and model list.
 * Placeholder until Phase 7.
 */
</script>

<template>
  <div class="train-view">
    <h1>Train Mode</h1>
    <p>Configure and run YOLO model training on your datasets.</p>
    <div class="placeholder">
      <p>Training configuration, progress display, and model list will be added in later phases.</p>
    </div>
  </div>
</template>

<style scoped>
.train-view {
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
