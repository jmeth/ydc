<script setup lang="ts">
/**
 * Dataset mode view.
 *
 * Hosts image grid, annotation editor, review queue, and bulk
 * actions. Placeholder until Phase 6.
 */
</script>

<template>
  <div class="dataset-view">
    <h1>Dataset Mode</h1>
    <p>Review, edit, and manage images and annotations in your dataset.</p>
    <div class="placeholder">
      <p>Image grid, annotation editor, and review queue will be added in later phases.</p>
    </div>
  </div>
</template>

<style scoped>
.dataset-view {
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
