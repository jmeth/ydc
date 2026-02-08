<script setup lang="ts">
/**
 * Scan mode view.
 *
 * Hosts inference video player, prompt editor, capture controls,
 * and real-time detection/capture stats. Placeholder until Phase 4-5.
 */
</script>

<template>
  <div class="scan-view">
    <h1>Scan Mode</h1>
    <p>Run YOLO inference on a video feed and auto-capture frames for your dataset.</p>
    <div class="placeholder">
      <p>Video player, prompt editor, and capture controls will be added in later phases.</p>
    </div>
  </div>
</template>

<style scoped>
.scan-view {
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
