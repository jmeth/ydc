<script setup lang="ts">
/**
 * Main navigation bar.
 *
 * Tab-style navigation between the four application modes.
 * Active state syncs with vue-router.
 */
</script>

<template>
  <nav class="app-nav">
    <RouterLink to="/scan" class="nav-link">Scan</RouterLink>
    <RouterLink to="/dataset" class="nav-link">Dataset</RouterLink>
    <RouterLink to="/train" class="nav-link">Train</RouterLink>
    <RouterLink to="/model" class="nav-link">Model</RouterLink>
  </nav>
</template>

<style scoped>
.app-nav {
  display: flex;
  gap: 0;
  background: var(--color-bg-nav);
  border-bottom: 1px solid var(--color-border);
}

.nav-link {
  padding: 0.625rem 1.25rem;
  text-decoration: none;
  color: var(--color-text-muted);
  font-size: 0.875rem;
  font-weight: 500;
  border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
}

.nav-link:hover {
  color: var(--color-text);
}

.nav-link.router-link-active {
  color: var(--color-accent);
  border-bottom-color: var(--color-accent);
}
</style>
