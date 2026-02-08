<script setup lang="ts">
import AppHeader from '@/components/common/AppHeader.vue'
import AppNav from '@/components/common/AppNav.vue'
import StatusBar from '@/components/common/StatusBar.vue'
import Toast from '@/components/common/Toast.vue'
</script>

<template>
  <div class="app">
    <AppHeader />
    <AppNav />
    <main class="app-content">
      <RouterView />
    </main>
    <StatusBar />
    <Toast />
  </div>
</template>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: var(--color-bg);
  color: var(--color-text);
}

.app-content {
  flex: 1;
  padding: 1rem;
}
</style>
