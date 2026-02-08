<script setup lang="ts">
/**
 * Persistent banner notification container.
 *
 * Renders active banners from the notification store at the top of
 * the viewport, below the header. Banners persist until manually
 * dismissed by the user.
 */
import { useNotificationStore } from '@/stores/notifications'

const notificationStore = useNotificationStore()
</script>

<template>
  <div class="banner-container">
    <TransitionGroup name="banner">
      <div
        v-for="banner in notificationStore.banners"
        :key="banner.id"
        class="banner"
        :class="`banner--${banner.level}`"
        role="alert"
      >
        <div class="banner-content">
          <strong>{{ banner.title }}</strong>
          <span>{{ banner.message }}</span>
        </div>
        <button
          class="banner-dismiss"
          @click="notificationStore.dismiss(banner.id)"
          aria-label="Dismiss"
        >
          &times;
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.banner-container {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1500;
  display: flex;
  flex-direction: column;
}

.banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.625rem 1rem;
  font-size: 0.875rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

.banner--info {
  background: var(--color-accent);
  color: #fff;
}

.banner--success {
  background: var(--color-success);
  color: #fff;
}

.banner--warning {
  background: var(--color-warning);
  color: #1a1a1a;
}

.banner--error {
  background: var(--color-error);
  color: #fff;
}

.banner-content {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
}

.banner-content strong {
  white-space: nowrap;
}

.banner-dismiss {
  background: none;
  border: none;
  font-size: 1.25rem;
  cursor: pointer;
  color: inherit;
  opacity: 0.8;
  line-height: 1;
  padding: 0 0.25rem;
}

.banner-dismiss:hover {
  opacity: 1;
}

/* Transition animations */
.banner-enter-active { transition: all 0.3s ease; }
.banner-leave-active { transition: all 0.2s ease; }
.banner-enter-from { opacity: 0; transform: translateY(-100%); }
.banner-leave-to { opacity: 0; transform: translateY(-100%); }
</style>
