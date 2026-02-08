<script setup lang="ts">
/**
 * Toast notification container.
 *
 * Renders active toasts from the notification store in a fixed
 * position at the top-right of the viewport. Each toast can be
 * dismissed manually or auto-dismisses after 5 seconds.
 */
import { useNotificationStore } from '@/stores/notifications'

const notificationStore = useNotificationStore()
</script>

<template>
  <div class="toast-container">
    <TransitionGroup name="toast">
      <div
        v-for="toast in notificationStore.toasts"
        :key="toast.id"
        class="toast"
        :class="`toast--${toast.level}`"
      >
        <div class="toast-content">
          <strong>{{ toast.title }}</strong>
          <p>{{ toast.message }}</p>
        </div>
        <button
          class="toast-dismiss"
          @click="notificationStore.dismiss(toast.id)"
          aria-label="Dismiss"
        >
          &times;
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.toast-container {
  position: fixed;
  top: 1rem;
  right: 1rem;
  z-index: 2000;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  max-width: 360px;
}

.toast {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  border-radius: 6px;
  background: var(--color-bg-surface);
  border-left: 4px solid var(--color-text-muted);
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
}

.toast--info { border-left-color: var(--color-accent); }
.toast--success { border-left-color: var(--color-success); }
.toast--warning { border-left-color: var(--color-warning); }
.toast--error { border-left-color: var(--color-error); }

.toast-content {
  flex: 1;
}

.toast-content strong {
  display: block;
  font-size: 0.875rem;
}

.toast-content p {
  margin: 0.25rem 0 0;
  font-size: 0.8125rem;
  color: var(--color-text-muted);
}

.toast-dismiss {
  background: none;
  border: none;
  font-size: 1.25rem;
  cursor: pointer;
  color: var(--color-text-muted);
  line-height: 1;
}

/* Transition animations */
.toast-enter-active { transition: all 0.3s ease; }
.toast-leave-active { transition: all 0.2s ease; }
.toast-enter-from { opacity: 0; transform: translateX(1rem); }
.toast-leave-to { opacity: 0; transform: translateX(1rem); }
</style>
